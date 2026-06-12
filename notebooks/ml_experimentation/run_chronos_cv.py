#!/usr/bin/env python3
"""
run_chronos_cv.py — Chronos-Bolt walk-forward backtest at horizon h=4.

Converts the foundational-models notebook into a standalone script with
h=HORIZON=4 to match the AutoETS and STL+ETS scripts and the Pipeline's
deployment-relevant forecast horizon.

What changed vs the h=3 notebook
--------------------------------
- HORIZON = 4 throughout (was 3).
- Output parquet shaped exactly for evaluation_method.load_chronos_predictions:
  columns include sbi_code, origin_date, target_date, horizon, y_true,
  y_hat (median), y_mean, and q10..q90 quantile bounds (only the quantiles
  Chronos-Bolt was actually trained on — see SETTINGS).
- Standalone script (no notebook plumbing) for reproducible re-runs.

Origin arithmetic (same as the notebook's logic, applied at h=4)
----------------------------------------------------------------
For a series of length n with horizon h and n_origins origins:
    last_origin_pos  = n - h - 1               # 0-indexed
    first_origin_pos = last_origin_pos - (n_origins - 1)
At each origin pos p:
    context  = series.iloc[: p + 1].values     # strictly past, inclusive of origin
    actuals  = series.iloc[p + 1 : p + 1 + h].values
    target_dates = series.index[p + 1 : p + 1 + h]
We need first_origin_pos >= 0 → n ≥ n_origins + h.

Usage
-----
    python run_chronos_cv.py                    # all eligible sectors
    python run_chronos_cv.py --segment T001081  # one sector
    python run_chronos_cv.py --dry-run
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoid hf tokenizer warning

import sys
import time
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DIR_DB_GOLD, DIR_DB_SILVER
from src.utils.m_gold_target_loader import (
    load_target_series_from_gold,
    load_target_series_from_silver,
)

# =============================================================================
# SETTINGS
# =============================================================================
HORIZON       = 4                                 # deployment-relevant horizon
N_ORIGINS     = 12                                # walk-forward test origins
# Chronos-Bolt was trained ONLY on quantiles {0.1, 0.2, ..., 0.9}.  Requesting
# anything outside that range (e.g. 0.025 or 0.975 for a 95% PI) makes the
# pipeline silently clip to the nearest trained level, so the returned "0.025
# quantile" is actually the 0.1 quantile and the "0.975 quantile" is the 0.9
# quantile.  That would produce a fake 95% PI identical to the 80% PI, which
# is misleading.  We only request the trained quantiles and leave the 95% PI
# columns in the canonical export as NaN.
QUANTILES     = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEASON_LENGTH = 4
MIN_HISTORY   = 33                                # SHARED with run_autoets_cv.py
                                                  # and run_stl_ets_cv.py — ensures
                                                  # all three scripts process the
                                                  # same sectors

MODEL_ID  = "amazon/chronos-bolt-base"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.bfloat16 if DEVICE == "cuda" else torch.float32

COVID_START = pd.Timestamp("2020-03-31")
COVID_END   = pd.Timestamp("2022-06-30")

OUTPUT_DIR  = Path("cv_output")
SEED        = 42


# =============================================================================
# FORECAST WRAPPER
# =============================================================================

def forecast_chronos(pipeline, context: np.ndarray,
                     horizon: int = HORIZON,
                     quantile_levels=QUANTILES) -> dict:
    """Single-series forecast with Chronos-Bolt.  Returns median/quantiles/mean arrays."""
    if context.ndim != 1:
        raise ValueError(f"context must be 1D, got shape {context.shape}")
    if np.isnan(context).any():
        raise ValueError("context contains NaNs; clean before calling.")
    if len(context) < 8:
        raise ValueError(f"context too short ({len(context)} obs); need ≥8.")

    ctx_tensor = torch.tensor(context, dtype=torch.float32)
    q_tensor, mean_tensor = pipeline.predict_quantiles(
        inputs=ctx_tensor, prediction_length=horizon,
        quantile_levels=quantile_levels,
    )
    q_arr    = q_tensor[0].cpu().float().numpy()
    mean_arr = mean_tensor[0].cpu().float().numpy()
    quantiles = {q: q_arr[:, i] for i, q in enumerate(quantile_levels)}
    median_idx = quantile_levels.index(0.5) if 0.5 in quantile_levels else None
    median_arr = q_arr[:, median_idx] if median_idx is not None else mean_arr
    return {"median": median_arr, "quantiles": quantiles, "mean": mean_arr}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_series(data_source: str = "gold") -> dict:
    """Load per-sector target series.

    Parameters
    ----------
    data_source : {"gold", "silver"}
        "gold" reads imputed target values from master_data_ml_preprocessed
        (matches what Pipeline trains on).  "silver" reads raw observations
        from 80072ned_silver with contiguous-tail extraction for the 13
        reorganized sectors (no fabricated values).

    Returns
    -------
    dict {sbi_code: pd.Series}
        Each Series is indexed by quarter-end timestamps.
    """
    if data_source == "gold":
        print("  Loading target series from gold table...")
        return load_target_series_from_gold(
            gold_db_path=DIR_DB_GOLD,
            min_history=MIN_HISTORY,
            verbose=True,
        )
    else:
        print("  Loading target series from silver table (contiguous-tail mode)...")
        return load_target_series_from_silver(
            silver_db_path=DIR_DB_SILVER,
            min_history=MIN_HISTORY,
            verbose=True,
        )


# =============================================================================
# BACKTEST
# =============================================================================

def walk_forward_backtest(pipeline, series: pd.Series, sbi_code: str,
                          n_origins: int = N_ORIGINS,
                          horizon: int = HORIZON,
                          quantile_levels=QUANTILES):
    """Walk-forward backtest for one sector at h=horizon."""
    s = series.dropna().sort_index()
    n = len(s)

    last_origin_pos  = n - horizon - 1
    first_origin_pos = last_origin_pos - (n_origins - 1)
    if first_origin_pos < 0:
        raise ValueError(
            f"[{sbi_code}] insufficient history: {n} obs, need ≥{n_origins + horizon}"
        )

    rows = []
    for origin_pos in range(first_origin_pos, last_origin_pos + 1):
        context_arr  = s.iloc[: origin_pos + 1].values
        actuals_arr  = s.iloc[origin_pos + 1 : origin_pos + 1 + horizon].values
        target_dates = s.index[origin_pos + 1 : origin_pos + 1 + horizon]
        origin_date  = s.index[origin_pos]

        out = forecast_chronos(pipeline, context_arr, horizon=horizon,
                               quantile_levels=quantile_levels)
        for h in range(horizon):
            row = {
                "sbi_code":      sbi_code,
                "origin_date":   origin_date,
                "target_date":   target_dates[h],
                "horizon":       h + 1,
                "y_true":        float(actuals_arr[h]),
                "y_hat":         float(out["median"][h]),
                "y_mean":        float(out["mean"][h]),
                "context_len":   origin_pos + 1,
                "in_covid":      bool(COVID_START <= target_dates[h] <= COVID_END),
            }
            for q in quantile_levels:
                row[f"q{int(q*100):02d}"] = float(out["quantiles"][q][h])
            rows.append(row)
    return rows


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=f"Chronos h={HORIZON} walk-forward backtest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--segment", type=str, default=None)
    parser.add_argument(
        "--data-source", choices=["gold", "silver"], default="gold",
        help="Where to read target series from.  "
             "'gold' (default): imputed values from master_data_ml_preprocessed, "
             "matching what Pipeline trains on.  "
             "'silver': raw observations from 80072ned_silver with contiguous-tail "
             "extraction for the 13 reorganized sectors (no fabricated values).  "
             "Use 'silver' for sensitivity analysis; outputs are written with a "
             "'_silver' suffix to avoid overwriting the primary gold run.",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Output-filename suffix lets the two data-source runs coexist on disk
    out_suffix = "_silver" if args.data_source == "silver" else ""

    print("=" * 60)
    print(f"  CHRONOS-BOLT BACKTEST — HORIZON = {HORIZON}")
    print(f"  Data source: {args.data_source.upper()}"
          f"{'  (sensitivity run; outputs *_silver.parquet)' if out_suffix else ''}")
    print(f"  Device: {DEVICE} | dtype: {DTYPE}")
    print("=" * 60)

    series_dict = load_series(data_source=args.data_source)
    total_sectors = len(series_dict)
    print(f"\n  Eligible: {total_sectors} sectors")

    if args.dry_run:
        if series_dict:
            sample = next(iter(series_dict.values()))
            print(f"    example sector range: "
                  f"{sample.index.min().date()} → {sample.index.max().date()} "
                  f"({len(sample)} quarters)")
        return

    print(f"\n  Loading {MODEL_ID} on {DEVICE} ...")
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID, device_map=DEVICE, dtype=DTYPE,
    )
    print("  Model ready.\n")

    warnings.filterwarnings("ignore")
    all_rows = []
    counter = 0
    wall_start = time.time()
    for sbi_code, series in series_dict.items():
        if args.segment and sbi_code != args.segment:
            continue
        counter += 1
        print(f"  [{counter:>3}/{total_sectors}] {sbi_code} ...",
              end="", flush=True)
        try:
            rows = walk_forward_backtest(
                pipeline, series, str(sbi_code),
            )
            all_rows.extend(rows)
            print(f" {len(rows)} predictions")
        except Exception as e:
            print(f" SKIPPED ({type(e).__name__}: {e})")

    total_min = (time.time() - wall_start) / 60
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("  No results.")
        return

    df["error"]     = df["y_hat"] - df["y_true"]
    df["abs_error"] = df["error"].abs()

    p_out = OUTPUT_DIR / f"chronos_backtest{out_suffix}.parquet"
    df.to_parquet(p_out, index=False)
    print(f"\n  ✅ {p_out} ({len(df)} rows)")

    # ----- Canonical-schema export for evaluation_method.py -----
    # Chronos-Bolt's trained quantiles only support an 80% PI (q10..q90).
    # A true 95% PI would require q025..q975, which the model doesn't
    # produce, so we leave those columns NaN rather than fabricating bounds
    # by clipping to the nearest trained quantile.
    canonical = pd.DataFrame({
        "model_name":   "Chronos_Bolt",
        "sector_code":  df["sbi_code"].astype(str),
        "origin_date":  pd.to_datetime(df["origin_date"]),
        "target_date":  pd.to_datetime(df["target_date"]),
        "horizon":      df["horizon"].astype(int),
        "y_true":       df["y_true"].astype(float),
        "y_pred":       df["y_hat"].astype(float),     # median forecast (q50)
        "y_lower_80":   df["q10"].astype(float),
        "y_upper_80":   df["q90"].astype(float),
        "y_lower_95":   np.nan,
        "y_upper_95":   np.nan,
    })
    p_canon = OUTPUT_DIR / f"chronos_predictions{out_suffix}.parquet"
    canonical.to_parquet(p_canon, index=False)
    print(f"  ✅ {p_canon} (canonical, {len(canonical)} rows)")
    print(f"  Total: {total_min:.1f} min")


if __name__ == "__main__":
    main()
