#!/usr/bin/env python3
"""
run_autoets_cv.py — Walk-forward CV with AutoETS at horizon h=4.

What changed vs the h=1 version
-------------------------------
- Each fold now produces FOUR predictions (h=1, h=2, h=3, h=4) instead of one.
- The last three fold positions are dropped because they can't yield four
  actual future quarters to evaluate against.
- Config selection on inner folds uses the MEAN absolute error across all
  four horizons.  This matches the deployment task: forecasting a year out.
- Output schema gains `horizon` and `origin_date` columns — required by
  evaluation_method.py to align this output with other model families.

How the index arithmetic works
------------------------------
For a series of length N with outer_start = M, a fold at origin position p:
    training data    = ts.iloc[:p+1]            (indices 0..p inclusive)
    origin_date      = ts.index[p]
    target indices   = p+1, p+2, p+3, p+4
    target_dates h=1..4 = ts.index[p+1..p+4]
We need ts.index[p+4] to exist → latest viable origin is p = N - 4 - 1 = N - 5.
So origin position p ranges from M to N-5 inclusive, giving N - M - 4 folds.

Usage
-----
    python run_autoets_cv.py                    # all segments
    python run_autoets_cv.py --dry-run          # config only
    python run_autoets_cv.py --segment T001081  # one segment
    python run_autoets_cv.py --workers 10       # override workers
"""

import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import json
import warnings
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

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
HORIZON       = 4               # forecast horizon (deployment-relevant)
N_WORKERS     = 10
OUTPUT_DIR    = Path("cv_output")
covid_start   = pd.Timestamp('2020-03-31')
covid_end     = pd.Timestamp('2022-06-30')
MIN_TRAIN_FRAC = 0.45           # minimum training fraction of series length
MIN_TRAIN_ABS  = 20             # absolute minimum training quarters
MIN_HISTORY    = 33             # SHARED with run_stl_ets_cv.py and run_chronos_cv.py
                                # — ensures the three scripts process the same sectors
INNER_FRAC    = 0.6             # fraction of folds for selection
SEED          = 42

# ETS specs: Multiplicative-error only (appropriate for positive rates)
# Format: (label, model_str, damped)
ETS_SPECS = [
    ("MNN",  "MNN", False), ("MAN",  "MAN", False), ("MAdN", "MAN", True),
    ("MNA",  "MNA", False), ("MAA",  "MAA", False), ("MAdA", "MAA", True),
    ("MNM",  "MNM", False), ("MAM",  "MAM", False), ("MAdM", "MAM", True),
]
ETS_LABELS = [s[0] for s in ETS_SPECS]


# =============================================================================
# COVID CORRECTIONS
# =============================================================================

def build_correction_variants(df_ts):
    """Build COVID correction variants (non-winsorized only)."""
    mask = (df_ts.index >= covid_start) & (df_ts.index <= covid_end)

    def _smean(ts, b):
        out = ts.copy()
        for d in ts.index[mask]:
            sq = ts.index[(ts.index.quarter == d.quarter) & (~mask)]
            refs = sq[sq < covid_start][-b:]
            if len(refs) > 0:
                out[d] = ts[refs].mean()
        return out

    v = {}
    v["no_correction"] = df_ts.copy()
    for b in [2, 3]:
        v[f"seasonal_mean|b{b}"] = _smean(df_ts, b)

    for label, ts in v.items():
        assert ts.index.equals(df_ts.index), f"Index mismatch: {label}"
        assert not ts.isna().any(), f"NaN in correction: {label}"

    return v


def winsorize_train(ts_train):
    """Winsorize using only training window quantiles (no look-ahead)."""
    lo = ts_train.quantile(0.05)
    hi = ts_train.quantile(0.95)
    return ts_train.clip(lower=lo, upper=hi)


def check_seasonal_stability(df_ts, threshold=0.3):
    """Compare seasonal indices pre/post COVID. Returns warning string or None."""
    pre = df_ts[df_ts.index < covid_start]
    post = df_ts[df_ts.index > covid_end]
    if len(pre) < 8 or len(post) < 8:
        return None

    def _seasonal_index(ts):
        qmeans = ts.groupby(ts.index.quarter).mean()
        return qmeans / qmeans.mean()

    idx_pre = _seasonal_index(pre)
    idx_post = _seasonal_index(post)
    diff = (idx_post - idx_pre).abs()
    max_shift = diff.max()
    worst_q = diff.idxmax()

    if max_shift > threshold:
        return (f"Seasonal shift: Q{worst_q} index changed by {max_shift:.3f} "
                f"(pre={idx_pre[worst_q]:.3f}, post={idx_post[worst_q]:.3f})")
    return None


# =============================================================================
# WORKER — h=4 walk-forward CV for one correction
# =============================================================================

def run_cv_for_correction(corr_label, corr_values, corr_index,
                          orig_values, outer_start, season_length,
                          ets_specs, ets_labels, seed, horizon):
    """
    Walk-forward CV at h=`horizon` for one correction (base + winsorized) × all ETS specs.

    The 'origin position' p represents the index of the last training observation.
    Training data is ts.iloc[:p+1]; we predict steps p+1..p+horizon.

    Latest viable origin: p = len(ts) - horizon - 1
    First origin:         p = outer_start - 1   (so first training has outer_start obs)

    Returns {variant_label: {fold_results: [...]}}.
    Each fold_result has 'preds' as dict {spec_label: array of `horizon` predictions}.
    """
    np.random.seed(seed)
    warnings.filterwarnings("ignore")
    ts = pd.Series(corr_values, index=corr_index)

    # Origin positions range from (outer_start - 1) to (len(ts) - horizon - 1) inclusive.
    # First origin trains on ts.iloc[:outer_start]  (outer_start observations).
    # Last origin trains on ts.iloc[:len(ts) - horizon]  and predicts the last horizon points.
    first_origin = outer_start - 1
    last_origin  = len(ts) - horizon - 1
    if last_origin < first_origin:
        return {}

    n_folds = last_origin - first_origin + 1
    all_results = {}

    for is_winsorized in [False, True]:
        variant_label = (f"{corr_label}|winsorized" if is_winsorized
                         else corr_label)
        fold_results = []

        for fold_idx in range(n_folds):
            origin_pos = first_origin + fold_idx          # last training index
            train_end  = origin_pos + 1                   # exclusive
            train_ts   = ts.iloc[:train_end].copy()
            if is_winsorized:
                train_ts = winsorize_train(train_ts)

            sf_df = pd.DataFrame({
                "unique_id": variant_label,
                "ds": train_ts.index,
                "y": train_ts.values,
            })

            # Pull `horizon` actuals (always positions origin_pos+1 .. origin_pos+horizon).
            actuals = orig_values[origin_pos + 1 : origin_pos + 1 + horizon].astype(float)
            target_dates = ts.index[origin_pos + 1 : origin_pos + 1 + horizon]
            origin_date  = ts.index[origin_pos]

            # Defensive: we may have lost trailing rows somehow
            if len(actuals) != horizon:
                continue

            preds = {}
            for spec_label, spec_model, spec_damped in ets_specs:
                try:
                    m = AutoETS(season_length=season_length, model=spec_model,
                                damped=spec_damped, alias=spec_label)
                    sf = StatsForecast(models=[m], freq="QE", n_jobs=1)
                    fc = sf.forecast(df=sf_df, h=horizon)
                    if spec_label in fc.columns:
                        pvals = fc[spec_label].values.astype(float)
                        # statsforecast returns horizon rows; align defensively
                        if len(pvals) >= horizon:
                            arr = pvals[:horizon]
                            arr = np.where(np.isfinite(arr), arr, np.nan)
                        else:
                            arr = np.full(horizon, np.nan)
                        preds[spec_label] = arr
                    else:
                        preds[spec_label] = np.full(horizon, np.nan)
                except Exception:
                    preds[spec_label] = np.full(horizon, np.nan)

            fold_results.append({
                "fold_pos":     fold_idx,
                "origin_pos":   origin_pos,
                "origin_date":  origin_date,
                "target_dates": target_dates,
                "actuals":      actuals,
                "preds":        preds,
            })

        all_results[variant_label] = {
            "correction":   variant_label,
            "fold_results": fold_results,
        }

    return all_results


# =============================================================================
# SEGMENT PROCESSING
# =============================================================================

def process_segment(sbi_code, df_ts, pool, ets_specs, ets_labels):
    """
    Full h=`HORIZON` CV for one SBI segment.

    Selection on inner folds (fold_pos < n_inner). Aggregation: MEAN MAE
    across all `HORIZON` horizons gives a single scalar per (correction, spec)
    for comparison and selection.

    Returns:
        cv_records:       winner-only fold-level records (4 rows per fold)
        all_pred_records: full prediction matrix (all configs × all folds × 4 horizons)
        config:           dict or None — winning config metadata
    """
    variants = build_correction_variants(df_ts)
    orig = df_ts.values.ravel().astype(np.float64)

    outer_start = max(MIN_TRAIN_ABS, int(len(df_ts) * MIN_TRAIN_FRAC))

    # Folds: see header docstring. We need 5 viable folds to maintain the
    # inner/outer split semantics; outer_start might leave too few.
    first_origin = outer_start - 1
    last_origin  = len(df_ts) - HORIZON - 1
    n_total_folds = max(0, last_origin - first_origin + 1)

    if n_total_folds < 5:
        print(f"    ⚠️  {sbi_code}: only {n_total_folds} folds with "
              f"outer_start={outer_start}, h={HORIZON}. Skipping.")
        return [], [], None

    seg_start = time.time()
    n_inner = max(1, int(n_total_folds * INNER_FRAC))
    n_outer = n_total_folds - n_inner

    print(f"      Training start: {outer_start} quarters | "
          f"Folds: {n_total_folds} = {n_inner} inner + {n_outer} outer | "
          f"Horizons: {HORIZON}")

    seasonal_warn = check_seasonal_stability(df_ts)
    if seasonal_warn:
        print(f"    ⚠️  SEASONAL: {seasonal_warn}")

    # Dispatch
    futures = {}
    for label, ts_c in variants.items():
        f = pool.submit(
            run_cv_for_correction, label,
            ts_c.values.ravel().astype(np.float64), ts_c.index,
            orig, outer_start, 4, ets_specs, ets_labels, SEED, HORIZON,
        )
        futures[f] = label

    corr_results = {}
    for f in as_completed(futures):
        base_label = futures[f]
        try:
            corr_results.update(f.result())
        except Exception as exc:
            print(f"    ⚠️  [{base_label}]: {exc}")
            for suffix in ["", "|winsorized"]:
                lbl = f"{base_label}{suffix}" if suffix else base_label
                corr_results[lbl] = {"correction": lbl, "fold_results": [],
                                     "error": str(exc)}

    # Diagnostic
    total_valid = 0
    total_nan   = 0
    for label, wr in corr_results.items():
        if wr.get("error"):
            continue
        for fr in wr.get("fold_results", []):
            for spec, arr in fr["preds"].items():
                total_valid += int(np.isfinite(arr).sum())
                total_nan   += int(np.isnan(arr).sum())
    print(f"      Diagnostics: {total_valid} valid horizon-preds, "
          f"{total_nan} NaN horizon-preds across all variants")

    # --- Build FULL prediction matrix (all configs, all folds, all horizons) ---
    all_pred_records = []
    for label, wr in corr_results.items():
        if wr.get("error") or not wr["fold_results"]:
            continue
        for fr in wr["fold_results"]:
            is_outer = fr["fold_pos"] >= n_inner
            for spec in ets_labels:
                arr = fr["preds"].get(spec, np.full(HORIZON, np.nan))
                for h_idx in range(HORIZON):
                    horizon = h_idx + 1
                    pred    = arr[h_idx]
                    actual  = float(fr["actuals"][h_idx])
                    ok      = np.isfinite(pred)
                    all_pred_records.append({
                        "sbi_code":    sbi_code,
                        "correction":  label,
                        "model_spec":  spec,
                        "origin_date": fr["origin_date"],
                        "target_date": fr["target_dates"][h_idx],
                        "horizon":     horizon,
                        "actual":      actual,
                        "pred":        float(pred) if ok else np.nan,
                        "abs_error":   float(abs(actual - pred)) if ok else np.nan,
                        "fold_set":    "outer" if is_outer else "inner",
                        "fold_pos":    fr["fold_pos"],
                    })

    # --- Select best (correction, spec) on inner folds using mean MAE
    #     averaged across all HORIZON horizons ---
    best_inner_mae  = np.inf
    best_corr_label = None
    best_model_spec = None

    for label, wr in corr_results.items():
        if wr.get("error") or not wr["fold_results"]:
            continue
        for spec in ets_labels:
            errs = []
            for fr in wr["fold_results"]:
                if fr["fold_pos"] >= n_inner:
                    continue
                arr = fr["preds"].get(spec, np.full(HORIZON, np.nan))
                actuals = fr["actuals"]
                for h_idx in range(HORIZON):
                    if np.isfinite(arr[h_idx]):
                        errs.append(abs(actuals[h_idx] - arr[h_idx]))
            if errs:
                mae_val = float(np.mean(errs))
                if mae_val < best_inner_mae:
                    best_inner_mae  = mae_val
                    best_corr_label = label
                    best_model_spec = spec

    # --- Winner-only records (4 rows per fold) ---
    cv_records = []
    outer_errors = []

    if best_corr_label is not None:
        wr = corr_results[best_corr_label]
        for fr in wr["fold_results"]:
            arr = fr["preds"].get(best_model_spec, np.full(HORIZON, np.nan))
            is_outer = fr["fold_pos"] >= n_inner
            for h_idx in range(HORIZON):
                horizon = h_idx + 1
                pred = arr[h_idx]
                actual = float(fr["actuals"][h_idx])
                ok = np.isfinite(pred)
                cv_records.append({
                    "sbi_code":    sbi_code,
                    "correction":  best_corr_label,
                    "model_spec":  best_model_spec,
                    "origin_date": fr["origin_date"],
                    "target_date": fr["target_dates"][h_idx],
                    "horizon":     horizon,
                    "actual":      actual,
                    "pred":        float(pred) if ok else np.nan,
                    "abs_error":   float(abs(actual - pred)) if ok else np.nan,
                    "fold_set":    "outer" if is_outer else "inner",
                    "fold_pos":    fr["fold_pos"],
                })
                if is_outer and ok:
                    outer_errors.append(abs(actual - pred))

    seg_min = (time.time() - seg_start) / 60
    outer_mae = float(np.mean(outer_errors)) if outer_errors else np.nan

    config = None
    if best_corr_label is not None:
        df_ts_corrected = variants[best_corr_label.replace("|winsorized", "")].copy()
        config = {
            "sbi_code":            sbi_code,
            "winning_correction":  best_corr_label,
            "best_model_spec":     best_model_spec,
            "inner_mae":           best_inner_mae,
            "outer_mae":           outer_mae,
            "n_inner_folds":       n_inner,
            "n_outer_folds":       n_outer,
            "horizon":             HORIZON,
            "seasonal_shift_warning": seasonal_warn,
            "df_ts_corrected":     df_ts_corrected,
            "seg_minutes":         seg_min,
        }

    return cv_records, all_pred_records, config


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=f"AutoETS h={HORIZON} walk-forward CV")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--segment", type=str, default=None,
                        help="If set, only process this SBI code")
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

    # Output-filename suffix lets the two data-source runs coexist on disk
    out_suffix = "_silver" if args.data_source == "silver" else ""

    print("=" * 60)
    print(f"  AUTOETS WALK-FORWARD CV — HORIZON = {HORIZON}")
    print(f"  Data source: {args.data_source.upper()}"
          f"{'  (sensitivity run; outputs *_silver.parquet)' if out_suffix else ''}")
    print("=" * 60)

    # --- Load per-sector target series ---
    # Gold: imputed values, identical inputs to Pipeline.
    # Silver: raw observations + contiguous-tail extraction for sectors with gaps.
    if args.data_source == "gold":
        print("  Loading target series from gold table...")
        segments_all = load_target_series_from_gold(
            gold_db_path=DIR_DB_GOLD,
            min_history=MIN_HISTORY,
            verbose=True,
        )
    else:
        print("  Loading target series from silver table (contiguous-tail mode)...")
        segments_all = load_target_series_from_silver(
            silver_db_path=DIR_DB_SILVER,
            min_history=MIN_HISTORY,
            verbose=True,
        )

    # Snap each series to quarterly frequency and filter to --segment if set
    segments = {}
    for c, ts in segments_all.items():
        if args.segment and c != args.segment:
            continue
        ts = ts.asfreq("QE")
        if ts.isna().any():
            print(f"  ⚠️  Skipping {c}: unexpected NaN ({ts.isna().sum()} NaN)")
            continue
        if len(ts) < MIN_HISTORY:
            print(f"  ⚠️  Skipping {c} ({len(ts)} obs, need {MIN_HISTORY})")
            continue
        segments[c] = ts

    if not segments:
        print("  No valid segments.")
        return

    n_base = 3
    n_total = n_base * 2
    print(f"\n  Segments: {len(segments)} | ETS specs: {len(ETS_SPECS)} | "
          f"Correction variants: {n_total}")

    if args.dry_run:
        print("\n  --dry-run: exiting.\n")
        return

    # --- Run ---
    wall_start = time.time()
    all_records = []
    all_pred_records = []
    all_configs = {}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (code, df_ts) in enumerate(segments.items()):
            print(f"\n  SEGMENT {i+1}/{len(segments)}: {code}")
            records, pred_records, config = process_segment(
                code, df_ts, pool, ETS_SPECS, ETS_LABELS
            )
            all_records.extend(records)
            all_pred_records.extend(pred_records)
            if config:
                all_configs[code] = config

    total_min = (time.time() - wall_start) / 60
    warnings.filterwarnings("default")

    df_cv = pd.DataFrame(all_records)
    df_all = pd.DataFrame(all_pred_records)

    if len(df_cv) == 0:
        print("  No results.")
        return

    # --- Save ---
    p1 = OUTPUT_DIR / f"autoets_cv_results{out_suffix}.parquet"
    p2 = OUTPUT_DIR / f"autoets_cv_all_predictions{out_suffix}.parquet"
    p3_json = OUTPUT_DIR / f"autoets_best_configs{out_suffix}.json"

    df_cv.to_parquet(p1, index=False);   print(f"\n  ✅ {p1}")
    df_all.to_parquet(p2, index=False);  print(f"  ✅ {p2} ({len(df_all)} rows)")

    configs_json = {}
    for code, cfg in all_configs.items():
        cfg.pop("df_ts_corrected", None)
        configs_json[code] = {
            k: (float(v) if isinstance(v, np.floating)
                else int(v) if isinstance(v, np.integer) else v)
            for k, v in cfg.items()
        }
    with open(p3_json, "w") as f:
        json.dump(configs_json, f, indent=2, default=str)
    print(f"  ✅ {p3_json}")

    # ----- Canonical-schema export for evaluation_method.py -----
    # Keep only OUTER-fold rows from the winning (correction, model_spec) per
    # sector — that's the unbiased estimate of out-of-sample performance.
    #
    # IMPORTANT: model_name is "AutoETS" (family-level), not per-spec.  Each
    # sector may select a different winning spec (MAdM, MNA, etc.) on inner
    # folds, but for the cross-model comparison we treat AutoETS as one
    # METHOD that internally chooses its spec.  Per-sector spec details are
    # preserved in autoets_cv_results.parquet / autoets_best_configs.json
    # for diagnostic inspection.  Using per-spec names here would create
    # multiple sparse model_names (one per spec) that break friedman_nemenyi
    # because each spec only has predictions for the sectors that selected it.
    winner_mask = df_cv["fold_set"] == "outer"
    winner_outer = df_cv[winner_mask].dropna(subset=["pred"]).copy()
    canonical = pd.DataFrame({
        "model_name":   "AutoETS",
        "sector_code":  winner_outer["sbi_code"].astype(str),
        "origin_date":  pd.to_datetime(winner_outer["origin_date"]),
        "target_date":  pd.to_datetime(winner_outer["target_date"]),
        "horizon":      winner_outer["horizon"].astype(int),
        "y_true":       winner_outer["actual"].astype(float),
        "y_pred":       winner_outer["pred"].astype(float),
        "y_lower_80":   np.nan,
        "y_upper_80":   np.nan,
        "y_lower_95":   np.nan,
        "y_upper_95":   np.nan,
    })
    p_canon = OUTPUT_DIR / f"autoets_predictions{out_suffix}.parquet"
    canonical.to_parquet(p_canon, index=False)
    print(f"  ✅ {p_canon} (canonical, {len(canonical)} rows)")
    print(f"\n  Total: {total_min:.1f} min")


if __name__ == "__main__":
    main()
