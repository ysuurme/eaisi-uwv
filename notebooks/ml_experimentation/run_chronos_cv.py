#!/usr/bin/env python3
"""
run_chronos_cv.py — Chronos-Bolt walk-forward CV at horizon h=4, exporting every
correction method from a SINGLE CV pass.

This is the single-pass version: one `uv run python run_chronos_cv.py` loads the
model once, runs the nested walk-forward CV over all six COVID-correction
variants once per data source, and writes the canonical parquet for every
method — zero-shot, the per-sector correction-selecting method, and each fixed
variant — from those shared results. No per-mode re-runs.

Why single-pass
---------------
`process_segment` forecasts ALL six correction variants on every sector
regardless of which method you ultimately want, so the full (variant × fold ×
horizon) matrix already contains everything. Looping the script once per mode
re-ran the (expensive) Chronos forecasting 7×; here we compute it once and just
slice/select per method at export time.

Methods exported (all derived from the one CV pass)
---------------------------------------------------
  Chronos_Bolt              zero-shot (no_correction), outer folds.
                            -> chronos_predictions.parquet   (drop-in for the
                            notebook; labelled "zero-shot" there).
  Chronos_Bolt_corrected    per-sector inner-fold selection over all six
                            variants (the AutoETS analogue), outer folds.
                            -> chronos_predictions_corrected.parquet
  Chronos_Bolt_<variant>    each fixed correction variant (except plain
                            no_correction, which == zero-shot), outer folds.
                            -> chronos_predictions_fixed-<variant>.parquet
                            (only with --methods all)

Use --methods core to export just the two the notebook consumes
(Chronos_Bolt + Chronos_Bolt_corrected); --methods all (default) adds the five
fixed-variant diagnostic files.

Data source & tags (unchanged)
------------------------------
--data-source gold (default) | silver | both.  The silver run keeps the
"_silver" filename tag exactly as before, so gold and silver outputs coexist:
e.g. chronos_predictions_silver.parquet, chronos_predictions_corrected_silver.parquet.
--data-source both runs gold then silver in the same process (model loaded once).

Comparison hygiene baked in
---------------------------
- Corrections (build_correction_variants / winsorize_train) are lifted verbatim
  from run_autoets_cv.py and applied to the CONTEXT only; actuals are always the
  ORIGINAL (uncorrected) series, so every method is scored against the same
  y_true. model_comparison.ipynb computes the seasonal-naive MASE denominator
  from those shared aligned actuals, so a correction can only move a method's
  own numerator — never the shared scale.
- Origin arithmetic is identical to AutoETS (outer_start = max(MIN_TRAIN_ABS,
  MIN_TRAIN_FRAC*N); origins outer_start-1 .. N-h-1; INNER_FRAC inner/outer
  split), so the canonical parquets share (sector, target_date, horizon) keys.
- Chronos-Bolt was trained ONLY on quantiles {0.1..0.9}: q10/q90 -> a real 80%
  PI; the 95% PI columns stay NaN rather than fabricating bounds.

Usage
-----
    uv run python run_chronos_cv.py                       # gold, all methods
    uv run python run_chronos_cv.py --data-source silver  # silver, all methods
    uv run python run_chronos_cv.py --data-source both     # gold + silver, one process
    uv run python run_chronos_cv.py --methods core         # only the 2 notebook methods
    uv run python run_chronos_cv.py --segment T001081
    uv run python run_chronos_cv.py --dry-run
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoid hf tokenizer warning

import sys
import time
import json
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
# SETTINGS  — kept numerically identical to run_autoets_cv.py wherever shared
# =============================================================================
HORIZON        = 4               # deployment-relevant horizon
SEASON_LENGTH  = 4               # quarterly; the seasonal-naive MASE lag (used
                                 # downstream in model_comparison.ipynb)
OUTPUT_DIR     = Path("cv_output")

covid_start    = pd.Timestamp("2020-03-31")   # lowercase to match the lifted
covid_end      = pd.Timestamp("2022-06-30")   # AutoETS correction functions

MIN_TRAIN_FRAC = 0.45            # minimum training fraction of series length
MIN_TRAIN_ABS  = 20              # absolute minimum training quarters
MIN_HISTORY    = 33              # SHARED with run_autoets_cv.py / run_stl_ets_cv.py
INNER_FRAC     = 0.6             # fraction of folds for selection
SEED           = 42

# Chronos-Bolt was trained ONLY on quantiles {0.1, ..., 0.9}; see module docstring.
QUANTILES      = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

MODEL_ID       = "amazon/chronos-bolt-base"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.bfloat16 if DEVICE == "cuda" else torch.float32
BATCH_SIZE     = 32              # contexts per predict_quantiles call

# Legal correction-variant labels (must match build_correction_variants output).
_VALID_FIXED_LABELS = [
    f"{base}{suffix}"
    for base in ("no_correction", "seasonal_mean|b2", "seasonal_mean|b3")
    for suffix in ("", "|winsorized")
]


# =============================================================================
# COVID CORRECTIONS  — lifted verbatim from run_autoets_cv.py
# =============================================================================

def build_correction_variants(df_ts):
    """Build COVID correction variants (non-winsorized only).

    Look-ahead-safe: each replaced COVID-window quarter uses only same-quarter
    references strictly BEFORE covid_start.
    """
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
    """Winsorize using only training-window quantiles (no look-ahead).

    Applied per-fold to the growing context, exactly as AutoETS winsorizes each
    fold's training window. For Chronos this clips the context the model sees.
    """
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
# FORECAST WRAPPER  (batched, with per-context fallback)
# =============================================================================

def _predict_chunk(pipeline, chunk, horizon, quantile_levels):
    """Predict a chunk of 1-D context tensors.

    Tries a single batched list call (fast path); if the installed chronos
    version rejects a variable-length list batch, falls back to per-context
    single-tensor calls.  Either way returns (q_arr [B, H, Q], mean_arr [B, H]).
    Contexts are validated upstream, so genuine data errors still surface.
    """
    try:
        q_tensor, mean_tensor = pipeline.predict_quantiles(
            inputs=chunk, prediction_length=horizon,
            quantile_levels=quantile_levels,
        )
        return q_tensor.cpu().float().numpy(), mean_tensor.cpu().float().numpy()
    except Exception:
        q_list, m_list = [], []
        for t in chunk:
            q_t, m_t = pipeline.predict_quantiles(
                inputs=t, prediction_length=horizon,
                quantile_levels=quantile_levels,
            )
            q_list.append(q_t[0].cpu().float().numpy())
            m_list.append(m_t[0].cpu().float().numpy())
        return np.stack(q_list, axis=0), np.stack(m_list, axis=0)


def forecast_chronos_batch(pipeline, contexts, horizon=HORIZON,
                           quantile_levels=QUANTILES, batch_size=BATCH_SIZE):
    """Forecast a list of 1-D contexts with Chronos-Bolt in batches.

    Returns a list (aligned to `contexts`) of dicts with keys
    {"median", "mean", "quantiles"}.
    """
    tensors = []
    for ctx in contexts:
        if ctx.ndim != 1:
            raise ValueError(f"context must be 1D, got shape {ctx.shape}")
        if np.isnan(ctx).any():
            raise ValueError("context contains NaNs; clean before calling.")
        if len(ctx) < 8:
            raise ValueError(f"context too short ({len(ctx)} obs); need >=8.")
        tensors.append(torch.tensor(ctx, dtype=torch.float32))

    median_idx = quantile_levels.index(0.5) if 0.5 in quantile_levels else None

    results = []
    for start in range(0, len(tensors), batch_size):
        chunk = tensors[start:start + batch_size]
        q_arr, mean_arr = _predict_chunk(pipeline, chunk, horizon, quantile_levels)
        for i in range(len(chunk)):
            quantiles = {q: q_arr[i, :, j] for j, q in enumerate(quantile_levels)}
            median = (q_arr[i, :, median_idx] if median_idx is not None
                      else mean_arr[i])
            results.append({"median": median,
                            "mean": mean_arr[i],
                            "quantiles": quantiles})
    return results


# =============================================================================
# CV FOR ONE CORRECTION  — mirrors run_cv_for_correction (no ETS-spec loop)
# =============================================================================

def run_cv_for_correction(pipeline, corr_label, corr_series, orig_values,
                          outer_start, horizon=HORIZON,
                          quantile_levels=QUANTILES, batch_size=BATCH_SIZE):
    """Walk-forward CV at h=`horizon` for one correction (base + winsorized).

    Returns {variant_label: {"fold_results": [...]}}.  Actuals come from
    `orig_values` (the ORIGINAL, uncorrected series).
    """
    n = len(corr_series)
    idx = corr_series.index
    first_origin = outer_start - 1
    last_origin  = n - horizon - 1
    if last_origin < first_origin:
        return {}
    n_folds = last_origin - first_origin + 1

    all_results = {}
    for is_winsorized in [False, True]:
        variant_label = (f"{corr_label}|winsorized" if is_winsorized
                         else corr_label)

        contexts, meta = [], []
        for fold_idx in range(n_folds):
            origin_pos = first_origin + fold_idx          # last context index
            ctx = corr_series.iloc[:origin_pos + 1].copy()
            if is_winsorized:
                ctx = winsorize_train(ctx)
            contexts.append(ctx.values.astype(np.float64))

            actuals      = orig_values[origin_pos + 1: origin_pos + 1 + horizon].astype(float)
            target_dates = idx[origin_pos + 1: origin_pos + 1 + horizon]
            origin_date  = idx[origin_pos]
            meta.append((fold_idx, origin_pos, origin_date, target_dates, actuals))

        outs = forecast_chronos_batch(pipeline, contexts, horizon,
                                      quantile_levels, batch_size)

        fold_results = []
        for (fold_idx, origin_pos, origin_date, target_dates, actuals), out in zip(meta, outs):
            if len(actuals) != horizon:
                continue
            fold_results.append({
                "fold_pos":     fold_idx,
                "origin_pos":   origin_pos,
                "origin_date":  origin_date,
                "target_dates": target_dates,
                "actuals":      actuals,
                "context_len":  origin_pos + 1,
                "median":       out["median"],
                "mean":         out["mean"],
                "quantiles":    out["quantiles"],
            })

        all_results[variant_label] = {"fold_results": fold_results}

    return all_results


# =============================================================================
# SEGMENT PROCESSING  — one CV pass over ALL variants for one sector
# =============================================================================

def process_segment(pipeline, sbi_code, df_ts, batch_size=BATCH_SIZE):
    """Run the full h=HORIZON CV over every correction variant for one sector.

    Returns:
        seg_records: list of dicts — the full prediction matrix for this sector
                     (every variant × fold × horizon, inner+outer), the single
                     source every export method is later sliced/selected from.
        meta:        dict — per-sector fold/diagnostic metadata, or None if the
                     sector was skipped (too few folds).
    """
    variants = build_correction_variants(df_ts)
    orig = df_ts.values.ravel().astype(np.float64)
    n = len(df_ts)

    outer_start = max(MIN_TRAIN_ABS, int(n * MIN_TRAIN_FRAC))
    first_origin = outer_start - 1
    last_origin  = n - HORIZON - 1
    n_total_folds = max(0, last_origin - first_origin + 1)

    if n_total_folds < 5:
        print(f"    ⚠️  {sbi_code}: only {n_total_folds} folds with "
              f"outer_start={outer_start}, h={HORIZON}. Skipping.")
        return [], None

    seg_start = time.time()
    n_inner = max(1, int(n_total_folds * INNER_FRAC))
    n_outer = n_total_folds - n_inner

    print(f"      Training start: {outer_start} quarters | "
          f"Folds: {n_total_folds} = {n_inner} inner + {n_outer} outer | "
          f"Horizons: {HORIZON}")

    seasonal_warn = check_seasonal_stability(df_ts)
    if seasonal_warn:
        print(f"    ⚠️  SEASONAL: {seasonal_warn}")

    corr_results = {}
    for label, ts_c in variants.items():
        corr_results.update(
            run_cv_for_correction(pipeline, label, ts_c, orig, outer_start,
                                  HORIZON, QUANTILES, batch_size)
        )

    seg_records = []
    for label, wr in corr_results.items():
        for fr in wr["fold_results"]:
            is_outer = fr["fold_pos"] >= n_inner
            for h_idx in range(HORIZON):
                horizon = h_idx + 1
                pred   = float(fr["median"][h_idx])
                actual = float(fr["actuals"][h_idx])
                ok     = np.isfinite(pred)
                tdate  = fr["target_dates"][h_idx]
                rec = {
                    "sbi_code":    sbi_code,
                    "correction":  label,
                    "origin_date": fr["origin_date"],
                    "target_date": tdate,
                    "horizon":     horizon,
                    "actual":      actual,
                    "pred":        pred if ok else np.nan,
                    "y_mean":      float(fr["mean"][h_idx]),
                    "abs_error":   float(abs(actual - pred)) if ok else np.nan,
                    "fold_set":    "outer" if is_outer else "inner",
                    "fold_pos":    fr["fold_pos"],
                    "context_len": fr["context_len"],
                    "in_covid":    bool(covid_start <= tdate <= covid_end),
                }
                for q in QUANTILES:
                    rec[f"q{int(q * 100):02d}"] = float(fr["quantiles"][q][h_idx])
                seg_records.append(rec)

    meta = {
        "sbi_code":               sbi_code,
        "n_inner_folds":          n_inner,
        "n_outer_folds":          n_outer,
        "outer_start":            outer_start,
        "horizon":                HORIZON,
        "seasonal_shift_warning": seasonal_warn,
        "seg_minutes":            (time.time() - seg_start) / 60,
    }
    return seg_records, meta


# =============================================================================
# EXPORT — build each method's canonical table from the shared CV matrix
# =============================================================================

def build_export_methods(which):
    """Declarative list of methods to export from one CV pass."""
    methods = [
        {"name": "Chronos_Bolt",           "tag": "",           "kind": "fixed",
         "label": "no_correction"},                              # zero-shot
        {"name": "Chronos_Bolt_corrected", "tag": "_corrected", "kind": "select",
         "label": None},                                         # per-sector pick
    ]
    if which == "all":
        for label in _VALID_FIXED_LABELS:
            if label == "no_correction":
                continue   # == zero-shot Chronos_Bolt
            safe = label.replace("|", "-")
            methods.append({"name": f"Chronos_Bolt_{safe}",
                            "tag": f"_fixed-{safe}", "kind": "fixed",
                            "label": label})
    return methods


def _select_winner_per_sector(sector_df):
    """Lowest inner-fold mean |actual - median| over all variants (AutoETS-style)."""
    inner = sector_df[sector_df["fold_set"] == "inner"]
    best_label, best_mae = None, np.inf
    for label, g in inner.groupby("correction"):
        ae = (g["actual"] - g["pred"]).abs()
        ae = ae[np.isfinite(ae)]
        if len(ae):
            m = float(ae.mean())
            if m < best_mae:
                best_mae, best_label = m, label
    return best_label, best_mae


def _canonical_from_rows(rows_df, model_name):
    """Map outer-fold prediction rows -> canonical schema for m_evaluation."""
    v = rows_df.dropna(subset=["pred"]).copy()
    return pd.DataFrame({
        "model_name":  model_name,
        "sector_code": v["sbi_code"].astype(str),
        "origin_date": pd.to_datetime(v["origin_date"]),
        "target_date": pd.to_datetime(v["target_date"]),
        "horizon":     v["horizon"].astype(int),
        "y_true":      v["actual"].astype(float),
        "y_pred":      v["pred"].astype(float),       # median (q50)
        "y_lower_80":  v["q10"].astype(float),
        "y_upper_80":  v["q90"].astype(float),
        "y_lower_95":  np.nan,
        "y_upper_95":  np.nan,
    })


def build_canonical(full_df, method, meta_by_sector):
    """Return (canonical_df, select_configs_or_None) for one export method."""
    if method["kind"] == "fixed":
        rows = full_df[(full_df["correction"] == method["label"])
                       & (full_df["fold_set"] == "outer")]
        return _canonical_from_rows(rows, method["name"]), None

    # kind == "select": per-sector inner-fold winner, then its outer rows
    parts, configs = [], {}
    for sec, sdf in full_df.groupby("sbi_code"):
        label, inner_mae = _select_winner_per_sector(sdf)
        if label is None:
            continue
        outer = sdf[(sdf["correction"] == label) & (sdf["fold_set"] == "outer")]
        parts.append(outer)
        om = (outer["actual"] - outer["pred"]).abs()
        om = om[np.isfinite(om)]
        configs[str(sec)] = {
            "winning_correction": label,
            "inner_mae":          inner_mae,
            "outer_mae":          float(om.mean()) if len(om) else np.nan,
            **{k: v for k, v in meta_by_sector.get(sec, {}).items()
               if k != "sbi_code"},
        }
    sel = pd.concat(parts, ignore_index=True) if parts else full_df.iloc[0:0]
    return _canonical_from_rows(sel, method["name"]), configs


# =============================================================================
# DATA LOADING
# =============================================================================

def load_series(data_source):
    if data_source == "gold":
        print("  Loading target series from gold table...")
        return load_target_series_from_gold(
            gold_db_path=DIR_DB_GOLD, min_history=MIN_HISTORY, verbose=True,
        )
    print("  Loading target series from silver table (contiguous-tail mode)...")
    return load_target_series_from_silver(
        silver_db_path=DIR_DB_SILVER, min_history=MIN_HISTORY, verbose=True,
    )


# =============================================================================
# ONE DATA SOURCE = one model pass + all method exports
# =============================================================================

def run_one_data_source(pipeline, data_source, args):
    """Load → CV (all variants, once) → export every method, for one data source."""
    ds = "_silver" if data_source == "silver" else ""
    methods = build_export_methods(args.methods)

    print("\n" + "=" * 64)
    print(f"  CHRONOS-BOLT SINGLE-PASS CV — HORIZON = {HORIZON}")
    print(f"  Data source: {data_source.upper()}"
          f"{'  (sensitivity run; *_silver outputs)' if ds else ''}")
    print(f"  Methods: {args.methods}  ->  "
          f"{', '.join(m['name'] for m in methods)}")
    print(f"  Device: {DEVICE} | dtype: {DTYPE} | batch: {args.batch_size}")
    print("=" * 64)

    # --- Load + snap to quarterly + filter (mirrors AutoETS) ---
    segments_all = load_series(data_source)
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

    print(f"\n  Segments: {len(segments)} | Correction variants: "
          f"{len(_VALID_FIXED_LABELS)}")

    if args.dry_run:
        sample = next(iter(segments.values()))
        print(f"    example sector range: "
              f"{sample.index.min().date()} → {sample.index.max().date()} "
              f"({len(sample)} quarters)")
        print("  --dry-run: no forecasting performed.")
        return

    # --- One CV pass over all variants ---
    wall_start = time.time()
    all_records, meta_by_sector = [], {}
    for i, (code, df_ts) in enumerate(segments.items()):
        print(f"\n  SEGMENT {i + 1}/{len(segments)}: {code}")
        recs, meta = process_segment(pipeline, str(code), df_ts, args.batch_size)
        all_records.extend(recs)
        if meta:
            meta_by_sector[str(code)] = meta
    total_min = (time.time() - wall_start) / 60

    full_df = pd.DataFrame(all_records)
    if full_df.empty:
        print("  No results.")
        return

    # --- Full prediction matrix (all variants × folds × horizons), once ---
    p_all = OUTPUT_DIR / f"chronos_cv_all_predictions{ds}.parquet"
    full_df.to_parquet(p_all, index=False)
    print(f"\n  ✅ {p_all} ({len(full_df)} rows, full variant matrix)")

    # --- Export each method's canonical parquet from the shared matrix ---
    select_configs = None
    for m in methods:
        canon, configs = build_canonical(full_df, m, meta_by_sector)
        if configs is not None:
            select_configs = configs
        p = OUTPUT_DIR / f"chronos_predictions{m['tag']}{ds}.parquet"
        canon.to_parquet(p, index=False)
        print(f"  ✅ {p}  [{m['name']}, {len(canon)} rows]")

    # --- Per-sector selection metadata for the corrected method ---
    if select_configs:
        p_cfg = OUTPUT_DIR / f"chronos_best_configs{ds}.json"
        clean = {
            sec: {k: (float(v) if isinstance(v, np.floating)
                      else int(v) if isinstance(v, np.integer) else v)
                  for k, v in cfg.items()}
            for sec, cfg in select_configs.items()
        }
        with open(p_cfg, "w") as f:
            json.dump(clean, f, indent=2, default=str)
        print(f"  ✅ {p_cfg} (corrected-method selection per sector)")

    print(f"\n  {data_source.upper()} total: {total_min:.1f} min")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"Chronos-Bolt h={HORIZON} single-pass CV — exports every "
                    f"correction method from one model pass.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--segment", type=str, default=None,
                        help="If set, only process this SBI code")
    parser.add_argument(
        "--methods", choices=["all", "core"], default="all",
        help="'all' (default): zero-shot + corrected + the five fixed-variant "
             "diagnostic files. 'core': only Chronos_Bolt (zero-shot) and "
             "Chronos_Bolt_corrected (the two the notebook consumes).",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--data-source", choices=["gold", "silver", "both"], default="gold",
        help="Where to read target series from.  "
             "'gold' (default): imputed values from master_data_ml_preprocessed, "
             "matching what Pipeline trains on.  "
             "'silver': raw observations from 80072ned_silver with contiguous-tail "
             "extraction (no fabricated values), written with a '_silver' tag for "
             "sensitivity analysis.  "
             "'both': run gold then silver in one process (model loaded once).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    sources = ["gold", "silver"] if args.data_source == "both" else [args.data_source]

    pipeline = None
    if not args.dry_run:
        print(f"\n  Loading {MODEL_ID} on {DEVICE} ...")
        pipeline = BaseChronosPipeline.from_pretrained(
            MODEL_ID, device_map=DEVICE, dtype=DTYPE,
        )
        print("  Model ready.")

    warnings.filterwarnings("ignore")
    for src in sources:
        run_one_data_source(pipeline, src, args)
    warnings.filterwarnings("default")


if __name__ == "__main__":
    main()
