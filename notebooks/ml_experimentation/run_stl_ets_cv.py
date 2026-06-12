#!/usr/bin/env python3
"""
run_stl_ets_cv.py — Nested CV for STL+ETS at horizon h=4, per sector.

WHAT CHANGED (vs previous version):
- Restructured so each WORKER processes ONE correction variant across ALL
  outer folds, with cross-fold caching of STL fits and ETS forecasts.  The
  previous version re-fit STL and ETS from scratch at every inner position
  for every outer fold, which made T001081 take many hours.  This version
  computes each (stl_config, ets_config, inner_j) ONCE per variant.  Wall
  time per sector: roughly 30x faster.
- Per-variant progress is printed as variants complete, so a long-running
  sector shows visible motion instead of going silent.
- Dropped seasonal=13 STL (often fails on short series; expensive on failure).
- Dropped error="mul" ETS (deseasoned values can be negative → fit failures
  that take seconds each to give up on).

What did NOT change:
- Outer fold semantics: outer_i is the last training position; targets are
  positions outer_i+1..outer_i+HORIZON.
- Inner CV no-leakage rule: inner_j <= outer_i - HORIZON + 1.
- COVID correction variants (raw, 9× seasonal_mean|b{1,2,3}_a{1,2,3}, each
  also with a winsorized version).
- Canonical schema written to stl_ets_predictions.parquet for evaluation_method.
- Selection objective: mean MAE across all 4 horizons within each outer fold's
  inner CV.

Tunable knobs at the top of SETTINGS:
- STL_SEASONAL_OPTS and STL_DEG_OPTS: STL config grid
- ETS_CONFIGS: ETS config grid
- INNER_MIN_OBS, OUTER_START: CV boundaries
"""

import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import warnings
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.stl import STLForecast

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
HORIZON       = 4
N_WORKERS     = 10
OUTPUT_DIR    = Path("cv_output")
covid_start   = pd.Timestamp('2020-03-31')
covid_end     = pd.Timestamp('2022-06-30')
outer_start   = 24
inner_min_obs = 12
MIN_HISTORY   = 33  # SHARED with run_autoets_cv.py and run_chronos_cv.py
                    # — ensures the three scripts process the same sectors

# Config grids — tunable.
STL_SEASONAL_OPTS = [5, 7]
STL_DEG_OPTS      = [0, 1]
STL_CONFIGS       = list(product(STL_SEASONAL_OPTS, STL_DEG_OPTS))  # 4 configs

ETS_CONFIGS = [(e, t, d)
               for e, t, d in product(["add"], ["add", None], [True, False])
               if not (t is None and d is True)]                    # 3 configs

TASK_TIMEOUT_S = 1800   # 30 minutes per variant safety cap


# =============================================================================
# UTILITIES
# =============================================================================

def _get_seasonal_forecast(stl_seasonal_values, n_steps, period=4):
    """Tile the last seasonal cycle to produce n_steps-ahead seasonal indices."""
    last_cycle = stl_seasonal_values[-period:]
    reps = (n_steps // period) + 1
    return np.tile(last_cycle, reps)[:n_steps]


def _failed_fold_result(corr_label, outer_i, horizon):
    return {
        "outer_i":     outer_i,
        "correction":  corr_label,
        "outer_preds": np.full(horizon, np.nan),
        "inner_mae":   np.inf,
        "stl_s": None, "stl_sd": None,
        "ets_e": None, "ets_t": None, "ets_d": None,
    }


# =============================================================================
# WORKER — process ONE variant across ALL outer folds, with cross-fold caching
# =============================================================================

def evaluate_variant(corr_label, corr_values, corr_index,
                     outer_positions, stl_configs, ets_configs,
                     inner_min_obs, orig_values, horizon):
    """
    Evaluate one correction variant across ALL outer folds.

    Strategy
    --------
    1. Pre-compute STL + ETS at every (stl_config, ets_config, inner_j) once.
       Inner_j ranges across ALL positions that could matter for any outer
       fold: [inner_min_obs, max(outer_positions) - horizon + 1].
    2. For each outer fold:
         a. Determine its valid inner positions (subset of the pre-computed set).
         b. For each (stl_config, ets_config), compute mean MAE across those
            inner positions using the cached absolute errors.
         c. Pick the best config; refit STL+ETS on outer training and forecast.
    3. Return per-fold results.

    Compared to the previous version (re-fit per outer fold), this saves a
    factor of ~30 in compute on long series.
    """
    warnings.filterwarnings("ignore")
    ts_corr = pd.Series(corr_values, index=corr_index)

    max_outer = max(outer_positions)
    all_inner_first = inner_min_obs
    all_inner_last  = max_outer - horizon + 1
    if all_inner_last < all_inner_first:
        return [_failed_fold_result(corr_label, outer_i, horizon)
                for outer_i in outer_positions]
    all_inner_positions = list(range(all_inner_first, all_inner_last + 1))

    # PASS 1: pre-compute |errors| for every (stl, ets, inner_j) ONCE
    inner_errs = {}
    for stl_idx, (stl_s, stl_sd) in enumerate(stl_configs):
        for inner_j in all_inner_positions:
            inner_train = ts_corr.iloc[:inner_j]
            try:
                stl_fit = STL(inner_train, period=4, seasonal=stl_s,
                              seasonal_deg=stl_sd, robust=True).fit()
                deseas_vals = (inner_train.values - stl_fit.seasonal.values)
                seas_arr    = stl_fit.seasonal.values
            except Exception:
                for ets_idx in range(len(ets_configs)):
                    inner_errs[(stl_idx, ets_idx, inner_j)] = np.full(horizon, np.nan)
                continue

            seas_fc = _get_seasonal_forecast(seas_arr, n_steps=horizon, period=4)
            deseas_series = pd.Series(deseas_vals, index=inner_train.index)
            inner_actuals = orig_values[inner_j : inner_j + horizon].astype(float)
            if len(inner_actuals) != horizon:
                for ets_idx in range(len(ets_configs)):
                    inner_errs[(stl_idx, ets_idx, inner_j)] = np.full(horizon, np.nan)
                continue

            for ets_idx, (ets_e, ets_t, ets_d) in enumerate(ets_configs):
                try:
                    ets_fit = ETSModel(
                        deseas_series,
                        error=ets_e, trend=ets_t, seasonal=None,
                        damped_trend=ets_d,
                    ).fit(disp=0)
                    raw_preds = ets_fit.forecast(horizon).values.astype(float)
                    preds = raw_preds + seas_fc
                    if np.any(~np.isfinite(preds)):
                        preds = np.where(np.isfinite(preds), preds, np.nan)
                    abs_errs = np.abs(inner_actuals - preds)
                except Exception:
                    abs_errs = np.full(horizon, np.nan)
                inner_errs[(stl_idx, ets_idx, inner_j)] = abs_errs

    # PASS 2: per outer fold, aggregate cached errors, pick best config,
    # then refit on outer training and forecast.
    fold_results = []
    for outer_i in outer_positions:
        inner_last_here = outer_i - horizon + 1
        valid_inner = [j for j in all_inner_positions
                       if inner_min_obs <= j <= inner_last_here]
        if not valid_inner:
            fold_results.append(_failed_fold_result(corr_label, outer_i, horizon))
            continue

        best_mae = np.inf
        best_cfg = None
        for stl_idx in range(len(stl_configs)):
            for ets_idx in range(len(ets_configs)):
                errs_flat = []
                for j in valid_inner:
                    arr = inner_errs.get((stl_idx, ets_idx, j))
                    if arr is None:
                        continue
                    fin = arr[np.isfinite(arr)]
                    errs_flat.extend(fin.tolist())
                if errs_flat:
                    m = float(np.mean(errs_flat))
                    if m < best_mae:
                        best_mae = m
                        best_cfg = (stl_idx, ets_idx)

        if best_cfg is None:
            fold_results.append(_failed_fold_result(corr_label, outer_i, horizon))
            continue

        stl_idx, ets_idx = best_cfg
        best_stl_s, best_stl_sd            = stl_configs[stl_idx]
        best_ets_e, best_ets_t, best_ets_d = ets_configs[ets_idx]

        outer_train = ts_corr.iloc[:outer_i + 1]
        try:
            outer_preds = STLForecast(
                outer_train, model=ETSModel,
                model_kwargs={"error": best_ets_e, "trend": best_ets_t,
                              "seasonal": None, "damped_trend": best_ets_d},
                period=4, seasonal=best_stl_s, seasonal_deg=best_stl_sd,
                robust=True,
            ).fit().forecast(horizon).values.astype(float)
            outer_preds = np.where(np.isfinite(outer_preds), outer_preds, np.nan)
        except Exception:
            outer_preds = np.full(horizon, np.nan)

        fold_results.append({
            "outer_i":     outer_i,
            "correction":  corr_label,
            "outer_preds": outer_preds,
            "inner_mae":   round(best_mae, 6),
            "stl_s":  best_stl_s,  "stl_sd": best_stl_sd,
            "ets_e":  best_ets_e,  "ets_t":  best_ets_t,  "ets_d": best_ets_d,
        })

    return fold_results


# =============================================================================
# CORRECTION VARIANTS
# =============================================================================

def build_correction_variants(df_ts):
    covid_mask = (df_ts.index >= covid_start) & (df_ts.index <= covid_end)

    def _seasonal_mean(ts, mask, b_win, a_win):
        ts_out = ts.copy()
        for date in ts.index[mask]:
            same_q = ts.index[(ts.index.quarter == date.quarter) & (~mask)]
            refs_b = same_q[same_q < covid_start][-b_win:]
            refs_a = same_q[same_q > covid_end][:a_win]
            ref_dates = refs_b.union(refs_a)
            if len(ref_dates) > 0:
                ts_out[date] = ts[ref_dates].mean()
        return ts_out

    def _winsorize(ts, lo=0.05, hi=0.95):
        return ts.clip(lower=ts.quantile(lo), upper=ts.quantile(hi))

    variants = {"no_correction": df_ts.copy(),
                "no_correction|winsorized": _winsorize(df_ts.copy())}
    for b, a in product([1, 2, 3], [1, 2, 3]):
        label = f"seasonal_mean|b{b}_a{a}"
        c = _seasonal_mean(df_ts, covid_mask, b, a)
        variants[label] = c
        variants[f"{label}|winsorized"] = _winsorize(c)

    for label, ts in variants.items():
        assert ts.index.equals(df_ts.index), f"Index mismatch: '{label}'"
    return variants


# =============================================================================
# PER-SECTOR PROCESSING
# =============================================================================

def process_sector(sbi_code, df_ts, n_workers):
    """Run nested CV for one sector — per-variant parallelism, cross-fold cached."""
    correction_variants = build_correction_variants(df_ts)
    orig_values = df_ts.values.ravel().astype(np.float64)

    first_outer = outer_start - 1
    last_outer  = len(df_ts) - HORIZON - 1
    if last_outer < first_outer:
        print(f"    {sbi_code}: too short (len={len(df_ts)}, h={HORIZON}). Skipping.")
        return []

    outer_positions = list(range(first_outer, last_outer + 1))

    n_inner_positions = max(0, last_outer - HORIZON + 1 - inner_min_obs + 1)
    est_fits_per_variant = (
        len(STL_CONFIGS) * n_inner_positions
        + len(STL_CONFIGS) * len(ETS_CONFIGS) * n_inner_positions
        + len(outer_positions)
    )
    print(f"    {sbi_code}: {len(outer_positions)} outer folds, "
          f"{len(correction_variants)} variants, "
          f"~{est_fits_per_variant:,} fits/variant, "
          f"{len(STL_CONFIGS)*len(ETS_CONFIGS)} (stl×ets) configs")

    t0 = time.time()
    fold_results_by_variant = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for lbl, ts_c in correction_variants.items():
            f = pool.submit(evaluate_variant,
                            lbl, ts_c.values.ravel(), ts_c.index,
                            outer_positions, STL_CONFIGS, ETS_CONFIGS,
                            inner_min_obs, orig_values, HORIZON)
            futures[f] = lbl

        n_done = 0
        n_total = len(futures)
        for f in as_completed(futures):
            lbl = futures[f]
            n_done += 1
            try:
                fr_list = f.result(timeout=TASK_TIMEOUT_S)
                fold_results_by_variant[lbl] = fr_list
                ok = sum(1 for r in fr_list if np.isfinite(r["outer_preds"]).any())
                print(f"      [{n_done:>2}/{n_total}] {lbl:40s}  "
                      f"{ok}/{len(fr_list)} folds with valid preds  "
                      f"({(time.time()-t0)/60:.1f} min)")
            except Exception as exc:
                print(f"      [{n_done:>2}/{n_total}] {lbl:40s}  FAILED ({exc})")
                fold_results_by_variant[lbl] = [
                    _failed_fold_result(lbl, oi, HORIZON) for oi in outer_positions
                ]

    cv_records = []
    for variant_lbl, fr_list in fold_results_by_variant.items():
        for r in fr_list:
            outer_i = r["outer_i"]
            outer_date = df_ts.index[outer_i]
            actuals = orig_values[outer_i + 1 : outer_i + 1 + HORIZON].astype(float)
            target_dates = df_ts.index[outer_i + 1 : outer_i + 1 + HORIZON]
            for h_idx in range(HORIZON):
                pred = r["outer_preds"][h_idx]
                actual = float(actuals[h_idx])
                ok = np.isfinite(pred)
                cv_records.append({
                    "sbi_code":         sbi_code,
                    "origin_date":      outer_date,
                    "target_date":      target_dates[h_idx],
                    "horizon":          h_idx + 1,
                    "correction":       r["correction"],
                    "actual":           actual,
                    "pred":             float(pred) if ok else np.nan,
                    "abs_error":        abs(actual - pred) if ok else np.nan,
                    "inner_mae":        r["inner_mae"],
                    "ets_error":        r["ets_e"],
                    "ets_trend":        str(r["ets_t"]),
                    "ets_damped":       r["ets_d"],
                    "stl_seasonal":     r["stl_s"],
                    "stl_seasonal_deg": r["stl_sd"],
                })

    print(f"    {sbi_code} done in {(time.time()-t0)/60:.1f} min")
    return cv_records


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=f"STL+ETS h={HORIZON} nested CV per sector")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
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

    # Output-filename suffix lets the two data-source runs coexist on disk
    out_suffix = "_silver" if args.data_source == "silver" else ""

    print("=" * 60)
    print(f"  STL+ETS NESTED CV — HORIZON = {HORIZON}")
    print(f"  Data source: {args.data_source.upper()}"
          f"{'  (sensitivity run; outputs *_silver.parquet)' if out_suffix else ''}")
    print(f"  STL grid: {STL_CONFIGS}")
    print(f"  ETS grid: {ETS_CONFIGS}")
    print("=" * 60)

    # --- Load per-sector target series ---
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

    # Snap to quarterly frequency and filter to --segment if set
    segments = {}
    for c, ts in segments_all.items():
        if args.segment and c != args.segment:
            continue
        ts = ts.asfreq("QE")
        if ts.isna().any():
            print(f"  ⚠️  Skipping {c}: unexpected NaN ({ts.isna().sum()} NaN)")
            continue
        if len(ts) < MIN_HISTORY:
            continue
        segments[c] = ts

    if args.dry_run:
        print(f"  Would process {len(segments)} sector(s).")
        for code, ts in list(segments.items())[:5]:
            print(f"    {code}: {len(ts)} obs")
        return

    all_records = []
    wall_start = time.time()
    for i, (code, df_ts) in enumerate(segments.items()):
        print(f"\n  [{i+1}/{len(segments)}] Processing {code} ({len(df_ts)} obs):")
        records = process_sector(code, df_ts, args.workers)
        all_records.extend(records)

    total_min = (time.time() - wall_start) / 60
    df_cv = pd.DataFrame(all_records)

    if df_cv.empty:
        print("  No results.")
        return

    # METHODOLOGICAL FIX (honest correction selection):
    # The winning correction variant is now picked by AVERAGE INNER-CV MAE
    # across outer folds — NOT by outer-fold abs_error.
    # Why: each outer fold's inner_mae is computed entirely on training-window
    # data (positions <= outer_i), so it's never contaminated by outer-fold
    # test predictions.  Picking the correction on outer abs_error would be
    # test-set selection (a milder form of the cherry-picking that previously
    # affected the Pipeline loader).
    #
    # Each row of df_cv has an `inner_mae` column populated by evaluate_variant:
    # it's the inner-CV MAE of the best (stl_s, stl_sd, ets_e, ets_t, ets_d)
    # config in that outer fold's inner search.  We average across folds.
    by_corr = (df_cv.dropna(subset=["inner_mae"])
                    .groupby(["sbi_code", "correction"])["inner_mae"]
                    .mean()
                    .reset_index()
                    .rename(columns={"inner_mae": "avg_inner_mae"}))
    winners = (by_corr.sort_values(["sbi_code", "avg_inner_mae"])
                       .drop_duplicates("sbi_code", keep="first")
                       .set_index("sbi_code")["correction"].to_dict())
    print(f"  Correction-variant winners chosen by avg inner-CV MAE "
          f"(honest selection across {by_corr['sbi_code'].nunique()} sectors)")

    p1 = OUTPUT_DIR / f"cv_results{out_suffix}.parquet"
    p2 = OUTPUT_DIR / f"stl_ets_best_configs{out_suffix}.json"

    df_cv.to_parquet(p1, index=False); print(f"\n  ✅ {p1}")
    with open(p2, "w") as f:
        json.dump({k: v for k, v in winners.items()}, f, indent=2)
    print(f"  ✅ {p2}")

    # Canonical-schema export for evaluation_method.py
    winner_series = df_cv["sbi_code"].map(winners)
    df_cv["_is_winner"] = df_cv["correction"] == winner_series
    winner_rows = df_cv[df_cv["_is_winner"]].dropna(subset=["pred"]).copy()
    canonical = pd.DataFrame({
        "model_name":   "STL_ETS",
        "sector_code":  winner_rows["sbi_code"].astype(str),
        "origin_date":  pd.to_datetime(winner_rows["origin_date"]),
        "target_date":  pd.to_datetime(winner_rows["target_date"]),
        "horizon":      winner_rows["horizon"].astype(int),
        "y_true":       winner_rows["actual"].astype(float),
        "y_pred":       winner_rows["pred"].astype(float),
        "y_lower_80":   np.nan,
        "y_upper_80":   np.nan,
        "y_lower_95":   np.nan,
        "y_upper_95":   np.nan,
    })
    p_canon = OUTPUT_DIR / f"stl_ets_predictions{out_suffix}.parquet"
    canonical.to_parquet(p_canon, index=False)
    print(f"  ✅ {p_canon} (canonical, {len(canonical)} rows)")
    print(f"\n  Total: {total_min:.1f} min")


if __name__ == "__main__":
    main()
