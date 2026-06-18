#!/usr/bin/env python3
"""
run_nested_cv.py — Standalone script for the optimized nested CV loop.

Usage:
    python run_nested_cv.py              # full run
    python run_nested_cv.py --dry-run    # print config and exit
    python run_nested_cv.py --workers 6  # override worker count

Run from the same directory as your notebook, or from the project root.
The script saves results to ./cv_output/:
    - cv_results.parquet    (full fold-level results)
    - cv_summary.parquet    (aggregated metrics per correction)
    - best_config.pkl       (winning params + corrected series + metrics)

Load results back in your notebook:
    import pickle, pandas as pd
    df_cv = pd.read_parquet("cv_output/cv_results.parquet")
    with open("cv_output/best_config.pkl", "rb") as f:
        config = pickle.load(f)
    df_ts_corrected = config["df_ts_corrected"]
    best_params     = config["best_params"]
    best_stl_params = config["best_stl_params"]
"""

# =============================================================================
# BLAS THREAD PINNING — before any numpy import
# =============================================================================
import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import pickle
import warnings
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.stl import STLForecast

# =============================================================================
# PATH RESOLUTION — uses __file__, not cwd()
# =============================================================================
# Resolve project root relative to THIS SCRIPT's location.
# Assumes: <project_root>/notebooks/<subfolder>/run_nested_cv.py
# Adjust the .parent chain if the script lives elsewhere.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DIR_DB_SILVER
from src.utils.m_query_database import f_query_database

# =============================================================================
# SETTINGS
# =============================================================================
N_WORKERS     = 10
OUTPUT_DIR    = Path("cv_output")
covid_start   = pd.Timestamp('2020-03-31')
covid_end     = pd.Timestamp('2022-06-30')
outer_start   = 24
inner_min_obs = 12


# =============================================================================
# WORKER FUNCTION
# =============================================================================
# Defined at module level for pickling. Uses ONLY its arguments —
# no closures over module globals. Safe for both fork and spawn.

def _get_seasonal_forecast(stl_seasonal_values, n_steps, period=4):
    """Replicate STLForecast's seasonal extrapolation."""
    last_cycle = stl_seasonal_values[-period:]
    reps = (n_steps // period) + 1
    return np.tile(last_cycle, reps)[:n_steps]


def evaluate_one_correction(corr_label, ts_corr_values, ts_corr_index,
                            outer_i, stl_configs, ets_configs,
                            inner_min_obs, orig_values):
    """Inner grid search for one correction variant at one outer fold."""
    warnings.filterwarnings("ignore")

    ts_corr = pd.Series(ts_corr_values, index=ts_corr_index)
    n_inner_folds = outer_i - inner_min_obs
    max_ets_nans = n_inner_folds * 0.2

    global_best_mae = np.inf
    global_best_cfg = None

    for stl_s, stl_sd in stl_configs:

        stl_cache = {}
        stl_fail_count = 0

        for inner_j in range(inner_min_obs, outer_i):
            inner_train = ts_corr.iloc[:inner_j]
            try:
                stl_fit = STL(inner_train, period=4, seasonal=stl_s,
                              seasonal_deg=stl_sd, robust=True).fit()
                deseas_vals = (inner_train - stl_fit.seasonal).values
                seas_fc = _get_seasonal_forecast(
                    stl_fit.seasonal.values, n_steps=1, period=4
                )[0]
                stl_cache[inner_j] = (deseas_vals, inner_train.index, seas_fc)
            except Exception:
                stl_cache[inner_j] = None
                stl_fail_count += 1

        if stl_fail_count > n_inner_folds * 0.5:
            continue

        n_stl_ok = n_inner_folds - stl_fail_count

        for ets_e, ets_t, ets_d in ets_configs:

            running_abs_err = 0.0
            n_valid       = 0
            ets_nan_count = 0
            abandoned     = False

            for inner_j in range(inner_min_obs, outer_i):

                inner_actual = orig_values[inner_j]

                cached = stl_cache[inner_j]
                if cached is None:
                    continue

                deseas_vals, deseas_index, seas_fc = cached

                try:
                    deseas_series = pd.Series(deseas_vals, index=deseas_index)
                    ets_fit = ETSModel(
                        deseas_series,
                        error=ets_e, trend=ets_t, seasonal=None,
                        damped_trend=ets_d,
                    ).fit(disp=0)
                    raw_pred = ets_fit.forecast(1).iloc[0]
                    inner_pred = raw_pred + seas_fc
                except Exception:
                    ets_nan_count += 1
                    if ets_nan_count > max_ets_nans:
                        abandoned = True
                        break
                    continue

                if np.isnan(inner_pred):
                    ets_nan_count += 1
                    if ets_nan_count > max_ets_nans:
                        abandoned = True
                        break
                    continue

                running_abs_err += abs(inner_actual - inner_pred)
                n_valid += 1

                best_possible_n = n_stl_ok - ets_nan_count
                if best_possible_n <= 0:
                    abandoned = True
                    break
                optimistic_mae = running_abs_err / best_possible_n
                if optimistic_mae > global_best_mae:
                    abandoned = True
                    break

            if abandoned or n_valid == 0:
                continue

            mae = running_abs_err / n_valid
            if mae < global_best_mae:
                global_best_mae = mae
                global_best_cfg = (stl_s, stl_sd, ets_e, ets_t, ets_d)

    # --- Outer prediction ---
    if global_best_cfg is None:
        return {
            "correction": corr_label, "pred": np.nan,
            "inner_mae": np.inf, "stl_s": None, "stl_sd": None,
            "ets_e": None, "ets_t": None, "ets_d": None,
        }

    best_stl_s, best_stl_sd, best_ets_e, best_ets_t, best_ets_d = global_best_cfg

    outer_train = ts_corr.iloc[:outer_i]
    try:
        outer_pred = STLForecast(
            outer_train,
            model=ETSModel,
            model_kwargs={
                "error": best_ets_e, "trend": best_ets_t,
                "seasonal": None, "damped_trend": best_ets_d,
            },
            period=4,
            seasonal=best_stl_s,
            seasonal_deg=best_stl_sd,
            robust=True,
        ).fit().forecast(1).iloc[0]
    except Exception:
        outer_pred = np.nan

    return {
        "correction": corr_label,
        "pred":       float(outer_pred) if pd.notna(outer_pred) else np.nan,
        "inner_mae":  round(global_best_mae, 6),
        "stl_s":      best_stl_s,
        "stl_sd":     best_stl_sd,
        "ets_e":      best_ets_e,
        "ets_t":      best_ets_t,
        "ets_d":      best_ets_d,
    }


# =============================================================================
# MAIN — all data loading and loop logic lives INSIDE this function,
# which is called ONLY from the __name__ == "__main__" guard.
#
# This is CRITICAL for Windows/macOS compatibility:
# ProcessPoolExecutor with "spawn" re-imports __main__ in each worker.
# Without the guard, every worker would re-run the SQL query, rebuild
# all correction variants, and waste minutes before starting work.
# With the guard, workers only see function definitions and imports.
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Nested CV for STL-ETS with COVID correction search"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configuration and exit")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help=f"Parallel workers (default: {N_WORKERS})")
    args = parser.parse_args()
    n_workers = args.workers

    OUTPUT_DIR.mkdir(exist_ok=True)

    # =================================================================
    # DATA LOADING
    # =================================================================
    print("=" * 60)
    print("  LOADING DATA")
    print("=" * 60)

    query = """
    SELECT
        Perioden as timeperiod_text,
        BedrijfskenmerkenSBI2008  as sbi_code,
        BedrijfskenmerkenSBI2008_Title as sbi_title,
        DATE(
            printf('%s-%s-01',
                substr(Perioden, 1, 4),
                CASE substr(Perioden, 7, 2)
                    WHEN '01' THEN '01'
                    WHEN '02' THEN '04'
                    WHEN '03' THEN '07'
                    WHEN '04' THEN '10'
                END
            ),
            '+3 months',
            '-1 day'
        ) AS period_enddate,
        CAST(substr(Perioden, 1, 4) as INTEGER) as "year",
        CAST(substr(Perioden, 8, 1) as INTEGER) as "quarter",
        CAST(Ziekteverzuimpercentage_1 AS REAL) as absenteeism_perc
    FROM "80072ned_silver"
    WHERE Perioden NOT LIKE '%JJ%'
    AND substr(Perioden, 1, 4) >= '2012'
    ORDER BY sbi_code, period_enddate ASC
    """

    df_org = f_query_database(DIR_DB_SILVER, query, "pandas")
    df_org["year"] = df_org["year"].astype(int)
    df_org["quarter"] = df_org["quarter"].astype(int)
    df_org["absenteeism_perc"] = pd.to_numeric(
        df_org["absenteeism_perc"], errors="coerce"
    )
    df_org["period_enddate"] = pd.to_datetime(df_org["period_enddate"])

    df_total = (
        df_org[df_org["sbi_code"] == "T001081"]
        .copy()
        .sort_values("period_enddate")
        .reset_index(drop=True)
    )
    df_ts = (
        df_total
        .set_index("period_enddate")["absenteeism_perc"]
        .asfreq("QE")
    )

    assert not df_ts.isna().any(), \
        f"df_ts contains {df_ts.isna().sum()} NaNs — fix before running"
    assert len(df_ts) > outer_start + 4, \
        f"Need >{outer_start + 4} observations, got {len(df_ts)}"

    print(f"  Loaded {len(df_ts)} quarterly observations")
    print(f"  Range: {df_ts.index[0].date()} → {df_ts.index[-1].date()}\n")

    # =================================================================
    # COVID CORRECTION VARIANTS
    # =================================================================
    print("=" * 60)
    print("  BUILDING COVID CORRECTION VARIANTS")
    print("=" * 60)

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

    def _interpolate(ts, mask, method):
        ts_out = ts.copy()
        ts_out.loc[mask] = np.nan
        return ts_out.interpolate(method=method).bfill().ffill()

    def _winsorize(ts, lo=0.05, hi=0.95):
        return ts.clip(lower=ts.quantile(lo), upper=ts.quantile(hi))

    correction_variants = {}
    correction_variants["no_correction"] = df_ts.copy()
    correction_variants["no_correction|winsorized"] = _winsorize(df_ts.copy())

    for b, a in product([1, 2, 3], [1, 2, 3]):
        label = f"seasonal_mean|b{b}_a{a}"
        c = _seasonal_mean(df_ts, covid_mask, b, a)
        correction_variants[label] = c
        correction_variants[f"{label}|winsorized"] = _winsorize(c)

    for m in ["linear", "cubic", "time"]:
        label = f"interp_{m}"
        c = _interpolate(df_ts, covid_mask, m)
        correction_variants[label] = c
        correction_variants[f"{label}|winsorized"] = _winsorize(c)

    for label, ts in correction_variants.items():
        assert ts.index.equals(df_ts.index), \
            f"Index mismatch: '{label}'"

    print(f"  Built {len(correction_variants)} variants\n")

    # =================================================================
    # CONFIG GRIDS
    # =================================================================
    stl_configs = list(product([5, 7, 13], [0, 1]))
    ets_configs = [
        (e, t, d) for e, t, d in product(
            ["add", "mul"], ["add", None], [True, False]
        ) if not (t is None and d is True)
    ]
    _valid_ets = {(e, str(t), d) for e, t, d in ets_configs}
    total_configs = len(stl_configs) * len(ets_configs)
    n_outer = len(df_ts) - outer_start
    _covid_idx = df_ts.index.searchsorted(covid_start)
    orig_values = df_ts.values.ravel().astype(np.float64)

    print("=" * 60)
    print("  CONFIGURATION")
    print("=" * 60)
    print(f"  Outer folds:    {n_outer}")
    print(f"  Corrections:    {len(correction_variants)}")
    print(f"  STL × ETS:      {len(stl_configs)} × {len(ets_configs)} = {total_configs}")
    print(f"  Workers:        {n_workers}")
    print(f"  Output dir:     {OUTPUT_DIR.resolve()}\n")

    if args.dry_run:
        print("  --dry-run: exiting without running.\n")
        return

    # =================================================================
    # NESTED CV LOOP
    # =================================================================
    cv_records = []
    warnings.filterwarnings("ignore")
    wall_start = time.time()

    print("=" * 60)
    print("  RUNNING NESTED CV")
    print("=" * 60)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:

        for fold_idx, outer_i in enumerate(range(outer_start, len(df_ts))):
            outer_date   = df_ts.index[outer_i]
            outer_actual = float(orig_values[outer_i])
            fold_start   = time.time()
            touches_covid = (outer_i > _covid_idx)

            if touches_covid:
                futures = {}
                for lbl, ts_c in correction_variants.items():
                    f = pool.submit(
                        evaluate_one_correction, lbl,
                        ts_c.values.ravel(), ts_c.index, outer_i,
                        stl_configs, ets_configs, inner_min_obs, orig_values,
                    )
                    futures[f] = lbl

                fold_results = []
                for f in as_completed(futures):
                    try:
                        fold_results.append(f.result())
                    except Exception as exc:
                        lbl = futures[f]
                        print(f"    ⚠️  Worker error [{lbl}]: {exc}")
                        fold_results.append({
                            "correction": lbl, "pred": np.nan,
                            "inner_mae": np.inf, "stl_s": None,
                            "stl_sd": None, "ets_e": None,
                            "ets_t": None, "ets_d": None,
                        })
            else:
                cache = {}
                fold_results = []
                for lbl, ts_c in correction_variants.items():
                    key = lbl.endswith("|winsorized")
                    if key not in cache:
                        cache[key] = evaluate_one_correction(
                            lbl, ts_c.values.ravel(), ts_c.index, outer_i,
                            stl_configs, ets_configs, inner_min_obs, orig_values,
                        )
                    r = cache[key].copy()
                    r["correction"] = lbl
                    fold_results.append(r)

            for r in fold_results:
                pred = r["pred"]
                ok = pd.notna(pred)
                cv_records.append({
                    "date":             outer_date,
                    "correction":       r["correction"],
                    "actual":           outer_actual,
                    "pred":             pred if ok else np.nan,
                    "abs_error":        abs(outer_actual - pred) if ok else np.nan,
                    "inner_mae":        r["inner_mae"],
                    "ets_error":        r["ets_e"],
                    "ets_trend":        str(r["ets_t"]),
                    "ets_damped":       r["ets_d"],
                    "stl_seasonal":     r["stl_s"],
                    "stl_seasonal_deg": r["stl_sd"],
                })

            errs = [(r["correction"], abs(outer_actual - r["pred"]))
                    for r in fold_results if pd.notna(r["pred"])]
            errs.sort(key=lambda x: x[1])
            fsec = time.time() - fold_start
            emin = (time.time() - wall_start) / 60
            tag = " [cached]" if not touches_covid else ""

            if errs:
                print(f"  Fold {fold_idx+1:>3}/{n_outer} "
                      f"| {outer_date.date()}{tag} "
                      f"| {fsec:5.1f}s ({emin:5.1f}m) "
                      f"| Best: {errs[0][0]:<30s} {errs[0][1]:.3f} "
                      f"| Worst: {errs[-1][0]:<30s} {errs[-1][1]:.3f}")
            else:
                print(f"  Fold {fold_idx+1:>3}/{n_outer} "
                      f"| {outer_date.date()}{tag} "
                      f"| {fsec:5.1f}s ({emin:5.1f}m) | ALL FAILED")

    total_min = (time.time() - wall_start) / 60
    warnings.filterwarnings("default")
    print(f"\n⏱️  Total wall time: {total_min:.1f} minutes")

    # =================================================================
    # RESULTS & WINNER
    # =================================================================
    df_cv = pd.DataFrame(cv_records)

    def _agg(g):
        v = g.dropna(subset=["pred"])
        if len(v) == 0:
            return pd.Series({"mean_mae": np.nan, "mean_rmse": np.nan,
                              "mape": np.nan, "n_folds": 0, "n_nan": len(g)})
        e = v["pred"] - v["actual"]
        return pd.Series({
            "mean_mae":  v["abs_error"].mean(),
            "mean_rmse": np.sqrt(np.mean(e**2)),
            "mape":      (np.abs(e / v["actual"]) * 100).mean(),
            "n_folds":   len(v),
            "n_nan":     g["pred"].isna().sum(),
        })

    summary = (
        df_cv.groupby("correction", group_keys=False)
        .apply(_agg).sort_values("mean_mae")
    )

    print(f"\n{'='*80}")
    print(f"  NESTED CV RESULTS — RANKED BY MEAN MAE")
    print(f"{'='*80}")
    print(summary[["mean_mae","mean_rmse","mape","n_folds","n_nan"]]
          .to_string(float_format="%.4f"))
    print(f"{'='*80}\n")

    win_corr = summary.index[0]
    print(f"🏆 Best: {win_corr}  (MAE = {summary.iloc[0]['mean_mae']:.4f})\n")

    wv = df_cv[df_cv["correction"] == win_corr].dropna(subset=["pred","stl_seasonal"])
    jc = (wv.groupby(["ets_error","ets_trend","ets_damped",
                       "stl_seasonal","stl_seasonal_deg"])
            .size().sort_values(ascending=False))

    if len(jc) == 0:
        raise RuntimeError("All configs failed.")

    be, bt_s, bd, bs, bsd = jc.index[0]
    bt = None if bt_s == "None" else bt_s
    assert (be, str(bt), bd) in _valid_ets

    df_ts_corrected = correction_variants[win_corr].copy()
    best_params     = {"error": be, "trend": bt, "damped_trend": bd}
    best_stl_params = {"seasonal": int(bs), "seasonal_deg": int(bsd)}

    a = wv["actual"].values.astype(float)
    p = wv["pred"].values.astype(float)
    mae_stl  = np.nanmean(np.abs(a - p))
    rmse_stl = np.sqrt(np.nanmean((a - p)**2))
    mape_stl = np.nanmean(np.abs((p - a) / a)) * 100

    print(f"  Joint config (chosen {jc.iloc[0]}/{len(wv)} folds):")
    print(f"    ETS:  {best_params}")
    print(f"    STL:  {best_stl_params}")
    print(f"  MAE={mae_stl:.3f}  RMSE={rmse_stl:.3f}  MAPE={mape_stl:.1f}%")

    print(f"\n  Top 5 param combos:")
    for combo, cnt in jc.head(5).items():
        print(f"    {combo} → {cnt}")

    # =================================================================
    # SAVE
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  SAVING")
    print(f"{'='*60}")

    p1 = OUTPUT_DIR / "cv_results.parquet"
    p2 = OUTPUT_DIR / "cv_summary.parquet"
    p3 = OUTPUT_DIR / "best_config.pkl"

    df_cv.to_parquet(p1, index=False);   print(f"  ✅ {p1}")
    summary.to_parquet(p2);              print(f"  ✅ {p2}")

    bundle = {
        "winning_correction": win_corr,
        "best_params": best_params,
        "best_stl_params": best_stl_params,
        "df_ts_corrected": df_ts_corrected,
        "mae_stl": mae_stl, "rmse_stl": rmse_stl, "mape_stl": mape_stl,
        "joint_counts": jc,
    }
    with open(p3, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  ✅ {p3}")

    print(f"\n  Load in notebook:")
    print(f'    df_cv = pd.read_parquet("{p1}")')
    print(f'    with open("{p3}", "rb") as f: config = pickle.load(f)')


if __name__ == "__main__":
    main()