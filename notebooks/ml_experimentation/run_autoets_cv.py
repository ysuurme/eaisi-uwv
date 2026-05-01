#!/usr/bin/env python3
"""
run_autoets_cv.py — Walk-forward CV with ETS across all SBI segments
                     and COVID correction variants.

Tests 9 ETS model specifications (M-error only) × 6 correction variants
(3 base + 3 winsorized). Model+correction selection uses inner folds
(first 60%); unbiased error is reported on outer folds (last 40%).

Also saves the full prediction matrix for all configs (not just the winner)
to enable downstream DM tests, ensembling, and post-hoc analysis.

Usage:
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

from config import DIR_DB_SILVER
from src.utils.m_query_database import f_query_database

# =============================================================================
# SETTINGS
# =============================================================================
N_WORKERS   = 10
OUTPUT_DIR  = Path("cv_output")
covid_start = pd.Timestamp('2020-03-31')
covid_end   = pd.Timestamp('2022-06-30')
MIN_TRAIN_FRAC = 0.45           # minimum training fraction of series length
MIN_TRAIN_ABS  = 20             # absolute minimum training quarters
INNER_FRAC  = 0.6               # fraction of folds for selection
SEED        = 42                # reproducibility

# ETS specs: Multiplicative-error only (appropriate for positive rates)
# Additive-error specs dropped: theoretically inappropriate for bounded
# positive rates and empirically dominated in prior results.
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
    """
    Build COVID correction variants (non-winsorized only).
    Winsorization is applied per-fold inside the CV worker.

    Only seasonal-mean corrections are used. Interpolation methods (linear,
    cubic, time) were removed because they use the full series for fitting,
    creating data leakage when training windows extend past the COVID period.

    _smean uses only pre-COVID seasonal references to avoid leakage.
    """
    mask = (df_ts.index >= covid_start) & (df_ts.index <= covid_end)

    def _smean(ts, b):
        """Replace COVID quarters with mean of b pre-COVID same-quarter values."""
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


# =============================================================================
# SEASONAL STABILITY DIAGNOSTIC
# =============================================================================

def check_seasonal_stability(df_ts, threshold=0.3):
    """
    Compare seasonal indices pre-COVID vs post-COVID.
    Returns a warning string if seasonal shape has changed materially,
    or None if stable.

    Seasonal index = quarterly mean / overall mean for each period.
    Threshold is the max allowable absolute change in any quarter's index.
    """
    pre = df_ts[df_ts.index < covid_start]
    post = df_ts[df_ts.index > covid_end]

    if len(pre) < 8 or len(post) < 8:  # need at least 2 full years each
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
# WORKER
# =============================================================================

def run_cv_for_correction(corr_label, corr_values, corr_index,
                          orig_values, outer_start, season_length,
                          ets_specs, ets_labels, seed):
    """
    Walk-forward CV for one correction (base + winsorized) × all ETS specs.

    Manual per-fold loop so:
      - Winsorization uses only training data (no look-ahead)
      - Each fold result carries fold_pos (temporal index)
      - No ordering ambiguity

    Returns dict: {base_label: {...}, winsorized_label: {...}}
    """
    np.random.seed(seed)
    warnings.filterwarnings("ignore")
    ts = pd.Series(corr_values, index=corr_index)
    n_folds = len(ts) - outer_start

    all_results = {}

    for is_winsorized in [False, True]:
        variant_label = (f"{corr_label}|winsorized" if is_winsorized
                         else corr_label)
        fold_results = []

        for fold_idx in range(n_folds):
            train_end = outer_start + fold_idx   # exclusive
            test_idx  = train_end                # predict this point

            train_ts = ts.iloc[:train_end].copy()
            if is_winsorized:
                train_ts = winsorize_train(train_ts)

            sf_df = pd.DataFrame({
                "unique_id": variant_label,
                "ds": train_ts.index,
                "y": train_ts.values,
            })

            actual = float(orig_values[test_idx])
            preds = {}

            # Fit each spec individually — one failure shouldn't kill the rest
            for spec_label, spec_model, spec_damped in ets_specs:
                try:
                    m = AutoETS(season_length=season_length, model=spec_model,
                                damped=spec_damped, alias=spec_label)
                    sf = StatsForecast(models=[m], freq="QE", n_jobs=1)
                    fc = sf.forecast(df=sf_df, h=1)
                    if spec_label in fc.columns:
                        p = float(fc[spec_label].iloc[0])
                        preds[spec_label] = p if np.isfinite(p) else np.nan
                    else:
                        preds[spec_label] = np.nan
                except Exception:
                    preds[spec_label] = np.nan

            fold_results.append({
                "fold_pos": fold_idx,
                "date":     ts.index[test_idx],
                "actual":   actual,
                "preds":    preds,
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
    Full CV for one SBI segment.

    Selection on inner folds (fold_pos < n_inner), reporting on outer
    (fold_pos >= n_inner). Uses fold_pos, not list position.

    Saves full prediction matrix for all configs, not just the winner.

    Returns:
        cv_records:       list of dicts — winner-only fold results
        all_pred_records: list of dicts — full prediction matrix (all configs)
        config:           dict or None — winning config metadata
    """
    variants = build_correction_variants(df_ts)
    orig = df_ts.values.ravel().astype(np.float64)

    # Dynamic outer_start: max of absolute minimum and fraction-based minimum
    outer_start = max(MIN_TRAIN_ABS, int(len(df_ts) * MIN_TRAIN_FRAC))
    n_total_folds = len(df_ts) - outer_start

    if n_total_folds < 5:
        print(f"    ⚠️  {sbi_code}: only {n_total_folds} folds with "
              f"outer_start={outer_start}. Skipping.")
        return [], [], None

    seg_start = time.time()

    n_inner = max(1, int(n_total_folds * INNER_FRAC))
    n_outer = n_total_folds - n_inner

    print(f"      Training start: {outer_start} quarters | "
          f"Folds: {n_total_folds} = {n_inner} inner + {n_outer} outer")

    # Seasonal stability check
    seasonal_warn = check_seasonal_stability(df_ts)
    if seasonal_warn:
        print(f"    ⚠️  SEASONAL: {seasonal_warn}")

    # Dispatch
    futures = {}
    for label, ts_c in variants.items():
        f = pool.submit(
            run_cv_for_correction, label,
            ts_c.values.ravel().astype(np.float64), ts_c.index,
            orig, outer_start, 4, ets_specs, ets_labels, SEED,
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
                corr_results[lbl] = {
                    "correction": lbl, "fold_results": [],
                    "error": str(exc),
                }

    n_variants = len(corr_results)

    # Diagnostic: count valid predictions
    total_valid = 0
    total_nan = 0
    for label, wr in corr_results.items():
        if wr.get("error"):
            continue
        for fr in wr.get("fold_results", []):
            valid = sum(1 for v in fr["preds"].values() if pd.notna(v))
            total_valid += valid
            total_nan += len(fr["preds"]) - valid
    print(f"      Diagnostics: {total_valid} valid preds, "
          f"{total_nan} NaN preds across all variants")

    # --- Build FULL prediction matrix (all configs, all folds) ---
    all_pred_records = []
    for label, wr in corr_results.items():
        if wr.get("error") or not wr["fold_results"]:
            continue
        for fr in wr["fold_results"]:
            is_outer = fr["fold_pos"] >= n_inner
            for spec in ets_labels:
                pred = fr["preds"].get(spec, np.nan)
                ok = pd.notna(pred)
                all_pred_records.append({
                    "sbi_code":   sbi_code,
                    "correction": label,
                    "model_spec": spec,
                    "date":       fr["date"],
                    "actual":     fr["actual"],
                    "pred":       pred if ok else np.nan,
                    "abs_error":  abs(fr["actual"] - pred) if ok else np.nan,
                    "fold_set":   "outer" if is_outer else "inner",
                    "fold_pos":   fr["fold_pos"],
                })

    # --- Select best (correction, model_spec) on inner folds ---
    best_inner_mae = np.inf
    best_corr_label = None
    best_model_spec = None
    n_configs_searched = 0

    for label, wr in corr_results.items():
        if wr.get("error") or not wr["fold_results"]:
            continue

        for spec in ets_labels:
            inner_errors = []
            for fr in wr["fold_results"]:
                if fr["fold_pos"] < n_inner:
                    p = fr["preds"].get(spec, np.nan)
                    if pd.notna(p):
                        inner_errors.append(abs(fr["actual"] - p))

            if inner_errors:
                n_configs_searched += 1
                mae_val = np.mean(inner_errors)
                if mae_val < best_inner_mae:
                    best_inner_mae = mae_val
                    best_corr_label = label
                    best_model_spec = spec

    # --- Build winner-only records (backward compatible) ---
    cv_records = []
    outer_errors = []

    if best_corr_label is not None:
        wr = corr_results[best_corr_label]
        for fr in wr["fold_results"]:
            pred = fr["preds"].get(best_model_spec, np.nan)
            ok = pd.notna(pred)
            is_outer = fr["fold_pos"] >= n_inner

            cv_records.append({
                "sbi_code":   sbi_code,
                "correction": best_corr_label,
                "date":       fr["date"],
                "actual":     fr["actual"],
                "pred":       pred if ok else np.nan,
                "abs_error":  abs(fr["actual"] - pred) if ok else np.nan,
                "model_spec": best_model_spec,
                "fold_set":   "outer" if is_outer else "inner",
                "fold_pos":   fr["fold_pos"],
            })

            if is_outer and ok:
                outer_errors.append(abs(fr["actual"] - pred))

    seg_min = (time.time() - seg_start) / 60

    # Structural break detection: rolling MAE with sustained drift
    struct_warn = None
    if best_corr_label and len(outer_errors) >= 4:
        folds = corr_results[best_corr_label]["fold_results"]
        abs_errors_all = []
        for fr in folds:
            p = fr["preds"].get(best_model_spec, np.nan)
            if pd.notna(p):
                abs_errors_all.append(abs(fr["actual"] - p))

        if len(abs_errors_all) >= 8:
            # Rolling window of 4 quarters
            errs = pd.Series(abs_errors_all)
            rolling_mae = errs.rolling(4, min_periods=4).mean().dropna()
            if len(rolling_mae) >= 2:
                early_mae = rolling_mae.iloc[:len(rolling_mae)//2].mean()
                late_mae = rolling_mae.iloc[len(rolling_mae)//2:].mean()
                if early_mae > 0:
                    ratio = late_mae / early_mae
                    if ratio > 1.8 or ratio < 0.55:
                        struct_warn = (f"Rolling MAE drift: late/early = "
                                       f"{ratio:.2f} (early={early_mae:.3f}, "
                                       f"late={late_mae:.3f})")
                        print(f"    ⚠️  BREAK: {struct_warn}")

    print(f"    ⏱️  {sbi_code}: {n_variants} variants × "
          f"{n_total_folds} folds × {len(ets_labels)} specs "
          f"in {seg_min:.1f} min")

    if best_corr_label is None:
        return cv_records, all_pred_records, None

    outer_mae = np.mean(outer_errors) if outer_errors else np.nan

    # Inner vs outer MAE ratio diagnostic
    inner_outer_ratio = (best_inner_mae / outer_mae
                         if outer_mae and outer_mae > 0 else np.nan)

    config = {
        "winning_correction":       best_corr_label,
        "df_ts_corrected":          _apply_winsorization_if_needed(
            variants, best_corr_label, df_ts),
        "correction_is_winsorized": "|winsorized" in best_corr_label,
        "best_model_spec":          best_model_spec,
        "inner_mae":                float(best_inner_mae),
        "outer_mae":                float(outer_mae),
        "inner_outer_ratio":        float(inner_outer_ratio)
                                    if np.isfinite(inner_outer_ratio) else None,
        "n_inner_folds":            n_inner,
        "n_outer_folds":            n_outer,
        "n_configs_searched":       n_configs_searched,
        "structural_break_warning": struct_warn,
        "seasonal_shift_warning":   seasonal_warn,
    }

    ratio_str = (f"{inner_outer_ratio:.2f}"
                 if isinstance(inner_outer_ratio, float)
                 and np.isfinite(inner_outer_ratio) else "n/a")
    print(f"    🏆 {best_corr_label} | {best_model_spec} | "
          f"inner={best_inner_mae:.4f} outer={outer_mae:.4f} "
          f"ratio={ratio_str}")
    return cv_records, all_pred_records, config


def _apply_winsorization_if_needed(variants, best_corr_label, df_ts_orig):
    """
    FIX: When the winning config is a winsorized variant, apply winsorization
    to the stored corrected series so that the downstream final model fit
    is consistent with what was selected during CV.

    Note: CV used per-fold winsorization (growing training window → different
    quantile boundaries per fold). Here we winsorize the full series, which
    is a pragmatic approximation. The quantile boundaries may differ slightly
    from what any individual CV fold used, but this is preferable to the
    original behavior of not winsorizing at all.
    """
    base_label = best_corr_label.split("|winsorized")[0]
    ts_corrected = variants.get(base_label, df_ts_orig).copy()

    if "|winsorized" in best_corr_label:
        lo = ts_corrected.quantile(0.05)
        hi = ts_corrected.quantile(0.95)
        ts_corrected = ts_corrected.clip(lower=lo, upper=hi)

    return ts_corrected


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ETS walk-forward CV across SBI segments"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--segment", type=str, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    warnings.filterwarnings("ignore")
    np.random.seed(SEED)

    print("=" * 60)
    print("  LOADING DATA")
    print("=" * 60)

    query = """
    SELECT
        Perioden as timeperiod_text,
        BedrijfskenmerkenSBI2008 as sbi_code,
        BedrijfskenmerkenSBI2008_Title as sbi_title,
        DATE(printf('%s-%s-01',
            substr(Perioden, 1, 4),
            CASE substr(Perioden, 7, 2)
                WHEN '01' THEN '01' WHEN '02' THEN '04'
                WHEN '03' THEN '07' WHEN '04' THEN '10'
            END), '+3 months', '-1 day'
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
    df_org["absenteeism_perc"] = pd.to_numeric(
        df_org["absenteeism_perc"], errors="coerce"
    )
    df_org["period_enddate"] = pd.to_datetime(df_org["period_enddate"])

    codes = sorted(df_org["sbi_code"].unique())
    if args.segment:
        if args.segment not in codes:
            print(f"  ❌ '{args.segment}' not found. Available: {codes}")
            return
        codes = [args.segment]

    # --- Build segments with frequency validation ---
    segments = {}

    for c in codes:
        sub = (df_org[df_org["sbi_code"] == c]
               .sort_values("period_enddate")
               .drop_duplicates("period_enddate")
               .set_index("period_enddate")["absenteeism_perc"])

        raw_dates = sub.index.copy()

        # Normalize to QuarterEnd
        try:
            sub.index = sub.index + pd.offsets.QuarterEnd(0)
            sub = sub[~sub.index.duplicated(keep='last')]
        except Exception:
            print(f"  ⚠️  Skipping {c}: failed to normalize to QE")
            continue

        # Validate quarterly spacing
        inferred = pd.infer_freq(sub.index)
        if inferred is not None:
            try:
                offset = pd.tseries.frequencies.to_offset(inferred)
                if not isinstance(offset, pd.offsets.QuarterEnd):
                    print(f"  ⚠️  Skipping {c}: freq '{inferred}' not QE")
                    continue
            except ValueError:
                # Fallback: check median day spacing
                diffs = sub.index.to_series().diff().dropna()
                median_days = diffs.dt.days.median()
                if not (80 <= median_days <= 100):
                    print(f"  ⚠️  Skipping {c}: spacing {median_days:.0f}d")
                    continue
        else:
            diffs = sub.index.to_series().diff().dropna()
            median_days = diffs.dt.days.median()
            if not (80 <= median_days <= 100):
                print(f"  ⚠️  Skipping {c}: spacing {median_days:.0f}d")
                continue

        ts = sub.asfreq("QE")

        # Handle NaNs from QE normalization
        if ts.isna().any():
            na_dates = ts.index[ts.isna()]
            truly_missing = []
            for nad in na_dates:
                if (np.abs(raw_dates - nad)).min() > pd.Timedelta(days=5):
                    truly_missing.append(nad)
            if truly_missing:
                print(f"  ⚠️  Skipping {c}: {len(truly_missing)} missing Qs")
                continue
            for nad in na_dates:
                nearest_i = (np.abs(raw_dates - nad)).argmin()
                ts[nad] = df_org.loc[
                    (df_org["sbi_code"] == c) &
                    (df_org["period_enddate"] == raw_dates[nearest_i]),
                    "absenteeism_perc"
                ].iloc[0]
            if ts.isna().any():
                print(f"  ⚠️  Skipping {c}: unresolvable NaN")
                continue

        # Minimum length: need enough for training + at least 5 folds
        min_len = MIN_TRAIN_ABS + 5
        if len(ts) < min_len:
            print(f"  ⚠️  Skipping {c} ({len(ts)} obs, need {min_len})")
            continue

        segments[c] = ts

    if not segments:
        print("  No valid segments.")
        return

    # --- Pre-run summary ---
    n_base = 3  # no_correction + 2×seasonal_mean
    n_total = n_base * 2  # ×2 for winsorized

    print(f"\n  Segments: {len(segments)} | ETS specs: {len(ETS_SPECS)} | "
          f"Correction variants: {n_total}")
    print(f"  Configs/segment: {n_total * len(ETS_SPECS)}")
    for c, ts in segments.items():
        os_val = max(MIN_TRAIN_ABS, int(len(ts) * MIN_TRAIN_FRAC))
        n_folds = len(ts) - os_val
        n_in = max(1, int(n_folds * INNER_FRAC))
        print(f"    {c}: {len(ts)} obs, {n_folds} folds "
              f"({n_in} inner + {n_folds - n_in} outer)")

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
            print(f"\n{'='*60}")
            print(f"  SEGMENT {i+1}/{len(segments)}: {code}")
            print(f"{'='*60}")
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
    df_all_preds = pd.DataFrame(all_pred_records)

    if len(df_cv) == 0:
        print("  No results.")
        return

    # --- Summary stats per fold set ---
    def _agg(g):
        v = g.dropna(subset=["pred"])
        if len(v) == 0:
            return pd.Series({"mean_mae": np.nan, "mean_rmse": np.nan,
                              "mape": np.nan, "n_folds": 0})
        e = v["pred"] - v["actual"]
        return pd.Series({
            "mean_mae":  v["abs_error"].mean(),
            "mean_rmse": np.sqrt(np.mean(e**2)),
            "mape":      (np.abs(e / v["actual"]) * 100).mean(),
            "n_folds":   len(v),
        })

    for fsl in ["inner", "outer"]:
        subset = df_cv[df_cv["fold_set"] == fsl]
        if len(subset) > 0:
            s = (subset.groupby(["sbi_code","correction"], group_keys=False)
                 .apply(_agg).sort_values(["sbi_code","mean_mae"]))
            p = OUTPUT_DIR / f"autoets_cv_summary_{fsl}.parquet"
            s.to_parquet(p); print(f"\n  ✅ {p}")

    # --- Results table ---
    print(f"\n{'='*60}")
    print(f"  RESULTS — {total_min:.1f} min")
    print(f"{'='*60}\n")
    print(f"  {'Seg':<12} {'Correction':<30} {'Spec':<8} "
          f"{'Inner':>7} {'Outer':>7} {'Ratio':>7} {'Warnings'}")
    print(f"  {'-'*90}")
    for c, cfg in sorted(all_configs.items()):
        warns = []
        if cfg.get("structural_break_warning"):
            warns.append("BREAK")
        if cfg.get("seasonal_shift_warning"):
            warns.append("SEASON")
        ratio_str = (f"{cfg['inner_outer_ratio']:.2f}"
                     if cfg.get('inner_outer_ratio') else "n/a")
        warn_str = ", ".join(warns) if warns else ""
        print(f"  {c:<12} {cfg['winning_correction']:<30} "
              f"{cfg['best_model_spec']:<8} "
              f"{cfg['inner_mae']:>7.4f} {cfg['outer_mae']:>7.4f} "
              f"{ratio_str:>7} {warn_str}")

    # --- Save ---
    p1 = OUTPUT_DIR / "autoets_cv_results.parquet"
    p2 = OUTPUT_DIR / "autoets_cv_all_predictions.parquet"
    p3_json = OUTPUT_DIR / "autoets_best_configs.json"
    p3_ts   = OUTPUT_DIR / "autoets_corrected_series.parquet"

    df_cv.to_parquet(p1, index=False)
    print(f"\n  ✅ {p1}")

    df_all_preds.to_parquet(p2, index=False)
    print(f"  ✅ {p2} ({len(df_all_preds)} rows)")

    configs_json = {}
    corrected_series = {}
    for code, cfg in all_configs.items():
        ts_corr = cfg.pop("df_ts_corrected")
        corrected_series[code] = ts_corr
        configs_json[code] = {
            k: (float(v) if isinstance(v, np.floating)
                else int(v) if isinstance(v, np.integer) else v)
            for k, v in cfg.items()
        }

    with open(p3_json, "w") as f:
        json.dump(configs_json, f, indent=2, default=str)
    print(f"  ✅ {p3_json}")

    ts_frames = []
    for code, ts in corrected_series.items():
        fr = ts.reset_index()
        fr.columns = ["period_enddate", "absenteeism_perc"]
        fr["sbi_code"] = code
        ts_frames.append(fr)
    if ts_frames:
        pd.concat(ts_frames, ignore_index=True).to_parquet(p3_ts, index=False)
        print(f"  ✅ {p3_ts}")

    print(f'\n  Load:\n'
          f'    df_cv = pd.read_parquet("{p1}")\n'
          f'    df_all = pd.read_parquet("{p2}")\n'
          f'    configs = json.load(open("{p3_json}"))\n'
          f'    # Unbiased: df_cv[df_cv["fold_set"]=="outer"]')


if __name__ == "__main__":
    main()