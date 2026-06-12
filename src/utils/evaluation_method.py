"""
evaluation_method.py
====================
Unified, methodology-aware comparison of point and probabilistic forecasts
produced by four different model families:

  1. The CBS-feature ML pipeline (Ridge / ElasticNet / PLS / HistGBR)
  2. AutoETS  (from run_autoets_cv.py)
  3. STL+ETS  (from run_nested_cv.py + the per-sector STL notebook)
  4. Chronos-Bolt foundation model (from the foundational-models notebook)

The brutal truth this module is built around
--------------------------------------------
These four experiments were NOT run on the same evaluation contract.  They
differ on forecast horizon (h=1,2,3 vs h=4), origin count, sector coverage,
prediction-interval support, and metric computation conventions.  A naïve
`pd.concat([...]).groupby('model').agg('mean')` will give an answer, but
it will be a wrong answer — biased by whichever model happened to forecast
the easiest (sector, target_date, horizon) combinations.

This module solves that by:
  (a) Normalising every model's native output to one canonical schema.
  (b) Aligning on the intersection of (sector, target_date, horizon) so
      every comparison metric is computed on a common set of points.
  (c) Reporting both the *aligned* (apples-to-apples) results AND the
      *native* (full-coverage) results so we know what got dropped.
  (d) Pairwise Diebold-Mariano tests, calibration of prediction intervals
      where supported, regime-split (pre-/post-2023) metrics, and a
      multi-criteria scorecard for the business-owner decision.

How to use this module
----------------------
Typical workflow at the end of the project:

    from evaluation_method import (
        load_pipeline_predictions, load_autoets_predictions,
        load_stl_ets_predictions,  load_chronos_predictions,
        compare_all_models,
    )

    pipeline = load_pipeline_predictions(eval_db_path="data/4_eval/eval_data.db")
    autoets  = load_autoets_predictions(
        winner_parquet="cv_output/autoets_cv_results.parquet",
        all_preds_parquet="cv_output/autoets_cv_all_predictions.parquet",
    )
    stl_ets  = load_stl_ets_predictions(
        results_parquet="cv_output/cv_results.parquet",
        sector_code="T001081",
    )
    chronos  = load_chronos_predictions(backtest_parquet="cv_output/chronos_backtest.parquet")

    report = compare_all_models([pipeline, autoets, stl_ets, chronos],
                                covid_end=pd.Timestamp("2022-12-31"),
                                output_dir="evaluation_output/")

The `report` object holds aligned metrics, scorecards, DM matrices, and
calibration tables for the write-up.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Canonical schema
# ----------------------------------------------------------------------------
# Every loader normalises its model's native output to this set of columns.
# Optional PI columns are present as NaN-filled when the model can't produce
# them, so downstream calibration code can detect support uniformly.
CANONICAL_COLS = [
    "model_name",      # str — e.g. "Pipeline_Ridge", "AutoETS_MAdM"
    "sector_code",     # str — SBI code or aggregate ("T001081")
    "origin_date",     # Timestamp — last training observation date
    "target_date",     # Timestamp — quarter being forecast
    "horizon",         # int — quarters ahead (1..4)
    "y_true",          # float — actual observed value
    "y_pred",          # float — point forecast (median for probabilistic)
    "y_lower_80",      # float | NaN — 80% PI lower bound
    "y_upper_80",      # float | NaN — 80% PI upper bound
    "y_lower_95",      # float | NaN — 95% PI lower bound
    "y_upper_95",      # float | NaN — 95% PI upper bound
]


def _empty_canonical() -> pd.DataFrame:
    """Empty DataFrame matching CANONICAL_COLS for safe concatenation."""
    df = pd.DataFrame(columns=CANONICAL_COLS)
    # Pre-cast dtypes so empty frames concat cleanly with populated ones
    df["origin_date"] = pd.to_datetime(df["origin_date"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["horizon"] = df["horizon"].astype("Int64")
    return df


def _validate_canonical(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    """
    Enforce the canonical schema: required columns present, dtypes consistent,
    no duplicate (sector, target_date, horizon) triples (within a single model
    each forecast must be unique on that key).
    """
    missing = [c for c in CANONICAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{model_label}] missing required columns: {missing}")

    df = df.copy()
    df["origin_date"] = pd.to_datetime(df["origin_date"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["horizon"] = df["horizon"].astype(int)
    df["sector_code"] = df["sector_code"].astype(str)

    dup_keys = df.duplicated(subset=["model_name", "sector_code", "target_date", "horizon"])
    if dup_keys.any():
        n_dup = int(dup_keys.sum())
        # Soft warn rather than raise — sometimes a model has multiple variants
        # and the loader picks the winner; if loader bugged we surface it here.
        warnings.warn(
            f"[{model_label}] {n_dup} duplicate (sector, target_date, horizon) "
            f"rows — keeping first occurrence.",
            stacklevel=2,
        )
        df = df[~dup_keys].copy()

    return df[CANONICAL_COLS]


# ============================================================================
# LOADERS — one per model family
# ============================================================================

def load_pipeline_predictions(
    eval_db_path: str | Path,
    table: str = "model_evaluation_records",
    run_filter: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Load predictions from the CBS-feature ML pipeline.

    Reads from the SQLite eval database populated by ``ml_6_model_validation``.
    For the per-sector evaluation, we want each sector's *best* preset+model
    combination (per the experiment design), so the loader filters to one
    winning row per (sector, target_date).

    The pipeline stores predictions at fold-level granularity (one row per
    sector × fold × horizon-step).  For h=4 with 5 origins, that's 20 rows
    per (sector, model_id).  We expect the upstream evaluator to have
    materialised actual+predicted on the per-step level — if it stored only
    aggregates we cannot use this comparison; raise loudly.

    Parameters
    ----------
    eval_db_path : path to ``eval_data.db``
    table : table name; default matches the orchestrator's writer
    run_filter : optional dict of column→value pairs to restrict the rows
                 (e.g. ``{"experiment_name": "overnight_2026_06_07"}``)

    Returns
    -------
    DataFrame in canonical schema.  ``model_name`` encodes the winning model
    family per sector (e.g. ``Pipeline_Ridge``, ``Pipeline_ElasticNet``).
    Prediction intervals are NaN since the base pipeline does not produce them.
    """
    from sqlalchemy import create_engine

    eval_db_path = Path(eval_db_path)
    if not eval_db_path.exists():
        raise FileNotFoundError(f"Eval DB not found: {eval_db_path}")

    engine = create_engine(f"sqlite:///{eval_db_path.as_posix()}")
    df = pd.read_sql_table(table, engine)

    if run_filter:
        for k, v in run_filter.items():
            df = df[df[k] == v]

    # Expected columns in the eval table — adjust if your schema differs
    required = ["sector_code", "model_name", "fold_origin_date",
                "target_date", "horizon_step", "y_true", "y_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Pipeline eval table missing required columns: {missing}. "
            f"Available: {list(df.columns)}. "
            f"Ensure ml_6_model_validation writes per-step predictions, not "
            f"only aggregate metrics."
        )

    out = pd.DataFrame({
        "model_name":   "Pipeline_" + df["model_name"].astype(str),
        "sector_code":  df["sector_code"].astype(str),
        "origin_date":  pd.to_datetime(df["fold_origin_date"]),
        "target_date":  pd.to_datetime(df["target_date"]),
        "horizon":      df["horizon_step"].astype(int),
        "y_true":       df["y_true"].astype(float),
        "y_pred":       df["y_pred"].astype(float),
        "y_lower_80":   np.nan,
        "y_upper_80":   np.nan,
        "y_lower_95":   np.nan,
        "y_upper_95":   np.nan,
    })

    # Per-sector winner selection: keep one model per (sector, target_date, horizon)
    # chosen by lowest absolute error on the corresponding fold.  This mirrors
    # how the user reports "best_model" per sector in their final tables.
    out["_ae"] = (out["y_pred"] - out["y_true"]).abs()
    out = (out.sort_values(["sector_code", "target_date", "horizon", "_ae"])
              .drop_duplicates(subset=["sector_code", "target_date", "horizon"],
                               keep="first")
              .drop(columns="_ae"))

    return _validate_canonical(out, "Pipeline")


def load_autoets_predictions(
    canonical_parquet: str | Path,
) -> pd.DataFrame:
    """
    Load AutoETS predictions from run_autoets_h4.py output.

    The script writes canonical schema directly (one row per (sector,
    target_date, horizon) for the per-sector-winning config on outer folds).
    This loader is a thin reader + validator.

    Parameters
    ----------
    canonical_parquet : path to ``autoets_predictions.parquet``
    """
    df = pd.read_parquet(canonical_parquet)
    return _validate_canonical(df, "AutoETS")


def load_stl_ets_predictions(
    canonical_parquet: str | Path,
) -> pd.DataFrame:
    """
    Load STL+ETS predictions from run_stl_ets_h4.py output.

    The script writes canonical schema directly.  This loader is a thin
    reader + validator.

    Parameters
    ----------
    canonical_parquet : path to ``stl_ets_predictions.parquet``
    """
    df = pd.read_parquet(canonical_parquet)
    return _validate_canonical(df, "STL_ETS")


def load_chronos_predictions(
    canonical_parquet: str | Path,
) -> pd.DataFrame:
    """
    Load Chronos-Bolt walk-forward predictions from run_chronos_h4.py output.

    The script writes canonical schema directly with q10/q90 → 80% PI and
    q02/q97 → 95% PI columns populated.  This loader is a thin reader.

    Parameters
    ----------
    canonical_parquet : path to ``chronos_predictions.parquet``
    """
    df = pd.read_parquet(canonical_parquet)
    return _validate_canonical(df, "Chronos_Bolt")


def load_external_predictions(
    parquet_path: str | Path,
    model_name: str,
    column_mapping: dict,
    horizon_value: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generic loader for collaborator-supplied predictions (e.g. ARIMA).

    Accepts a parquet/CSV with non-canonical column names and a mapping dict
    that translates each canonical field to the source column name.  Missing
    canonical fields can be filled with constants (e.g. horizon_value=1).

    Example:
        load_external_predictions(
            "arima_predictions.parquet",
            model_name="ARIMA",
            column_mapping={
                "sector_code": "sbi",
                "target_date": "fcst_date",
                "y_true": "actual",
                "y_pred": "forecast",
            },
            horizon_value=1,
        )
    """
    parquet_path = Path(parquet_path)
    if parquet_path.suffix == ".csv":
        df = pd.read_csv(parquet_path)
    else:
        df = pd.read_parquet(parquet_path)

    out = pd.DataFrame()
    out["model_name"] = [model_name] * len(df)

    for canon_col in ["sector_code", "origin_date", "target_date", "horizon",
                      "y_true", "y_pred",
                      "y_lower_80", "y_upper_80", "y_lower_95", "y_upper_95"]:
        src = column_mapping.get(canon_col)
        if src is not None and src in df.columns:
            out[canon_col] = df[src]
        elif canon_col == "horizon" and horizon_value is not None:
            out[canon_col] = horizon_value
        elif canon_col == "origin_date":
            # Derive from target_date if not provided (assume h=1)
            if "target_date" in out.columns:
                out[canon_col] = pd.to_datetime(out["target_date"]) \
                                  - pd.tseries.offsets.QuarterEnd()
            else:
                out[canon_col] = pd.NaT
        else:
            out[canon_col] = np.nan

    return _validate_canonical(out, model_name)


# ============================================================================
# ALIGNMENT — the heart of fair comparison
# ============================================================================

@dataclass
class AlignmentReport:
    """Diagnostic about what survived alignment and what got dropped."""
    n_models: int
    n_aligned_rows: int
    per_model_native: dict             # model_name → row count in its own data
    per_model_aligned: dict            # model_name → row count after intersection
    aligned_sectors: set
    aligned_horizons: set
    dropped_keys: dict                 # model_name → row count dropped
    intersection_key: tuple = ("sector_code", "target_date", "horizon")

    def __str__(self):
        lines = [
            f"=== Alignment report ===",
            f"  intersection key: {self.intersection_key}",
            f"  models aligned:   {self.n_models}",
            f"  aligned rows:     {self.n_aligned_rows}",
            f"  aligned sectors:  {len(self.aligned_sectors)}",
            f"  aligned horizons: {sorted(self.aligned_horizons)}",
            f"",
            f"  per-model coverage (native → aligned):",
        ]
        for m in self.per_model_native:
            n_nat = self.per_model_native[m]
            n_ali = self.per_model_aligned.get(m, 0)
            pct = 100 * n_ali / n_nat if n_nat > 0 else 0
            lines.append(f"    {m:30s}: {n_nat:6d} → {n_ali:6d}  ({pct:.1f}% retained)")
        return "\n".join(lines)


def align_predictions(
    model_dfs: Sequence[pd.DataFrame],
    on: tuple = ("sector_code", "target_date", "horizon"),
) -> tuple[pd.DataFrame, AlignmentReport]:
    """
    Restrict every model's predictions to the intersection of (sector,
    target_date, horizon) tuples present across ALL models.

    Returns
    -------
    aligned_df : long-format DataFrame with one row per (model, sector,
                 target_date, horizon), all keys present across every model.
    report : AlignmentReport with coverage diagnostics.

    Why intersection (and not union)
    --------------------------------
    Comparing models on different subsets of dates is not a comparison — it
    rewards the model that happened to forecast the easier dates.  The fair
    comparison computes every metric on the SAME (sector, target_date, h)
    triples that all models attempted.

    If you also want to know how a model performs on its full native data
    (e.g. Chronos covered fewer sectors but more origins), use
    ``compute_native_metrics`` separately.
    """
    if not model_dfs:
        raise ValueError("No model DataFrames provided")

    per_model_native = {df["model_name"].iloc[0]: len(df) for df in model_dfs}

    # Build the set of keys present in every model
    key_sets = [
        set(map(tuple, df[list(on)].values.tolist()))
        for df in model_dfs
    ]
    common_keys = set.intersection(*key_sets)

    if not common_keys:
        raise ValueError(
            "No keys (sector, target_date, horizon) are shared across all "
            "models.  Check horizon settings — most models forecast h=1 while "
            "the pipeline forecasts h=4 by default."
        )

    # Filter each model to the intersection
    filtered = []
    per_model_aligned = {}
    dropped = {}
    for df in model_dfs:
        keys = list(map(tuple, df[list(on)].values.tolist()))
        mask = np.array([k in common_keys for k in keys])
        kept = df[mask].copy()
        name = df["model_name"].iloc[0]
        per_model_aligned[name] = len(kept)
        dropped[name] = len(df) - len(kept)
        filtered.append(kept)

    aligned = pd.concat(filtered, ignore_index=True)
    aligned = aligned.sort_values(["model_name", "sector_code", "target_date", "horizon"])

    report = AlignmentReport(
        n_models=len(model_dfs),
        n_aligned_rows=len(aligned),
        per_model_native=per_model_native,
        per_model_aligned=per_model_aligned,
        aligned_sectors=set(aligned["sector_code"].unique()),
        aligned_horizons=set(aligned["horizon"].unique()),
        dropped_keys=dropped,
        intersection_key=on,
    )
    return aligned, report


# ============================================================================
# METRICS — point-forecast accuracy
# ============================================================================

def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _mape(y_true, y_pred):
    eps = 1e-9  # avoid division by zero on truly missing series
    return float(np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + eps))) * 100)


def _r2(y_true, y_pred):
    """R² that handles the zero-variance edge case."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def _bias(y_true, y_pred):
    return float(np.mean(y_pred - y_true))


def _directional_accuracy(y_true, y_pred):
    """% of predictions that get the sign of change right."""
    if len(y_true) < 2:
        return float("nan")
    dt = np.sign(np.diff(y_true))
    dp = np.sign(np.diff(y_pred))
    return float(np.mean(dt == dp) * 100)


def compute_point_metrics(df: pd.DataFrame, group_by: list[str] = None) -> pd.DataFrame:
    """
    Compute point-forecast metrics with optional grouping.

    Each metric is computed on rows with non-NaN y_pred.  Models that
    couldn't predict a given point contribute zero rows to that group.
    """
    if group_by is None:
        group_by = ["model_name"]

    def _agg(g):
        v = g.dropna(subset=["y_pred", "y_true"])
        if len(v) == 0:
            return pd.Series({"n": 0, "MAE": np.nan, "RMSE": np.nan,
                              "MAPE": np.nan, "R2": np.nan, "bias": np.nan,
                              "dir_acc": np.nan})
        return pd.Series({
            "n":       len(v),
            "MAE":     _mae(v["y_true"].values, v["y_pred"].values),
            "RMSE":    _rmse(v["y_true"].values, v["y_pred"].values),
            "MAPE":    _mape(v["y_true"].values, v["y_pred"].values),
            "R2":      _r2(v["y_true"].values, v["y_pred"].values),
            "bias":    _bias(v["y_true"].values, v["y_pred"].values),
            "dir_acc": _directional_accuracy(v["y_true"].values, v["y_pred"].values),
        })

    return (df.groupby(group_by, dropna=False).apply(_agg, include_groups=False)
              .reset_index())


def compute_regime_metrics(
    df: pd.DataFrame,
    regime_split_date: pd.Timestamp = pd.Timestamp("2023-01-01"),
) -> pd.DataFrame:
    """
    Pre- vs post-regime metrics.  Default split at 2023-01-01 to match the
    user's existing pre2023/post2023 bins.
    """
    df = df.copy()
    df["regime"] = np.where(df["target_date"] < regime_split_date,
                            "pre_2023", "post_2023")
    return compute_point_metrics(df, group_by=["model_name", "regime"])


def compute_per_sector_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Metrics by (model, sector_code) — the per-sector heatmap source."""
    return compute_point_metrics(df, group_by=["model_name", "sector_code"])


def compute_per_horizon_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Metrics by (model, horizon) — shows how each model decays with h."""
    return compute_point_metrics(df, group_by=["model_name", "horizon"])


def compute_skill_score(
    df: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    baseline_name: str = "rolling_mean",
) -> pd.DataFrame:
    """
    Skill score relative to a baseline (rolling-mean, naïve, or random walk).

    Skill = 1 − MAE_model / MAE_baseline.  Skill > 0 means the model is better
    than the baseline; the magnitude is the proportional reduction in error.

    ``baseline_predictions`` must be in canonical schema and share the same
    aligned (sector, target_date, horizon) keys as ``df``.
    """
    # Inner join baseline onto each row
    merged = df.merge(
        baseline_predictions[["sector_code", "target_date", "horizon", "y_pred"]]
            .rename(columns={"y_pred": "y_baseline"}),
        on=["sector_code", "target_date", "horizon"],
        how="inner",
    )
    if len(merged) == 0:
        warnings.warn("No overlap between predictions and baseline — skill not computable")
        return pd.DataFrame()

    merged["ae_model"]    = (merged["y_pred"]     - merged["y_true"]).abs()
    merged["ae_baseline"] = (merged["y_baseline"] - merged["y_true"]).abs()

    g = merged.groupby("model_name", dropna=False)
    out = pd.DataFrame({
        "n":            g.size(),
        "MAE_model":    g["ae_model"].mean(),
        "MAE_baseline": g["ae_baseline"].mean(),
    }).reset_index()
    out["skill_score"] = 1 - out["MAE_model"] / out["MAE_baseline"]
    out["baseline_name"] = baseline_name
    return out.sort_values("skill_score", ascending=False)


# ============================================================================
# PROBABILISTIC METRICS — calibration of prediction intervals
# ============================================================================

def compute_pi_coverage(
    df: pd.DataFrame,
    levels: tuple = (0.80, 0.95),
) -> pd.DataFrame:
    """
    Empirical coverage of nominal prediction intervals.

    Coverage = % of actuals that fell inside the model's predicted interval.
    A calibrated 80% PI should have ~80% empirical coverage.

    Models without PI support (NaN lower/upper bounds) are excluded from this
    output with a warning, since coverage is undefined for them.
    """
    records = []
    for model, sub in df.groupby("model_name"):
        for level in levels:
            lo_col = f"y_lower_{int(level*100)}"
            hi_col = f"y_upper_{int(level*100)}"
            valid = sub.dropna(subset=[lo_col, hi_col, "y_true"])
            if len(valid) == 0:
                # Model doesn't produce this PI level
                continue
            in_interval = ((valid["y_true"] >= valid[lo_col])
                         & (valid["y_true"] <= valid[hi_col]))
            records.append({
                "model_name": model,
                "nominal":    level,
                "observed":   float(in_interval.mean()),
                "n":          len(valid),
                "mean_width": float((valid[hi_col] - valid[lo_col]).mean()),
            })
    out = pd.DataFrame(records)
    if not out.empty:
        out["miscalibration"] = out["observed"] - out["nominal"]
    return out


def compute_crps_quantile_approx(
    df: pd.DataFrame,
    quantile_cols: dict = None,
) -> pd.DataFrame:
    """
    Approximate CRPS from a small set of quantile predictions.

    For each row, approximates the CDF as a step function at the supplied
    quantile levels and integrates the squared distance to the empirical CDF
    (1 if y ≤ z, 0 otherwise).  More quantiles → better approximation.

    Only computable for models that supplied prediction intervals.
    """
    if quantile_cols is None:
        # Default: use 80% PI bounds + median (y_pred) as the 0.5 quantile
        quantile_cols = {0.10: "y_lower_80", 0.50: "y_pred", 0.90: "y_upper_80"}
    sorted_levels = sorted(quantile_cols.keys())

    def _row_crps(row, y_true):
        # Quantile values from the row
        qs = np.array([row[quantile_cols[lvl]] for lvl in sorted_levels])
        if np.any(np.isnan(qs)):
            return np.nan
        levels = np.array(sorted_levels)
        # CRPS = ∫ (F(z) - 1{y≤z})² dz approximated as Riemann sum
        # over the discrete quantile points.
        contrib = 0.0
        for i in range(len(levels) - 1):
            z_lo, z_hi = qs[i], qs[i + 1]
            F_avg = (levels[i] + levels[i + 1]) / 2
            indicator = float(y_true <= (z_lo + z_hi) / 2)
            contrib += (F_avg - indicator) ** 2 * abs(z_hi - z_lo)
        return contrib

    records = []
    for model, sub in df.groupby("model_name"):
        if any(sub[c].isna().all() for c in quantile_cols.values()):
            continue  # Model lacks the quantile predictions
        crps_vals = []
        for _, row in sub.iterrows():
            v = _row_crps(row, row["y_true"])
            if not np.isnan(v):
                crps_vals.append(v)
        if crps_vals:
            records.append({
                "model_name": model,
                "n": len(crps_vals),
                "CRPS_approx": float(np.mean(crps_vals)),
            })
    return pd.DataFrame(records).sort_values("CRPS_approx") if records else pd.DataFrame()


# ============================================================================
# STATISTICAL TESTS — Diebold-Mariano
# ============================================================================

def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    horizon: int = 1,
    loss: str = "squared",
) -> dict:
    """
    Two-sided Diebold-Mariano test on paired forecast errors.

    Tests H0: E[loss(e_a) - loss(e_b)] = 0  (the two models have equal
    expected forecast accuracy) vs H1: not equal.

    Uses the Harvey, Leybourne & Newbold (1997) small-sample correction.
    Heteroskedasticity- and autocorrelation-consistent (HAC) variance is
    estimated with truncation lag = h-1.

    Returns
    -------
    dict with keys ``dm_stat``, ``p_value``, ``n``, ``loss_diff_mean``,
    interpretation note.
    """
    from scipy.stats import norm
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    if errors_a.shape != errors_b.shape:
        raise ValueError("errors_a and errors_b must be the same length")

    mask = np.isfinite(errors_a) & np.isfinite(errors_b)
    if mask.sum() < 8:
        return {"dm_stat": np.nan, "p_value": np.nan, "n": int(mask.sum()),
                "loss_diff_mean": np.nan,
                "note": "fewer than 8 paired observations — test not reliable"}

    if loss == "squared":
        d = errors_a[mask] ** 2 - errors_b[mask] ** 2
    elif loss == "absolute":
        d = np.abs(errors_a[mask]) - np.abs(errors_b[mask])
    else:
        raise ValueError(f"unknown loss '{loss}', use 'squared' or 'absolute'")

    n = len(d)
    d_mean = float(np.mean(d))

    # HAC variance with truncation lag = h-1
    gamma0 = float(np.var(d, ddof=0))
    gammas = [gamma0]
    for k in range(1, horizon):
        cov = float(np.mean((d[k:] - d_mean) * (d[:-k] - d_mean)))
        gammas.append(cov)
    long_run_var = gammas[0] + 2 * sum(gammas[1:])
    if long_run_var <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan, "n": n,
                "loss_diff_mean": d_mean,
                "note": "non-positive long-run variance estimate"}

    dm = d_mean / np.sqrt(long_run_var / n)

    # Harvey small-sample correction
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_hln = dm * correction

    p = 2 * (1 - norm.cdf(abs(dm_hln)))
    direction = "A worse than B" if d_mean > 0 else ("A better than B" if d_mean < 0 else "tied")
    return {
        "dm_stat": float(dm_hln),
        "p_value": float(p),
        "n":       n,
        "loss_diff_mean": d_mean,
        "note":    f"two-sided HLN-corrected DM, loss={loss}, direction={direction}",
    }


def pairwise_dm_matrix(
    aligned_df: pd.DataFrame,
    alpha: float = 0.05,
    horizon: int = 1,
    loss: str = "squared",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise DM tests between every pair of models on the aligned data.

    Returns
    -------
    p_value_matrix : DataFrame indexed by model_name × model_name with p-values
    win_loss_tie : DataFrame counting per-pair (wins, losses, ties) where a
                   "win" for model A vs B is a (sector, target_date, horizon)
                   where A's squared error is strictly lower than B's by a
                   significant margin in a per-sector DM test.

    The win_loss_tie matrix is reported per pair across all (sector,
    target_date, horizon) keys — the per-sector DM with alpha = 0.05 is
    typical convention for forecasting comparisons.
    """
    models = sorted(aligned_df["model_name"].unique())
    n_models = len(models)

    # Pivot to wide form: rows are (sector, target_date, horizon), cols are models
    wide = aligned_df.pivot_table(
        index=["sector_code", "target_date", "horizon"],
        columns="model_name", values="y_pred", aggfunc="first",
    ).reset_index()
    actuals = aligned_df.drop_duplicates(
        subset=["sector_code", "target_date", "horizon"]
    )[["sector_code", "target_date", "horizon", "y_true"]]
    wide = wide.merge(actuals, on=["sector_code", "target_date", "horizon"])

    p_matrix = pd.DataFrame(np.eye(n_models), index=models, columns=models)
    wlt_matrix = pd.DataFrame("", index=models, columns=models)

    for i, m_a in enumerate(models):
        for j, m_b in enumerate(models):
            if i == j:
                wlt_matrix.loc[m_a, m_b] = "—"
                continue
            valid = wide.dropna(subset=[m_a, m_b, "y_true"])
            if len(valid) < 8:
                p_matrix.loc[m_a, m_b] = np.nan
                wlt_matrix.loc[m_a, m_b] = "n/a"
                continue
            err_a = valid[m_a].values - valid["y_true"].values
            err_b = valid[m_b].values - valid["y_true"].values
            r = diebold_mariano_test(err_a, err_b, horizon=horizon, loss=loss)
            p_matrix.loc[m_a, m_b] = r["p_value"]

            # Per-sector DM aggregation for win/loss/tie matrix
            wins = losses = ties = 0
            for sec, sub in valid.groupby("sector_code"):
                if len(sub) < 8:
                    continue
                e_a = sub[m_a].values - sub["y_true"].values
                e_b = sub[m_b].values - sub["y_true"].values
                rs = diebold_mariano_test(e_a, e_b, horizon=horizon, loss=loss)
                if np.isnan(rs["p_value"]):
                    continue
                if rs["p_value"] < alpha:
                    if rs["loss_diff_mean"] < 0:
                        wins += 1     # A's loss < B's loss → A better
                    else:
                        losses += 1
                else:
                    ties += 1
            wlt_matrix.loc[m_a, m_b] = f"{wins}-{losses}-{ties}"

    return p_matrix, wlt_matrix


def friedman_nemenyi(aligned_df: pd.DataFrame) -> dict:
    """
    Non-parametric ranking test across all sectors.

    Returns Friedman test p-value (is there any difference between models?)
    and the Nemenyi critical distance (which pairs are significantly different?).
    """
    from scipy.stats import friedmanchisquare, rankdata

    # One MAE per (sector, model) on the aligned data
    pivot = (compute_per_sector_metrics(aligned_df)
             .pivot(index="sector_code", columns="model_name", values="MAE"))
    pivot = pivot.dropna()  # only sectors where ALL models have an MAE
    if pivot.shape[0] < 3 or pivot.shape[1] < 3:
        return {"friedman_p": np.nan, "n_sectors": int(pivot.shape[0]),
                "note": "need >=3 sectors and >=3 models for Friedman"}

    stat, p = friedmanchisquare(*[pivot[m].values for m in pivot.columns])

    # Average ranks: lower MAE = better → rank 1 for best, etc.
    ranks = pivot.apply(lambda r: rankdata(r.values), axis=1, result_type="expand")
    ranks.columns = pivot.columns
    mean_ranks = ranks.mean().sort_values()

    # Nemenyi critical distance at alpha = 0.05 — q_alpha looked up from table
    # for the studentized range distribution at infinity df.  Hand-coded a few
    # common values:
    n_models = pivot.shape[1]
    n_sectors = pivot.shape[0]
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}.get(n_models, 2.728)
    cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_sectors))

    return {
        "friedman_p":   float(p),
        "friedman_stat": float(stat),
        "n_sectors":    int(n_sectors),
        "n_models":     int(n_models),
        "mean_ranks":   mean_ranks.to_dict(),
        "nemenyi_cd":   float(cd),
        "note":         (f"Models with rank difference > {cd:.3f} are "
                         f"significantly different at α=0.05 (Nemenyi)"),
    }


# ============================================================================
# SCORECARD — multi-criteria decision support
# ============================================================================

@dataclass
class Scorecard:
    """The multi-criteria comparison object the report should reference."""
    aligned_metrics:      pd.DataFrame   # MAE/RMSE/R²/bias per model
    per_horizon_metrics:  pd.DataFrame   # per (model, horizon)
    per_sector_metrics:   pd.DataFrame   # per (model, sector)
    regime_metrics:       pd.DataFrame   # pre/post 2023
    skill_scores:         Optional[pd.DataFrame]  # vs baseline (if provided)
    pi_calibration:       pd.DataFrame   # nominal vs observed coverage
    crps_approx:          pd.DataFrame   # for probabilistic models
    dm_p_matrix:          pd.DataFrame
    dm_win_loss_tie:      pd.DataFrame
    friedman:             dict
    alignment_report:     AlignmentReport
    operational_table:    pd.DataFrame   # explainability + data dependency

    def to_dict_of_dfs(self) -> dict[str, pd.DataFrame]:
        return {
            "aligned_metrics":     self.aligned_metrics,
            "per_horizon_metrics": self.per_horizon_metrics,
            "per_sector_metrics":  self.per_sector_metrics,
            "regime_metrics":      self.regime_metrics,
            "skill_scores":        self.skill_scores if self.skill_scores is not None else pd.DataFrame(),
            "pi_calibration":      self.pi_calibration,
            "crps_approx":         self.crps_approx,
            "dm_p_matrix":         self.dm_p_matrix,
            "dm_win_loss_tie":     self.dm_win_loss_tie,
            "operational_table":   self.operational_table,
        }

    def save(self, output_dir: str | Path):
        """Save every table as parquet for downstream report generation."""
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        for name, frame in self.to_dict_of_dfs().items():
            if not frame.empty:
                frame.to_parquet(outdir / f"{name}.parquet")
        with open(outdir / "friedman.json", "w") as f:
            json.dump(self.friedman, f, indent=2, default=float)
        with open(outdir / "alignment_report.txt", "w") as f:
            f.write(str(self.alignment_report))


# Operational characteristics that don't come from the data — these are
# qualitative scores curated for the report's decision matrix.  Edit if
# your operational reality differs.
_OPERATIONAL_DEFAULTS = {
    "Pipeline_*": {
        "data_dependency":       "high (CBS feature pipeline, monthly+quarterly+yearly tables)",
        "publication_lag":       "exposed to feature publication lags",
        "retraining_cadence":    "quarterly",
        "feature_attribution":   3,  # coefficients / SHAP
        "decomposition_score":   1,  # monolithic regression
        "auditability":          3,  # deterministic + traceable
        "domain_explanation":    3,  # named coefficients
    },
    "AutoETS_*": {
        "data_dependency":       "low (target only)",
        "publication_lag":       "robust — only target lag matters",
        "retraining_cadence":    "quarterly (cheap)",
        "feature_attribution":   2,  # global components
        "decomposition_score":   3,  # full ETS decomposition
        "auditability":          3,
        "domain_explanation":    3,  # trend/seasonal/level in domain terms
    },
    "STL_ETS": {
        "data_dependency":       "low (target only)",
        "publication_lag":       "robust",
        "retraining_cadence":    "quarterly",
        "feature_attribution":   2,
        "decomposition_score":   3,
        "auditability":          3,
        "domain_explanation":    3,
    },
    "Chronos_Bolt": {
        "data_dependency":       "low (target only, pre-trained model)",
        "publication_lag":       "robust",
        "retraining_cadence":    "never (frozen foundation model)",
        "feature_attribution":   1,  # black box
        "decomposition_score":   1,
        "auditability":          2,  # deterministic but opaque
        "domain_explanation":    1,
    },
    "ARIMA": {
        "data_dependency":       "low (target only) or moderate (SARIMAX with exog)",
        "publication_lag":       "robust if univariate",
        "retraining_cadence":    "quarterly",
        "feature_attribution":   2,
        "decomposition_score":   2,
        "auditability":          3,
        "domain_explanation":    3,
    },
}


def _build_operational_table(model_names: Iterable[str]) -> pd.DataFrame:
    """Map model names (which include the variant suffix) to operational scores."""
    rows = []
    for name in sorted(set(model_names)):
        # Try exact match first, then prefix-wildcard match
        meta = _OPERATIONAL_DEFAULTS.get(name)
        if meta is None:
            for key, val in _OPERATIONAL_DEFAULTS.items():
                if key.endswith("*") and name.startswith(key[:-1]):
                    meta = val
                    break
        if meta is None:
            meta = {
                "data_dependency":     "unknown",
                "publication_lag":     "unknown",
                "retraining_cadence":  "unknown",
                "feature_attribution": np.nan,
                "decomposition_score": np.nan,
                "auditability":        np.nan,
                "domain_explanation":  np.nan,
            }
        row = {"model_name": name, **meta}
        # Composite explainability score (sum of 4 axes, max = 12)
        comp = [row[k] for k in ("feature_attribution", "decomposition_score",
                                  "auditability", "domain_explanation")
                if isinstance(row[k], (int, float)) and not np.isnan(row[k])]
        row["explainability_score"] = float(np.sum(comp)) if comp else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_decision_matrix(scorecard: Scorecard) -> pd.DataFrame:
    """
    Convert the scorecard into a recommendation matrix the business owner reads.

    For each business priority, names the recommended model based on which
    one dominates on that priority's primary metric.
    """
    am = scorecard.aligned_metrics.set_index("model_name")
    op = scorecard.operational_table.set_index("model_name")

    rows = []

    # Lowest MAE on the aligned comparison
    if not am.empty:
        rows.append({
            "business_priority":      "Lowest forecast error",
            "primary_metric":         "MAE (aligned)",
            "recommended_model":      am["MAE"].idxmin(),
            "rationale":              f"MAE = {am['MAE'].min():.4f}",
        })

    # Best regime robustness
    rm = scorecard.regime_metrics.pivot_table(
        index="model_name", columns="regime", values="R2", aggfunc="first")
    if not rm.empty and "post_2023" in rm.columns:
        rows.append({
            "business_priority":      "Robust under regime shift",
            "primary_metric":         "R² post-2023",
            "recommended_model":      rm["post_2023"].idxmax(),
            "rationale":              f"R² post-2023 = {rm['post_2023'].max():.3f}",
        })

    # Best calibration
    if not scorecard.pi_calibration.empty:
        # Lowest absolute miscalibration averaged across levels
        miscal = (scorecard.pi_calibration.assign(
                    abs_mis=lambda d: d["miscalibration"].abs())
                  .groupby("model_name")["abs_mis"].mean())
        rows.append({
            "business_priority":      "Honest uncertainty bands",
            "primary_metric":         "|observed − nominal coverage|",
            "recommended_model":      miscal.idxmin(),
            "rationale":              f"mean |miscalibration| = {miscal.min():.3f}",
        })

    # Highest explainability
    if "explainability_score" in op.columns:
        rows.append({
            "business_priority":      "Explainable to stakeholders",
            "primary_metric":         "Explainability score (out of 12)",
            "recommended_model":      op["explainability_score"].idxmax(),
            "rationale":              f"score = {op['explainability_score'].max():.0f}/12",
        })

    # Lowest infrastructure cost
    if "data_dependency" in op.columns:
        low_dep = op[op["data_dependency"].astype(str).str.startswith("low")]
        if not low_dep.empty:
            # Among low-dependency models, pick the one with best MAE
            common = low_dep.index.intersection(am.index)
            if len(common) > 0:
                best = am.loc[common, "MAE"].idxmin()
                rows.append({
                    "business_priority":      "Lowest infrastructure cost",
                    "primary_metric":         "MAE among target-only models",
                    "recommended_model":      best,
                    "rationale":              f"MAE = {am.loc[best, 'MAE']:.4f} with no exogenous data",
                })

    return pd.DataFrame(rows)


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def compare_all_models(
    model_dfs: Sequence[pd.DataFrame],
    baseline_df: Optional[pd.DataFrame] = None,
    covid_end: pd.Timestamp = pd.Timestamp("2022-12-31"),
    regime_split_date: pd.Timestamp = pd.Timestamp("2023-01-01"),
    horizons_to_compare: Optional[list[int]] = None,
    output_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> Scorecard:
    """
    Run the full comparison pipeline.

    Steps:
      1. Validate every model is in canonical schema.
      2. Optionally filter to a specific set of horizons (default: keep all
         shared horizons).
      3. Align on the intersection of (sector, target_date, horizon).
      4. Compute point metrics, regime metrics, PI calibration, CRPS.
      5. Run DM pairwise tests and Friedman/Nemenyi.
      6. Build the operational and decision matrices.
      7. Save everything if output_dir provided.

    Parameters
    ----------
    model_dfs : sequence of canonical-schema DataFrames
    baseline_df : optional baseline predictions (canonical schema) for skill scores
    covid_end / regime_split_date : split dates for regime analysis
    horizons_to_compare : if set, keep only these horizons in alignment.
                          Default keeps all horizons present in every model.
    output_dir : if set, save tables to disk
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  UNIFIED MODEL COMPARISON")
        print(f"{'='*70}\n")

    # 1. Validate
    validated = [_validate_canonical(df, df["model_name"].iloc[0]) for df in model_dfs]

    if verbose:
        print(f"  Models supplied:")
        for df in validated:
            name = df["model_name"].iloc[0]
            print(f"    {name:30s} | {len(df):>5} rows | "
                  f"{df['sector_code'].nunique():>3} sectors | "
                  f"horizons={sorted(df['horizon'].unique().tolist())}")

    # 2. Filter horizons if requested
    if horizons_to_compare:
        validated = [df[df["horizon"].isin(horizons_to_compare)].copy()
                     for df in validated]

    # 3. Align
    aligned, alignment_report = align_predictions(validated)
    if verbose:
        print(f"\n{alignment_report}\n")

    # 4. Point metrics
    aligned_metrics     = compute_point_metrics(aligned)
    per_horizon_metrics = compute_per_horizon_metrics(aligned)
    per_sector_metrics  = compute_per_sector_metrics(aligned)
    regime_metrics      = compute_regime_metrics(aligned, regime_split_date)

    # 5. Skill scores (optional)
    skill_scores = None
    if baseline_df is not None:
        baseline_aligned = baseline_df[
            baseline_df.set_index(["sector_code", "target_date", "horizon"]).index
            .isin(aligned.set_index(["sector_code", "target_date", "horizon"]).index)
        ]
        skill_scores = compute_skill_score(aligned, baseline_aligned)

    # 6. Probabilistic metrics
    pi_calibration = compute_pi_coverage(aligned)
    crps_approx    = compute_crps_quantile_approx(aligned)

    # 7. Statistical tests
    if verbose:
        print(f"  Running pairwise DM tests...")
    dm_p_matrix, dm_wlt = pairwise_dm_matrix(aligned)
    if verbose:
        print(f"  Running Friedman + Nemenyi...")
    friedman = friedman_nemenyi(aligned)

    # 8. Operational + decision
    op_table = _build_operational_table(aligned["model_name"].unique())

    scorecard = Scorecard(
        aligned_metrics=aligned_metrics,
        per_horizon_metrics=per_horizon_metrics,
        per_sector_metrics=per_sector_metrics,
        regime_metrics=regime_metrics,
        skill_scores=skill_scores,
        pi_calibration=pi_calibration,
        crps_approx=crps_approx,
        dm_p_matrix=dm_p_matrix,
        dm_win_loss_tie=dm_wlt,
        friedman=friedman,
        alignment_report=alignment_report,
        operational_table=op_table,
    )

    if verbose:
        print(f"\n  Aligned point metrics:")
        print(scorecard.aligned_metrics.to_string(index=False,
              float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
        if friedman.get("friedman_p") is not None:
            print(f"\n  Friedman p-value: {friedman['friedman_p']:.4f}")
            print(f"  Nemenyi CD: {friedman['nemenyi_cd']:.3f}")
            print(f"  Mean ranks: {friedman['mean_ranks']}")

    if output_dir:
        scorecard.save(output_dir)
        decision = make_decision_matrix(scorecard)
        decision.to_parquet(Path(output_dir) / "decision_matrix.parquet")
        if verbose:
            print(f"\n  Saved scorecard tables to {output_dir}")
            print(f"\n  Decision matrix:")
            print(decision.to_string(index=False))

    return scorecard
