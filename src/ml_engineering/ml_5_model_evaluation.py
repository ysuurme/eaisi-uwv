"""
Step 5 — ML Model Evaluation (with nested-CV fold labelling).

Computes regression metrics using walk-forward (rolling-origin) evaluation
across n_test_points quarters, split into 4-quarter forecast windows.

Walk-forward methodology
------------------------
For each of the (n_test_points // 4) rolling origins:
    1. Clone the fitted estimator (produces an unfitted copy)
    2. Refit on the expanding training window up to that origin
    3. Forecast 4 quarters ahead (matching the project's 4Q objective)
    4. Collect predictions and actuals

Production-honest exogenous features (no leakage)
-------------------------------------------------
By default (``x_future_mode="production"``) the X rows for the forecast
window are CONSTRUCTED via ``build_future_x`` (defined below): deterministic
structural columns are extended from the dates, stochastic exogenous columns
are carried forward from the last observed value.  This matches exactly what
is available when producing a real forward forecast (Step 7) — models are
never evaluated with future covariate values they would not have in
production.  ``x_future_mode="actual"`` restores the old behaviour (actual
future X rows) for diagnostic comparisons only.

Inner / outer fold split (added for honest variant selection)
-------------------------------------------------------------
The walk-forward origins are split chronologically into:

* INNER folds — earliest ``ceil(n_origins * _INNER_FRACTION)`` origins.
  Used by the downstream ``m_pipeline_loader`` to pick which variant
  (model family / preset combination) represents Pipeline for each sector.

* OUTER folds — remaining (latest) origins.  Used as the honest evaluation
  of the chosen variant.  These predictions are NEVER inspected during
  variant selection.

This yields proper nested cross-validation:

    Layer 1 (ml_4):       ExpandingWindowSplitter on TRAINING set
                          → chooses hyperparameters within a variant
    Layer 2 (ml_5):       Walk-forward CV across test quarters
                          → first 40% of origins = inner (variant selection)
                          → remaining 60% = outer (honest evaluation)
    Layer 3 (loader):     Pick best variant per sector using INNER MAE,
                          report OUTER predictions as canonical

The headline aggregate metrics returned by this module (and stored in
``model_evaluations``) are computed on OUTER folds only — the honest
out-of-sample estimate.  Inner-fold MAE is logged as a separate
``mae_inner_diagnostic`` metric on the MLflow run for transparency.

Per-row prediction logging
--------------------------
Every individual (origin, horizon) prediction is stored in
``model_predictions`` with sector_code, origin_date, target_date, horizon,
y_true, y_pred, and the ``fold_set`` column ("inner" or "outer").

Who consumes the inner/outer split
----------------------------------
The split is NOT vestigial.  ``run_pipeline`` deliberately does not call the
variant-selection loader (a single-estimator run has nothing to select between —
cross-family selection is handled later by the Step-6 champion/challenger gate),
but the ``fold_set`` label is consumed by:

* this module — the headline aggregate metrics use OUTER folds only;
* ``m_sector_quality.per_horizon_mape`` + ``m_model_viz.plot_predicted_vs_actual``
  — the report filters to OUTER folds for the honest view;
* ``m_pipeline_loader`` (a standalone cross-method-comparison CLI) — uses INNER
  folds to PICK a variant per sector and reports OUTER as canonical.

``ModelPredictionRecord`` already declares the ``fold_set`` column; the eval DB
is rebuilt from the ORM by ``_ensure_eval_db`` (no manual migration needed).
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sktime.forecasting.base import ForecastingHorizon
from sqlalchemy.orm import Session

from src.ml_engineering.model_configs import (
    ModelEvaluationRecord,
    ModelPredictionRecord,
    SectorQuarterRollingMean,
)
from src.utils.m_log import f_log


_FH_STEPS = 4          # quarters per forecast window — matches 4Q project objective
_STEP_LENGTH = 4       # rolling-origin step size (1 year, quarterly aligned)
_INNER_FRACTION = 0.4  # First 40% of origins are inner folds (variant selection);
                       # remainder are outer folds (honest evaluation).
                       # With n_test_points=20 → 5 origins → 2 inner + 3 outer.


# ---------------------------------------------------------------------------
# Production-honest future feature rows (shared by Step 5 eval + Step 7 inference)
# ---------------------------------------------------------------------------
# Given the feature matrix observed up to a forecast origin, construct the X
# rows for the next ``n_steps`` quarters WITHOUT using any future information —
# exactly what is available when producing a real forecast.  Deterministic
# structural columns are recomputed from the future quarter-end dates
# (definitions mirror ``data_loader_gold``); every other (stochastic exogenous)
# column is carried forward from the last observed row.  Evaluating models under
# the same rule (ml_5) removes the optimistic bias of feeding actual future
# covariates during backtests, and Step 7 forecasts under identical conditions.

#: Day count of one quarter (365.25 / 4) — mirrors data_loader_gold.
QUARTER_DAYS = 91.3125

#: Regime boundaries (quarter end-dates) — mirror data_loader_gold.
PRE_COVID_END = pd.Timestamp("2019-12-31")  # last pre-COVID quarter end
COVID_START   = pd.Timestamp("2020-03-31")  # end of Q1 2020
COVID_END     = pd.Timestamp("2022-12-31")  # end of Q4 2022


def build_future_x(x_hist: pd.DataFrame, n_steps: int = 4) -> pd.DataFrame:
    """Extend a feature matrix ``n_steps`` quarters beyond its last observation.

    Deterministic structural columns (``year``/``quarter``/``trend_index``/
    ``covid_period``/``post_covid``/``covid_depth``/``recovery_quarters`` and the
    ``*_x_post_covid`` interactions) are recomputed from the future quarter-end
    dates; all other columns are carried forward from the last observed row (the
    honest production stance — future CBS releases are unknown at forecast time).

    Args:
        x_hist: Feature matrix observed up to the forecast origin.  Must have a
            DatetimeIndex of quarter-end dates (as produced by Step 3) and at
            least one row.
        n_steps: Number of future quarters to construct (default 4 = the
            project's forecast horizon).

    Returns:
        DataFrame with ``n_steps`` rows, indexed by the next ``n_steps``
        quarter-end dates, same columns as ``x_hist`` — deterministic structural
        columns recomputed, all other columns carried forward.
    """
    if x_hist.empty:
        raise ValueError("x_hist is empty — cannot extend into the future.")

    last_date = pd.Timestamp(x_hist.index[-1])
    future_idx = pd.DatetimeIndex(
        pd.date_range(last_date, periods=n_steps + 1, freq="QE")[1:],
        freq=None,
        name=x_hist.index.name,
    )

    # Carry-forward base: repeat the last observed row (production stance for
    # stochastic exogenous features — future CBS values are unknown).
    future_x = pd.DataFrame(
        np.repeat(x_hist.iloc[[-1]].to_numpy(), n_steps, axis=0),
        columns=x_hist.columns,
        index=future_idx,
    )

    # Deterministic structural columns — recomputed from the future dates.
    deterministic = {
        "year":              future_idx.year.to_numpy(dtype=float),
        "quarter":           future_idx.quarter.to_numpy(dtype=float),
        "covid_period":      ((future_idx >= COVID_START) & (future_idx <= COVID_END)).astype(float),
        "post_covid":        (future_idx > COVID_END).astype(float),
        "covid_depth":       np.clip((future_idx - PRE_COVID_END).days / QUARTER_DAYS, 0.0, 12.0),
        "recovery_quarters": np.clip((future_idx - COVID_END).days / QUARTER_DAYS, 0.0, None),
    }
    if "trend_index" in future_x.columns:
        deterministic["trend_index"] = (
            float(x_hist["trend_index"].iloc[-1])
            + np.arange(1, n_steps + 1, dtype=float)
        )
    for col, values in deterministic.items():
        if col in future_x.columns:
            future_x[col] = values

    # Interactions — recomputed from their (extended) parents.
    if {"trend_x_post_covid", "trend_index", "post_covid"} <= set(future_x.columns):
        future_x["trend_x_post_covid"] = future_x["trend_index"] * future_x["post_covid"]
    if {"quarter_x_post_covid", "quarter", "post_covid"} <= set(future_x.columns):
        future_x["quarter_x_post_covid"] = future_x["quarter"] * future_x["post_covid"]

    return future_x


class ModelEvaluator:
    """Evaluates a trained model via walk-forward refit with nested-CV labelling."""

    def __init__(self, session: Session):
        self.session = session

    def evaluate(
        self,
        run_id: str,
        fitted_model: Any,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        n_test_points: int = 20,
        sector_code: Optional[str] = None,
        x_future_mode: str = "production",
    ) -> Dict[str, float]:
        """Walk-forward evaluation across n_test_points quarters.

        Returns aggregate r2/mae/rmse computed on **outer folds only** (the
        honest out-of-sample estimate).  Inner-fold MAE is exposed as
        ``mae_inner_diagnostic`` on the MLflow run but is not part of the
        returned dict.

        Args:
            run_id: MLflow run ID to log metrics against.
            fitted_model: Model fitted by Step 4 on y_train; used as the clone
                template (re-fitted at every origin).
            x_train, y_train: Training window.
            x_test, y_test: Evaluation window (n_test_points rows).
            model_name: Identifier for the evaluation DB record.
            n_test_points: Number of test quarters; must be divisible by _FH_STEPS.
            sector_code: Sector identifier for per-row prediction logging.
                If None, parsed from the trailing token of ``model_name``.
            x_future_mode: "production" (default) constructs forecast-window X
                via ``build_future_x`` (no future covariate leakage);
                "actual" feeds the actual future X rows (diagnostic only).

        Returns:
            {'r2': ..., 'mae': ..., 'rmse': ...} computed on outer folds.
        """
        effective_sector = sector_code or _parse_sector_from_model_name(model_name)

        metrics, pred_records, diagnostics = _walk_forward_metrics(
            fitted_model, x_train, y_train, x_test, y_test, n_test_points, run_id,
            x_future_mode=x_future_mode,
        )
        self._persist_record(
            run_id, model_name, effective_sector, metrics, fitted_model, pred_records,
        )

        f_log(
            f"Evaluation ({diagnostics['n_inner']} inner + {diagnostics['n_outer']} outer origins, "
            f"{_FH_STEPS}Q each) | "
            f"OUTER: MASE={metrics['mase']:.4f} MAPE={metrics['mape']:.4f} "
            f"R²={metrics['r2']:.4f} MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} | "
            f"per-row predictions saved: {len(pred_records)}",
            c_type="success",
        )
        return metrics

    def _persist_record(
        self,
        run_id: str,
        model_name: str,
        sector_code: str,
        metrics: Dict[str, float],
        fitted_model: Any,
        pred_records: List[Dict[str, Any]],
    ) -> None:
        # --- Aggregate evaluation record (outer-fold honest metrics) ---
        # Model artifacts live in MLflow (the single model store) — the eval
        # DB holds metrics/analytics only.  No pickled blob is persisted here.
        record = ModelEvaluationRecord(
            run_id=run_id,
            model_name=model_name,
            mase=metrics["mase"],
            r2=metrics["r2"],
            mae=metrics["mae"],
            mape=metrics["mape"],
            rmse=metrics["rmse"],
            passed_gate=0,  # Step 6 updates this after the quality gate check
        )
        self.session.merge(record)

        # --- Per-row predictions with fold_set column ---
        # Delete-then-insert protects against duplicates on accidental re-runs
        # of the same run_id.
        if pred_records:
            (self.session.query(ModelPredictionRecord)
                         .filter(ModelPredictionRecord.run_id == run_id)
                         .delete(synchronize_session=False))
            self.session.bulk_insert_mappings(
                ModelPredictionRecord,
                [
                    {
                        "run_id":      run_id,
                        "model_name":  model_name,
                        "sector_code": sector_code,
                        **rec,
                    }
                    for rec in pred_records
                ],
            )

        f_log("Evaluation record + per-row predictions (with fold_set) stored",
              c_type="store")


# ---------------------------------------------------------------------------
# Walk-forward helpers — module-level so they are independently testable
# ---------------------------------------------------------------------------

def _split_origins_inner_outer(n_origins: int) -> Tuple[int, int]:
    """Return (n_inner, n_outer) where n_inner is the count of inner-fold origins.

    Uses ``_INNER_FRACTION`` and ceil(), guaranteeing:
    * For n_origins == 0: returns (0, 0)
    * For n_origins == 1: returns (0, 1) — single origin treated as outer
    * For n_origins >= 2: at least 1 inner and 1 outer fold

    With the default _INNER_FRACTION=0.4:
        n=2 → (1, 1)     n=5 → (2, 3)
        n=3 → (2, 1)     n=6 → (3, 3)
        n=4 → (2, 2)     n=10 → (4, 6)
    """
    if n_origins <= 1:
        return 0, max(0, n_origins)
    n_inner = max(1, int(math.ceil(n_origins * _INNER_FRACTION)))
    n_inner = min(n_inner, n_origins - 1)   # always leave ≥ 1 outer origin
    n_outer = n_origins - n_inner
    return n_inner, n_outer


def _walk_forward_metrics(
    fitted_model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n_test_points: int,
    run_id: str,
    x_future_mode: str = "production",
) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, float]]:
    """Walk-forward evaluation with inner/outer fold labelling.

    Returns
    -------
    metrics : dict
        Aggregate r2 / mae / rmse computed on OUTER FOLDS ONLY (the honest
        out-of-sample estimate).  If no outer folds were produced (e.g. CV
        truncated by insufficient data), falls back to inner-fold metrics
        with a warning.
    pred_records : list of dict
        Per-row prediction records including ``fold_set`` ("inner" / "outer").
    diagnostics : dict
        Inner-fold MAE and origin counts for transparency.  Logged to MLflow
        but not used as headline metrics.
    """
    y_full = pd.concat([y_train, y_test])
    X_full = pd.concat([x_train, x_test])

    initial_window = len(y_train)
    n_origins = n_test_points // _FH_STEPS
    n_inner_target, n_outer_target = _split_origins_inner_outer(n_origins)

    pred_records: list = []
    inner_y_true: list = []
    inner_y_pred: list = []
    outer_y_true: list = []
    outer_y_pred: list = []
    n_inner_actual = 0
    n_outer_actual = 0

    for i in range(n_origins):
        origin = initial_window + i * _STEP_LENGTH
        if origin + _FH_STEPS > len(y_full):
            f_log(
                f"Walk-forward: stopping early at origin {i + 1} — insufficient data "
                f"(needed {origin + _FH_STEPS}, have {len(y_full)}).",
                c_type="warning",
            )
            break

        is_inner = i < n_inner_target
        fold_set = "inner" if is_inner else "outer"

        y_tr  = y_full.iloc[:origin]
        y_fut = y_full.iloc[origin:origin + _FH_STEPS]
        X_tr  = X_full.iloc[:origin]
        if x_future_mode == "production":
            # No leakage: forecast-window X is constructed from information
            # available at the origin (deterministic structure extended,
            # exogenous values carried forward) — same rule as Step 7.
            X_fut = build_future_x(X_tr, n_steps=_FH_STEPS)
        else:
            X_fut = X_full.iloc[origin:origin + _FH_STEPS]

        estimator = clone(fitted_model)
        y_pred = np.asarray(_predict_origin(estimator, y_tr, X_tr, X_fut)).ravel()

        # Per-row prediction records (used by the cross-method comparison)
        origin_ts = _to_timestamp(y_tr.index[-1])
        for h_idx in range(min(_FH_STEPS, len(y_fut), len(y_pred))):
            pred_records.append({
                "origin_date": origin_ts,
                "target_date": _to_timestamp(y_fut.index[h_idx]),
                "horizon":     int(h_idx + 1),
                "y_true":      float(y_fut.iloc[h_idx]),
                "y_pred":      float(y_pred[h_idx]),
                "fold_set":    fold_set,
            })

        if is_inner:
            n_inner_actual += 1
            inner_y_true.extend(y_fut.values.tolist())
            inner_y_pred.extend(y_pred.tolist())
        else:
            n_outer_actual += 1
            outer_y_true.extend(y_fut.values.tolist())
            outer_y_pred.extend(y_pred.tolist())

    # --- Compute aggregate metrics on OUTER folds only (honest estimate) ---
    if outer_y_true:
        yt = np.array(outer_y_true)
        yp = np.array(outer_y_pred)
        r2   = float(r2_score(yt, yp))
        mae  = float(mean_absolute_error(yt, yp))
        mape = float(mean_absolute_percentage_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    elif inner_y_true:
        # CV truncated before reaching outer folds — fall back with a warning
        f_log(
            "Walk-forward: no outer folds produced (CV truncated). Headline "
            "metrics fall back to inner folds; treat as upper bound.",
            c_type="warning",
        )
        yt = np.array(inner_y_true)
        yp = np.array(inner_y_pred)
        r2   = float(r2_score(yt, yp))
        mae  = float(mean_absolute_error(yt, yp))
        mape = float(mean_absolute_percentage_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    else:
        r2 = mae = mape = rmse = float("nan")
        f_log("Walk-forward produced no predictions at all.", c_type="error")

    # --- Inner-fold diagnostic MAE ---
    mae_inner_diag = (
        float(mean_absolute_error(inner_y_true, inner_y_pred))
        if inner_y_true else float("nan")
    )

    # --- MASE: THE comparison metric (scale-free, baseline-aware) ---
    # MASE = MAE / in-sample MAE of the seasonal-naive forecast (m=4 quarters).
    # <1 beats the seasonal naive; lower is better.  The scaler is computed on
    # the training window (y_train) only — never the forecast/test rows.
    scaler = _seasonal_naive_mae(y_train, sp=_FH_STEPS)
    mase = (
        float(mae / scaler)
        if (math.isfinite(mae) and math.isfinite(scaler) and scaler > 0)
        else float("nan")
    )
    mase_inner_diag = (
        float(mae_inner_diag / scaler)
        if (math.isfinite(mae_inner_diag) and math.isfinite(scaler) and scaler > 0)
        else float("nan")
    )

    # --- Log to MLflow ---
    with mlflow.start_run(run_id=run_id):
        # Evaluation-honesty lineage: how forecast-window X was supplied.
        mlflow.log_param("x_future_mode", x_future_mode)
        mlflow.log_metrics({
            # Headline = OUTER FOLDS (honest).  MASE is THE champion-gate metric
            # (scale-free, comparable across sectors); MAPE/R²/MAE/RMSE are kept
            # informative.
            "mean_absolute_scaled_error":     mase,
            "mean_absolute_percentage_error": mape,
            "r2_score":                r2,
            "mean_absolute_error":     mae,
            "root_mean_squared_error": rmse,
            # Diagnostic / provenance
            "mase_inner_diagnostic":   mase_inner_diag,
            "mae_inner_diagnostic":    mae_inner_diag,
            "seasonal_naive_mae":      scaler,
            "n_inner_origins":         float(n_inner_actual),
            "n_outer_origins":         float(n_outer_actual),
            "n_inner_predictions":     float(len(inner_y_true)),
            "n_outer_predictions":     float(len(outer_y_true)),
        })
        # Consolidate the evaluation data INTO MLflow as table artifacts (the run's
        # "Evaluation" tab + cross-run table comparison), so the actual-vs-predicted
        # rows and the metric breakdown live alongside the metrics — not only in the
        # eval DB's model_predictions.
        preds_tbl, metrics_tbl = _build_eval_tables(
            pred_records, mase, mape, r2, mae, rmse, mae_inner_diag,
        )
        if not preds_tbl.empty:
            mlflow.log_table(preds_tbl, artifact_file="eval/walk_forward_predictions.json")
        mlflow.log_table(metrics_tbl, artifact_file="eval/metrics_summary.json")

    diagnostics = {
        "mae_inner":  mae_inner_diag,
        "n_inner":    n_inner_actual,
        "n_outer":    n_outer_actual,
    }
    return {"mase": mase, "r2": r2, "mae": mae, "mape": mape, "rmse": rmse}, pred_records, diagnostics


def _seasonal_naive_mae(y_train, sp: int = 4) -> float:
    """In-sample MAE of the seasonal-naive forecast — the MASE scaler.

    ``mean |y_t - y_{t-sp}|`` over the TRAINING window (sp=4 = same quarter last
    year for quarterly data).  Returns NaN if the window is too short.
    """
    y = np.asarray(y_train, dtype=float)
    if len(y) <= sp:
        return float("nan")
    return float(np.mean(np.abs(y[sp:] - y[:-sp])))


def _build_eval_tables(
    pred_records: List[Dict[str, Any]],
    mase: float, mape: float, r2: float, mae: float, rmse: float, mae_inner: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the two MLflow eval table artifacts from walk-forward output.

    Returns ``(predictions_df, metrics_df)``:

    * ``predictions_df`` — one row per (origin, horizon, fold_set) carrying
      ``y_true`` / ``y_pred`` plus derived ``abs_error`` / ``abs_pct_error`` (the
      rich actual-vs-predicted view the Evaluation tab renders).
    * ``metrics_df`` — tidy ``scope / metric / value`` rows: the honest outer-fold
      headline (MASE — THE metric — + MAPE/R²/MAE/RMSE), the inner-fold MAE
      diagnostic, and per-horizon outer MAPE — the cross-run comparison.

    Pure (no MLflow/IO) so it is unit-testable; the caller logs both via
    ``mlflow.log_table``.
    """
    preds = pd.DataFrame(pred_records)
    if not preds.empty:
        preds = preds.copy()
        preds["origin_date"] = preds["origin_date"].astype(str)
        preds["target_date"] = preds["target_date"].astype(str)
        preds["abs_error"] = (preds["y_true"] - preds["y_pred"]).abs()
        denom = preds["y_true"].abs().replace(0.0, np.nan)
        preds["abs_pct_error"] = preds["abs_error"] / denom

    rows: List[Dict[str, Any]] = [
        {"scope": "outer", "metric": "MASE", "value": mase},
        {"scope": "outer", "metric": "MAPE", "value": mape},
        {"scope": "outer", "metric": "R2",   "value": r2},
        {"scope": "outer", "metric": "MAE",  "value": mae},
        {"scope": "outer", "metric": "RMSE", "value": rmse},
        {"scope": "inner", "metric": "MAE",  "value": mae_inner},
    ]
    if not preds.empty and "fold_set" in preds.columns:
        outer = preds[preds["fold_set"] == "outer"]
        for h, grp in outer.groupby("horizon"):
            valid = grp["abs_pct_error"].dropna()
            if len(valid):
                rows.append({
                    "scope": f"outer_h{int(h)}", "metric": "MAPE",
                    "value": float(valid.mean()),
                })
    return preds, pd.DataFrame(rows)


def _predict_origin(
    estimator: Any,
    y_train: pd.Series,
    X_train: pd.DataFrame,
    X_future: pd.DataFrame,
) -> np.ndarray:
    """Fits a cloned estimator on one expanding window and forecasts _FH_STEPS ahead.

    Dispatches to the sklearn API (SectorQuarterRollingMean) or the sktime
    API (all other forecasters) based on estimator type.
    """
    if isinstance(estimator, SectorQuarterRollingMean):
        estimator.fit(X_train, y_train)
        return estimator.predict(X_future)

    # sktime forecasters: strip non-numeric columns (OHE SBI cols already absent)
    X_tr_num  = X_train.select_dtypes(include="number")
    X_fut_num = X_future.select_dtypes(include="number")
    X_tr  = X_tr_num  if not X_tr_num.empty  else None
    X_fut = X_fut_num if not X_fut_num.empty else None

    fh = ForecastingHorizon(range(1, _FH_STEPS + 1), is_relative=True)
    # fh at fit (not only predict) so strategy="direct" reducers — which require
    # fh-in-fit — evaluate through the same walk-forward path; harmless for the
    # recursive/stat forecasters, which accept and store it.
    estimator.fit(y=y_train, X=X_tr, fh=fh)
    return np.asarray(estimator.predict(fh=fh, X=X_fut))


# ---------------------------------------------------------------------------
# Small utility helpers — module level for testability
# ---------------------------------------------------------------------------

def _to_timestamp(x: Any) -> pd.Timestamp:
    """Convert a Period, Timestamp, or datetime-like to a pandas Timestamp."""
    if hasattr(x, "to_timestamp"):
        return x.to_timestamp()
    return pd.Timestamp(x)


def _parse_sector_from_model_name(model_name: str) -> str:
    """Fallback: parse the trailing token from model_name ("ridge_300003" → "300003")."""
    if "_" in model_name:
        return model_name.rsplit("_", 1)[-1]
    return model_name
