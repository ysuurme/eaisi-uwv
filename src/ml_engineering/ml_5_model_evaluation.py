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
y_true, y_pred, and the new ``fold_set`` column ("inner" or "outer").

REQUIRES
--------
* ``ModelPredictionRecord`` (in model_configs.py) must have a ``fold_set``
  column.  See ``model_configs_fold_set_patch.py`` for the patch.
* If you already have a populated ``model_predictions`` table, run the
  migration (DROP TABLE or ALTER TABLE) before re-running the pipeline.
  The orchestrator's ``_ensure_eval_db`` will fail-fast if the column is
  missing.
"""
import math
import pickle
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

        Returns:
            {'r2': ..., 'mae': ..., 'rmse': ...} computed on outer folds.
        """
        effective_sector = sector_code or _parse_sector_from_model_name(model_name)

        metrics, pred_records, diagnostics = _walk_forward_metrics(
            fitted_model, x_train, y_train, x_test, y_test, n_test_points, run_id
        )
        self._persist_record(
            run_id, model_name, effective_sector, metrics, fitted_model, pred_records,
        )

        f_log(
            f"Evaluation ({diagnostics['n_inner']} inner + {diagnostics['n_outer']} outer origins, "
            f"{_FH_STEPS}Q each) | "
            f"OUTER: R²={metrics['r2']:.4f} MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} | "
            f"INNER MAE (diag)={diagnostics['mae_inner']:.4f} | "
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
        record = ModelEvaluationRecord(
            run_id=run_id,
            model_name=model_name,
            r2=metrics["r2"],
            mae=metrics["mae"],
            rmse=metrics["rmse"],
            passed_gate=0,  # Step 6 updates this after the quality gate check
            model_blob=(
                pickle.dumps(fitted_model)
                if isinstance(fitted_model, SectorQuarterRollingMean)
                else None  # sktime blobs are stored as pyfunc artifacts by Step 4
            ),
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
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    else:
        r2 = mae = rmse = float("nan")
        f_log("Walk-forward produced no predictions at all.", c_type="error")

    # --- Inner-fold diagnostic MAE ---
    mae_inner_diag = (
        float(mean_absolute_error(inner_y_true, inner_y_pred))
        if inner_y_true else float("nan")
    )

    # --- Log to MLflow ---
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            # Headline = OUTER FOLDS (honest)
            "r2_score":                r2,
            "mean_absolute_error":     mae,
            "root_mean_squared_error": rmse,
            # Diagnostic / provenance
            "mae_inner_diagnostic":    mae_inner_diag,
            "n_inner_origins":         float(n_inner_actual),
            "n_outer_origins":         float(n_outer_actual),
            "n_inner_predictions":     float(len(inner_y_true)),
            "n_outer_predictions":     float(len(outer_y_true)),
        })

    diagnostics = {
        "mae_inner":  mae_inner_diag,
        "n_inner":    n_inner_actual,
        "n_outer":    n_outer_actual,
    }
    return {"r2": r2, "mae": mae, "rmse": rmse}, pred_records, diagnostics


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
    estimator.fit(y=y_train, X=X_tr)
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
