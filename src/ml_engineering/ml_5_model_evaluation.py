"""
Step 5 — ML Model Evaluation.

Computes regression metrics using walk-forward (rolling-origin) evaluation
across n_test_points quarters, split into 4-quarter forecast windows.

Walk-forward methodology
------------------------
For each of the (n_test_points // 4) rolling origins:
    1. Clone the fitted estimator (produces an unfitted copy)
    2. Refit on the expanding training window up to that origin
    3. Forecast 4 quarters ahead (matching the project's 4Q objective)
    4. Collect predictions and actuals

Aggregate R², MAE, RMSE are computed across all n_test_points predictions.
This is equivalent to the ExpandingWindowSplitter used in tuning and produces
a reliable multi-origin view of 4Q-ahead forecast accuracy.

A single-origin test on 4 points is statistically unreliable and can be
inflated by lucky alignment or feature overfitting (e.g. Prophet + 400
features).  Walk-forward across 5 × 4-point windows gives 20 evaluation
points and a metric that reflects genuine out-of-sample generalisation.
"""
import pickle
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sktime.forecasting.base import ForecastingHorizon
from sqlalchemy.orm import Session

from src.ml_engineering.model_configs import ModelEvaluationRecord, SectorQuarterRollingMean
from src.utils.m_log import f_log


_FH_STEPS = 4       # quarters per forecast window — matches 4Q project objective
_STEP_LENGTH = 4    # rolling-origin step size (1 year, quarterly aligned)


class ModelEvaluator:
    """Evaluates a trained model via walk-forward refit across rolling origins."""

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
    ) -> Dict[str, float]:
        """Walk-forward evaluation across n_test_points quarters.

        Args:
            run_id: MLflow run ID to log metrics against.
            fitted_model: The model fitted in Step 4 on y_train.  Used as the
                template for cloning; also persisted as the registry artifact.
            x_train: Feature matrix for the training window.
            y_train: Target series for the training window.
            x_test: Feature matrix for the evaluation window (n_test_points rows).
            y_test: Target series for the evaluation window (n_test_points rows).
            model_name: Name used for the evaluation DB record.
            n_test_points: Total number of quarterly test observations.
                Must be divisible by _FH_STEPS (4).  Defaults to 20 (5 origins).

        Returns:
            Dict with keys 'r2', 'mae', 'rmse' — aggregated across all origins.
        """
        metrics = _walk_forward_metrics(
            fitted_model, x_train, y_train, x_test, y_test, n_test_points, run_id
        )
        self._persist_record(run_id, model_name, metrics, fitted_model)
        n_origins = len(y_test) // _FH_STEPS
        f_log(
            f"Evaluation ({n_origins} origins × {_FH_STEPS}Q = {n_origins * _FH_STEPS} pts) | "
            f"R2: {metrics['r2']:.4f} | MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f}",
            c_type="success",
        )
        return metrics

    def _persist_record(
        self,
        run_id: str,
        model_name: str,
        metrics: Dict[str, float],
        fitted_model: Any,
    ) -> None:
        record = ModelEvaluationRecord(
            run_id=run_id,
            model_name=model_name,
            r2=metrics["r2"],
            mae=metrics["mae"],
            rmse=metrics["rmse"],
            passed_gate=0,  # Updated by Step 6 after quality gate validation
            model_blob=(
                pickle.dumps(fitted_model)
                if isinstance(fitted_model, SectorQuarterRollingMean)
                else None   # sktime blobs stored as pyfunc artifact in Step 4
            ),
        )
        self.session.merge(record)
        f_log("Evaluation record stored | eval_data.db", c_type="store")


# ---------------------------------------------------------------------------
# Walk-forward helpers — module-level so they are independently testable
# ---------------------------------------------------------------------------

def _walk_forward_metrics(
    fitted_model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n_test_points: int,
    run_id: str,
) -> Dict[str, float]:
    """Clones the estimator and evaluates over rolling 4Q origins.

    Reconstructs the full series (train + test) then steps through
    n_test_points // _FH_STEPS expanding origins, refitting from scratch at
    each one.  All predictions are concatenated before computing aggregate
    metrics.
    """
    y_full = pd.concat([y_train, y_test])
    X_full = pd.concat([x_train, x_test])

    initial_window = len(y_train)
    n_origins = n_test_points // _FH_STEPS

    all_y_true: list = []
    all_y_pred: list = []

    for i in range(n_origins):
        origin = initial_window + i * _STEP_LENGTH
        if origin + _FH_STEPS > len(y_full):
            f_log(
                f"Walk-forward: stopping early at origin {i + 1} — insufficient data.",
                c_type="warning",
            )
            break

        y_tr  = y_full.iloc[:origin]
        y_fut = y_full.iloc[origin:origin + _FH_STEPS]
        X_tr  = X_full.iloc[:origin]
        X_fut = X_full.iloc[origin:origin + _FH_STEPS]

        estimator = clone(fitted_model)
        y_pred = _predict_origin(estimator, y_tr, X_tr, X_fut)

        all_y_true.extend(y_fut.values.tolist())
        all_y_pred.extend(np.asarray(y_pred).ravel().tolist())

    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)

    r2   = float(r2_score(y_true_arr, y_pred_arr))
    mae  = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "r2_score":               r2,
            "mean_absolute_error":    mae,
            "root_mean_squared_error": rmse,
            "n_eval_origins":         len(all_y_true) // _FH_STEPS,
            "n_eval_points":          len(all_y_true),
        })

    return {"r2": r2, "mae": mae, "rmse": rmse}


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
