"""
Step 5 — ML Model Evaluation.

Computes standardized regression metrics on the held-out test set.
Persists evaluation records to the eval DB.

Two evaluation paths based on estimator type:
    SectorQuarterRollingMean  — sklearn-compatible; uses mlflow.models.evaluate
                                for native MLflow observability (shap, residuals, etc.)
    sktime forecasters        — uses ForecastingHorizon + .predict(fh, X);
                                logs metrics manually via mlflow.log_metrics
"""
import numpy as np
import pickle
from typing import Any, Dict

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sktime.forecasting.base import ForecastingHorizon
from sqlalchemy.orm import Session

from src.ml_engineering.model_configs import ModelEvaluationRecord, SectorQuarterRollingMean
from src.utils.m_log import f_log


class ModelEvaluator:
    """Evaluates a trained model on the test set and persists metrics."""

    def __init__(self, session: Session):
        self.session = session

    def evaluate(
        self,
        run_id: str,
        fitted_model: Any,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
    ) -> Dict[str, float]:
        """Computes metrics and persists evaluation record. Returns metrics dict."""
        metrics = self._compute_metrics(fitted_model, x_test, y_test, run_id)
        self._persist_record(run_id, model_name, metrics, fitted_model)
        f_log(
            f"Evaluation | R2: {metrics['r2']:.4f} | MAE: {metrics['mae']:.4f} | "
            f"RMSE: {metrics['rmse']:.4f}",
            c_type="success",
        )
        return metrics

    @staticmethod
    def _compute_metrics(
        fitted_model: Any,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        run_id: str,
    ) -> Dict[str, float]:
        """Dispatches to the appropriate evaluation strategy based on model type."""

        if isinstance(fitted_model, SectorQuarterRollingMean):
            # ── sklearn path: use mlflow.models.evaluate for native MLflow observability ──
            # Builds an eval DataFrame (features + target) and scores the registered model.
            eval_data = x_test.copy()
            target_col = y_test.name if y_test.name else "target"
            eval_data[target_col] = y_test.values

            with mlflow.start_run(run_id=run_id):
                result = mlflow.models.evaluate(
                    model=f"runs:/{run_id}/model",
                    data=eval_data,
                    targets=target_col,
                    model_type="regressor",
                    evaluators=["default"],
                )
            return {
                "r2":   result.metrics.get("r2_score", 0.0),
                "mae":  result.metrics.get("mean_absolute_error", 0.0),
                "rmse": result.metrics.get("root_mean_squared_error", 0.0),
            }

        # ── sktime path: ForecastingHorizon + predict, then log metrics manually ──
        X_numeric = x_test.select_dtypes(include="number")
        X = X_numeric if not X_numeric.empty else None
        fh = ForecastingHorizon(y_test.index, is_relative=False)
        y_pred = fitted_model.predict(fh=fh, X=X)

        r2   = float(r2_score(y_test, y_pred))
        mae  = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                "r2_score": r2,
                "mean_absolute_error": mae,
                "root_mean_squared_error": rmse,
            })
        return {"r2": r2, "mae": mae, "rmse": rmse}

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
            model_blob=pickle.dumps(fitted_model) if isinstance(fitted_model, SectorQuarterRollingMean)
                       else None,  # sktime blobs stored as pickle artifact in Step 4
        )
        self.session.merge(record)
        f_log("Evaluation record stored | eval_data.db", c_type="store")
