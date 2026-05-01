"""
Step 5 — ML Model Evaluation.

Computes standardized regression metrics on the held-out test set
using mlflow.models.evaluate. Persists evaluation records to eval DB.
"""
import skops.io as sio
import pandas as pd
from typing import Any, Dict

import mlflow
from sqlalchemy.orm import Session

from src.ml_engineering.model_configs import ModelEvaluationRecord
from src.utils.m_log import f_log


class ModelEvaluator:
    """Evaluates a trained model on the test set and persists metrics."""

    def __init__(self, session: Session):
        self.session = session

    def evaluate(
        self, run_id: str, fitted_model: Any, x_test: pd.DataFrame,
        y_test: pd.Series, model_name: str,
    ) -> Dict[str, float]:
        """Computes metrics and persists evaluation record. Returns metrics dict."""
        with mlflow.start_run(run_id=run_id):
            metrics = self._compute_metrics(run_id, x_test, y_test)
            self._persist_record(run_id, model_name, metrics, fitted_model)
            f_log(f"Evaluation | R2: {metrics['r2']:.4f} | MAE: {metrics['mae']:.4f}", c_type="success")
            return metrics

    @staticmethod
    def _compute_metrics(run_id: str, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        eval_data = x_test.copy()
        target_col = y_test.name if y_test.name else "target"
        eval_data[target_col] = y_test.values
        result = mlflow.models.evaluate(
            model=f"runs:/{run_id}/model", data=eval_data,
            targets=target_col, model_type="regressor", evaluators=["default"],
        )
        return {
            "r2": result.metrics.get("r2_score", 0.0),
            "mae": result.metrics.get("mean_absolute_error", 0.0),
            "rmse": result.metrics.get("root_mean_squared_error", 0.0),
        }

    def _persist_record(self, run_id: str, model_name: str,
                        metrics: Dict[str, float], fitted_model: Any) -> None:
        record = ModelEvaluationRecord(
            run_id=run_id, model_name=model_name,
            r2=metrics["r2"], mae=metrics["mae"], rmse=metrics["rmse"],
            passed_gate=0,  # Updated by step 6 after validation
            model_blob=sio.dumps(fitted_model),
        )
        self.session.merge(record)
        f_log("Evaluation record stored | eval_data.db", c_type="store")
