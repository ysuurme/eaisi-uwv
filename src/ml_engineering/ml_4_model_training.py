"""
Step 4 — ML Model Training.

Given an estimator configuration and training data, fits the model
(with optional hyperparameter tuning via GridSearchCV) and logs to MLflow.
"""
import json
from typing import Any, Dict, Tuple

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sqlalchemy.orm import Session

from src.ml_engineering.model_configs import ModelExperiment, ModelTuningRecord
from src.utils.m_log import f_log


class ModelTrainer:
    """Trains a single estimator and logs results to MLflow."""

    def __init__(self, session: Session):
        self.session = session

    def train(
        self, experiment: ModelExperiment, x_train: pd.DataFrame,
        y_train: pd.Series, run_name: str, lineage: Dict[str, Any],
    ) -> Tuple[Any, str]:
        """Fits the estimator inside an MLflow run. Returns (fitted_model, run_id)."""
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            mlflow.set_tags({"data_source": lineage["dataset"], "target": lineage["target"]})
            fitted_model = self._fit_or_tune(experiment, x_train, y_train, run_id)
            self._log_model_artifact(fitted_model, x_train)
            f_log(f"Training complete | Run: {run_name}", c_type="success")
            return fitted_model, run_id

    def _fit_or_tune(self, experiment: ModelExperiment, x_train: pd.DataFrame,
                     y_train: pd.Series, run_id: str) -> Any:
        if not experiment.param_grid:
            experiment.estimator.fit(x_train, y_train)
            return experiment.estimator

        f_log(f"Tuning {experiment.name}...", c_type="process")
        tscv = TimeSeriesSplit(n_splits=5)
        grid = GridSearchCV(experiment.estimator, experiment.param_grid,
                            cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
        grid.fit(x_train, y_train)
        self._store_tuning_results(run_id, grid.cv_results_)
        return grid.best_estimator_

    def _store_tuning_results(self, run_id: str, cv_results: dict) -> None:
        serializable = {k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in cv_results.items()}
        record = ModelTuningRecord(
            run_id=run_id,
            experiment_name=mlflow.active_run().info.experiment_id,
            cv_results_json=json.dumps(serializable),
        )
        self.session.merge(record)

    @staticmethod
    def _log_model_artifact(fitted_model: Any, x_train: pd.DataFrame) -> None:
        signature = infer_signature(x_train, fitted_model.predict(x_train.head(5)))
        mlflow.sklearn.log_model(
            sk_model=fitted_model, name="model", signature=signature,
            input_example=x_train.head(1), conda_env=CONDA_ENV,
            serialization_format="skops",
        )
