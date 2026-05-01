"""
Step 4 — ML Model Training.

Given an estimator configuration and training data, fits the model
(with optional hyperparameter tuning) and logs to MLflow.

Two execution paths based on estimator type:
    SectorQuarterRollingMean  — sklearn-compatible baseline; standard .fit(X, y)
    sktime forecasters        — make_reduction / Prophet; sktime .fit(y, X) API
                                tuned via ForecastingGridSearchCV + ExpandingWindowSplitter
"""
import json
import os
import pickle
import tempfile
from typing import Any, Dict, Tuple

import pandas as pd
import mlflow
import mlflow.sklearn
import skops.io as sio
from sktime.forecasting.model_selection import ExpandingWindowSplitter, ForecastingGridSearchCV
from sqlalchemy.orm import Session

from src.ml_engineering.model_configs import (
    ModelExperiment,
    ModelTuningRecord,
    SectorQuarterRollingMean,
)
from src.utils.m_log import f_log


class ModelTrainer:
    """Trains a single estimator and logs results to MLflow."""

    def __init__(self, session: Session):
        self.session = session

    def train(
        self,
        experiment: ModelExperiment,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        run_name: str,
        lineage: Dict[str, Any],
    ) -> Tuple[Any, str]:
        """Fits the estimator inside an MLflow run. Returns (fitted_model, run_id)."""
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            mlflow.set_tags({
                "data_source": lineage["dataset"],
                "target": lineage["target"],
                "sector": lineage.get("sector", "T001081"),
                "forecast_horizon": "4Q",
            })
            fitted_model = self._fit_or_tune(experiment, x_train, y_train, run_id)
            self._log_model_artifact(fitted_model, x_train, y_train)
            f_log(f"Training complete | Run: {run_name}", c_type="success")
            return fitted_model, run_id

    def _fit_or_tune(
        self,
        experiment: ModelExperiment,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        run_id: str,
    ) -> Any:
        """Dispatches to sklearn or sktime fit path based on estimator type."""
        estimator = experiment.estimator

        # ── Baseline: sklearn-compatible, receives full X (incl. SBI text col) ──
        if isinstance(estimator, SectorQuarterRollingMean):
            estimator.fit(x_train, y_train)
            return estimator

        # ── sktime forecasters: drop the non-numeric SBI text column ──
        X_numeric = x_train.select_dtypes(include="number")
        X = X_numeric if (experiment.feature_groups is not None and not X_numeric.empty) else None

        if not experiment.param_grid:
            estimator.fit(y=y_train, X=X)
            return estimator

        f_log(f"Tuning {experiment.name} with ForecastingGridSearchCV...", c_type="process")
        # fh=[1,2,3,4]: evaluate all 4 quarters of the forecast horizon in each fold.
        # step_length=4: advance one full year per fold (quarterly seasonal alignment).
        # initial_window=40: require 10 years of data before the first CV fold.
        cv = ExpandingWindowSplitter(initial_window=40, step_length=4, fh=[1, 2, 3, 4])
        grid = ForecastingGridSearchCV(
            forecaster=estimator,
            param_grid=experiment.param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(y=y_train, X=X)
        self._store_tuning_results(run_id, grid.cv_results_)
        return grid.best_forecaster_

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
    def _log_model_artifact(
        fitted_model: Any,
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """Logs model params and serialized artifact to MLflow.

        - SectorQuarterRollingMean (sklearn): logged via mlflow.sklearn.log_model
          so that mlflow.models.evaluate can load and score it natively.
        - sktime forecasters: serialized via pickle (skops does not support sktime
          internals) and logged as a raw artifact.
        """
        mlflow.log_param("model_class", type(fitted_model).__name__)
        mlflow.log_param("train_rows", len(y_train))
        mlflow.log_param("feature_count",
                         x_train.select_dtypes(include="number").shape[1])

        if isinstance(fitted_model, SectorQuarterRollingMean):
            # sklearn-compatible → mlflow.sklearn so mlflow.models.evaluate works
            signature = mlflow.models.infer_signature(
                x_train, fitted_model.predict(x_train)
            )
            mlflow.sklearn.log_model(
                sk_model=fitted_model,
                name="model",
                signature=signature,
                input_example=x_train.head(1),
            )
        else:
            # sktime forecaster → pickle artifact
            model_bytes = pickle.dumps(fitted_model)
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                f.write(model_bytes)
                tmp_path = f.name
            mlflow.log_artifact(tmp_path, artifact_path="model")
            os.unlink(tmp_path)
