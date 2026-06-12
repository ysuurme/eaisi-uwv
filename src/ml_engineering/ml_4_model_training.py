"""
Step 4 — ML Model Training.

Given an estimator configuration and training data, fits the model
(with optional hyperparameter tuning) and logs to MLflow.

Two execution paths based on estimator type:
    SectorQuarterRollingMean  — sklearn-compatible baseline; standard .fit(X, y)
                                logged via mlflow.sklearn.log_model so that
                                mlflow.models.evaluate works natively in Step 5.
    sktime forecasters        — make_reduction / Prophet; sktime .fit(y, X) API
                                tuned via ForecastingGridSearchCV + ExpandingWindowSplitter
                                logged via mlflow.pyfunc so the MLflow registry can
                                resolve runs:/{run_id}/model in Step 6.
"""
import hashlib
import json
import os
import pickle
import tempfile
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import ExpandingWindowSplitter, ForecastingGridSearchCV
from sktime.performance_metrics.forecasting import MeanSquaredError
from sqlalchemy.orm import Session

from src.ml_engineering.model_configs import (
    _FEATURE_CATALOG_FILE,
    ModelExperiment,
    ModelTuningRecord,
    SectorQuarterRollingMean,
)
from src.utils.m_log import f_log


# ---------------------------------------------------------------------------
# pyfunc wrapper — bridges MLflow's predict(model_input) with sktime's
# predict(fh, X) API, enabling sktime models to be registered in the registry.
# Defined at module level so mlflow.pyfunc can serialize it correctly.
# ---------------------------------------------------------------------------

def _base_estimator_name(estimator: Any) -> str:
    """Unwrap a model to its underlying algorithm name for clear MLflow metadata.

    sktime's ``make_reduction`` wraps an sklearn estimator (often a ``Pipeline``)
    inside a forecaster whose class is the opaque ``RecursiveTabular
    RegressionForecaster``.  This peels back to the actual algorithm — e.g.
    ``LinearRegression``, ``RandomForestRegressor`` — so the model *type* is
    legible in the MLflow UI.  Falls back to the estimator's own class name
    (e.g. ``SectorQuarterRollingMean`` for the baseline).
    """
    inner = getattr(estimator, "estimator", None)  # sktime reducer → wrapped sklearn est
    if inner is None:
        # QuarterlyPeriodForecaster (and similar adapters) → wrapped forecaster
        inner = getattr(estimator, "forecaster", None)
    target = inner if inner is not None else estimator
    steps = getattr(target, "steps", None)          # sklearn Pipeline → final step
    if steps:
        return type(steps[-1][1]).__name__
    return type(target).__name__


class _SktimePyfuncWrapper(mlflow.pyfunc.PythonModel):
    """Minimal pyfunc wrapper for sktime forecasters.

    Enables MLflow model registration (mlflow.register_model) for sktime
    models whose predict signature — predict(fh, X) — differs from sklearn's
    predict(X).  Hardcodes a 4-quarter relative horizon matching the project's
    forecast objective.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        with open(context.artifacts["model_pkl"], "rb") as f:
            self._forecaster = pickle.load(f)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.Series:
        fh = ForecastingHorizon([1, 2, 3, 4], is_relative=True)
        X = model_input.select_dtypes(include="number")
        return self._forecaster.predict(fh=fh, X=X if not X.empty else None)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

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
                "preset": os.environ.get("PRESET_NAME", "unknown"),
            })
            self._log_lineage(experiment, x_train, lineage)
            fitted_model = self._fit_or_tune(experiment, x_train, y_train, run_id)
            self._log_model_artifact(fitted_model, x_train, y_train)
            f_log(f"Training complete | Run: {run_name}", c_type="success")
            return fitted_model, run_id

    @staticmethod
    def _log_lineage(
        experiment: ModelExperiment,
        x_train: pd.DataFrame,
        lineage: Dict[str, Any],
    ) -> None:
        """Logs feature + config provenance so any run is reproducible.

        Captures the named feature groups, the feature-catalog file, the
        catalog key, the estimator class, and a stable hash of the resolved feature columns
        (also dumped as a ``features.json`` artifact).  Together with the
        best-params logged after tuning, this makes two runs of the same
        estimator distinguishable purely from MLflow metadata.
        """
        numeric_cols = sorted(x_train.select_dtypes(include="number").columns.tolist())
        feature_set_hash = hashlib.sha1(",".join(numeric_cols).encode()).hexdigest()[:12]
        groups = experiment.feature_groups
        base_estimator = _base_estimator_name(experiment.estimator)
        mlflow.log_params({
            "experiment_key": lineage.get("experiment_key", "unknown"),
            "model_name": experiment.name,
            "feature_groups": json.dumps(groups) if groups is not None else "discovery",
            "feature_catalog": _FEATURE_CATALOG_FILE,
            "estimator_class": type(experiment.estimator).__name__,
            "base_estimator_class": base_estimator,
        })
        # Clear, filterable model identity in the MLflow runs table.
        mlflow.set_tags({
            "feature_set_hash": feature_set_hash,
            "model_family": experiment.name,
            "model_type": base_estimator,
        })
        mlflow.log_dict(
            {"resolved_features": numeric_cols, "feature_set_hash": feature_set_hash},
            "features.json",
        )

    def _fit_or_tune(
        self,
        experiment: ModelExperiment,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        run_id: str,
    ) -> Any:
        """Dispatches to sklearn or sktime fit path based on estimator type."""
        estimator = experiment.estimator

        # ── Baseline: sklearn-compatible ──────────────────────────────────────
        if isinstance(estimator, SectorQuarterRollingMean):
            estimator.fit(x_train, y_train)
            return estimator

        # ── sktime forecasters ────────────────────────────────────────────────
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
            scoring=MeanSquaredError(square_root=False),
        )
        grid.fit(y=y_train, X=X)
        self._store_tuning_results(run_id, grid.cv_results_)
        # Reproducibility: the chosen hyperparameters and the full grid searched.
        mlflow.log_params({
            "best_params": json.dumps(grid.best_params_, default=str),
            "param_grid": json.dumps(experiment.param_grid, default=str),
        })
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
        """Logs model params and a registered MLflow model artifact.

        SectorQuarterRollingMean (sklearn-compatible)
            Logged via mlflow.sklearn.log_model — enables mlflow.models.evaluate
            in Step 5 and creates a LoggedModel entry for the registry.

        sktime forecasters
            Pickled then wrapped in _SktimePyfuncWrapper and logged via
            mlflow.pyfunc.log_model — creates a LoggedModel entry so that
            mlflow.register_model(runs:/{run_id}/model) resolves in Step 6.
        """
        mlflow.log_param("model_class", type(fitted_model).__name__)
        mlflow.log_param("train_rows", len(y_train))
        mlflow.log_param("feature_count",
                         x_train.select_dtypes(include="number").shape[1])

        if isinstance(fitted_model, SectorQuarterRollingMean):
            signature = mlflow.models.infer_signature(
                x_train, fitted_model.predict(x_train)
            )
            mlflow.sklearn.log_model(
                sk_model=fitted_model,
                name="model",
                signature=signature,
                input_example=x_train.head(1),
            )
            return

        # sktime: pickle the forecaster, wrap in pyfunc for registry compatibility
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(fitted_model, f)
            tmp_path = f.name

        try:
            mlflow.pyfunc.log_model(
                name="model",
                python_model=_SktimePyfuncWrapper(),
                artifacts={"model_pkl": tmp_path},
                # input_example intentionally omitted: MLflow's validation call passes
                # training rows as future X to the sktime adapter, which then tries to
                # .loc[] them by absolute forecast dates — those dates don't exist in
                # the training index, causing a spurious KeyError during validation.
            )
        finally:
            os.unlink(tmp_path)
