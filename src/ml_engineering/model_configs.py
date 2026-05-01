"""
Centralized ML Configuration Module.

Single source of truth for:
- ORM Base class (unified metadata registry for all ML tables)
- ORM table definitions (tuning results, evaluation records)
- Estimator configurations (ModelConfiguration catalog)
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sqlalchemy import String, Float, LargeBinary, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


# ---------------------------------------------------------------------------
# ORM Base — single metadata registry for all ML evaluation tables
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Base class for all ML evaluation ORM models."""
    pass


class ModelTuningRecord(Base):
    """Stores full GridSearchCV results as serialized JSON per run."""
    __tablename__ = "model_tuning_results"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    experiment_name: Mapped[str] = mapped_column(String)
    cv_results_json: Mapped[str] = mapped_column(String)
    timestamp: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.now())


class ModelEvaluationRecord(Base):
    """Stores test-set metrics and serialized model blob per run."""
    __tablename__ = "model_evaluations"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_name: Mapped[Optional[str]] = mapped_column(String)
    r2: Mapped[Optional[float]] = mapped_column(Float)
    mae: Mapped[Optional[float]] = mapped_column(Float)
    rmse: Mapped[Optional[float]] = mapped_column(Float)
    passed_gate: Mapped[Optional[int]] = mapped_column(Float)
    model_blob: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    timestamp: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Estimator Configuration Dataclass + Catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelExperiment:
    """Container for a single model experiment configuration."""
    name: str
    estimator: Any
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    features: Optional[List[str]] = None
    description: str = ""


class ModelConfiguration:
    """Catalog of available estimator configurations.

    Usage:
        config = ModelConfiguration.get("linear")
        config.estimator  # LinearRegression()
        config.param_grid  # {}
    """

    _CATALOG: Dict[str, ModelExperiment] = {
        "baseline": ModelExperiment(
            name="Baseline_Mean",
            estimator=DummyRegressor(strategy="mean"),
            features=None,  # Uses all numeric features for context, though ignored by Dummy
            description="Naïve baseline predicting the training set mean."
        ),
        "linear": ModelExperiment(
            name="LinearRegression",
            estimator=LinearRegression(),
            features=["unemployment_rate", "stress_rate", "employment_pct"],  # Example subset
            description="Simple regressor capturing the linear trend."
        ),
        "random_forest": ModelExperiment(
            name="RandomForest",
            estimator=RandomForestRegressor(random_state=42),
            features=None,  # Discovery mode (all numeric features)
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
            },
            description="Random Forest Regressor (6 combos x 5 folds = 30 fits)."
        ),
        "gradient_boosting": ModelExperiment(
            name="GradientBoosting",
            estimator=GradientBoostingRegressor(random_state=42),
            features=None,
            param_grid={
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
            },
            description="Gradient Boosting Regressor (6 combos x 5 folds = 30 fits)."
        ),
        "hist_gradient_boosting": ModelExperiment(
            name="HistGradientBoosting",
            estimator=HistGradientBoostingRegressor(random_state=42),
            features=None,
            param_grid={
                "learning_rate": [0.05, 0.1, 0.2],
                "max_iter": [100, 300],
                "max_leaf_nodes": [30, 60],
            },
            description="Histogram-based Gradient Boosting (12 combos x 5 folds = 60 fits)."
        )
    }

    @classmethod
    def get(cls, key: str) -> ModelExperiment:
        """Retrieves an experiment by its catalog key."""
        if key not in cls._CATALOG:
            available = ", ".join(cls._CATALOG.keys())
            raise ValueError(f"Experiment '{key}' not found. Available: {available}")
        return cls._CATALOG[key]

    @classmethod
    def get_all(cls) -> List[ModelExperiment]:
        """Returns all registered experiments."""
        return list(cls._CATALOG.values())

    @classmethod
    def get_tuning_suite(cls) -> List[ModelExperiment]:
        """Returns only models that have a non-empty param_grid."""
        return [exp for exp in cls._CATALOG.values() if exp.param_grid]
