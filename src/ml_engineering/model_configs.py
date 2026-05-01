"""
Centralized ML Configuration Module.

Single source of truth for:
- ORM Base class (unified metadata registry for all ML tables)
- ORM table definitions (tuning results, evaluation records)
- Feature catalog (named feature groups with source metadata)
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
# Feature Catalog — named groups of features with source metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureGroup:
    """A named, documented set of features originating from a single CBS source."""
    name: str
    columns: List[str]
    source_table: str
    description: str


# Placeholder column names — fill in with actual gold DB column names after running
# discovery mode (uv run main.py master_data_ml_preprocessed random_forest) and
# inspecting the logged feature count to confirm which columns are available.
FEATURE_CATALOG: Dict[str, FeatureGroup] = {
    "temporal": FeatureGroup(
        name="temporal",
        columns=["year", "quarter", "trend_index"],
        source_table="derived",
        description="Engineered time features: calendar year, quarter, and a linear trend index.",
    ),
    "labor_market": FeatureGroup(
        name="labor_market",
        columns=[
            # TODO: replace with actual column names from master_data_ml_preprocessed
            # e.g. "unemployment_rate", "employment_pct", "labor_participation_rate"
        ],
        source_table="83415NED",
        description="Labor market indicators: employment and unemployment rates by sector.",
    ),
    "stress_indicators": FeatureGroup(
        name="stress_indicators",
        columns=[
            # TODO: replace with actual column names from master_data_ml_preprocessed
            # e.g. "stress_rate", "burnout_rate", "work_pressure_index"
        ],
        source_table="83157NED",
        description="Work-related stress and burnout indicators by sector.",
    ),
}


# ---------------------------------------------------------------------------
# Estimator Configuration Dataclass + Catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelExperiment:
    """Container for a single model experiment configuration."""
    name: str
    estimator: Any
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    feature_groups: Optional[List[str]] = None  # None = all columns (discovery mode)
    description: str = ""


class ModelConfiguration:
    """Catalog of available estimator configurations.

    Usage:
        config = ModelConfiguration.get("linear")
        config.estimator  # LinearRegression()
        config.feature_groups  # None = discovery mode
    """

    _CATALOG: Dict[str, ModelExperiment] = {
        "baseline": ModelExperiment(
            name="Baseline_Mean",
            estimator=DummyRegressor(strategy="mean"),
            feature_groups=None,
            description="Naïve baseline predicting the training set mean."
        ),
        "linear": ModelExperiment(
            name="LinearRegression",
            estimator=LinearRegression(),
            feature_groups=["temporal", "labor_market", "stress_indicators"],
            description="Simple regressor using temporal and labor market features."
        ),
        "random_forest": ModelExperiment(
            name="RandomForest",
            estimator=RandomForestRegressor(random_state=42),
            feature_groups=None,
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
            },
            description="Random Forest Regressor (6 combos x 5 folds = 30 fits)."
        ),
        "gradient_boosting": ModelExperiment(
            name="GradientBoosting",
            estimator=GradientBoostingRegressor(random_state=42),
            feature_groups=None,
            param_grid={
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
            },
            description="Gradient Boosting Regressor (6 combos x 5 folds = 30 fits)."
        ),
        "hist_gradient_boosting": ModelExperiment(
            name="HistGradientBoosting",
            estimator=HistGradientBoostingRegressor(random_state=42),
            feature_groups=None,
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
