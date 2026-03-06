"""
Centralized registry for ML model configurations.
Provides a scalable way to define and retrieve estimators and 
their associated hyperparameter search spaces.
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

@dataclass(frozen=True)
class ModelExperiment:
    """Container for a single model experiment configuration."""
    name: str
    estimator: Any
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    description: str = ""

class ModelRegistry:
    """Factory and registry for ModelExperiment configurations."""
    
    _REGISTRY: Dict[str, ModelExperiment] = {
        "baseline": ModelExperiment(
            name="Baseline_Mean",
            estimator=DummyRegressor(strategy="mean"),
            description="Naïve baseline predicting the training set mean."
        ),
        "linear": ModelExperiment(
            name="LinearRegression",
            estimator=LinearRegression(),
            description="Simple regressor capturing the linear trend."
        ),
        "random_forest": ModelExperiment(
            name="RandomForest",
            estimator=RandomForestRegressor(random_state=42),
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
            },
            description="Random Forest Regressor (6 combos x 5 folds = 30 fits)."
        ),
        "gradient_boosting": ModelExperiment(
            name="GradientBoosting",
            estimator=GradientBoostingRegressor(random_state=42),
            param_grid={
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
            },
            description="Gradient Boosting Regressor (6 combos x 5 folds = 30 fits)."
        ),
        "hist_gradient_boosting": ModelExperiment(
            name="HistGradientBoosting",
            estimator=HistGradientBoostingRegressor(random_state=42),
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
        """Retrieves an experiment by its registry key."""
        if key not in cls._REGISTRY:
            available = ", ".join(cls._REGISTRY.keys())
            raise ValueError(f"Experiment '{key}' not found. Available: {available}")
        return cls._REGISTRY[key]

    @classmethod
    def get_all(cls) -> List[ModelExperiment]:
        """Returns all registered experiments."""
        return list(cls._REGISTRY.values())

    @classmethod
    def get_tuning_suite(cls) -> List[ModelExperiment]:
        """Returns only models that have a non-empty param_grid."""
        return [exp for exp in cls._REGISTRY.values() if exp.param_grid]
