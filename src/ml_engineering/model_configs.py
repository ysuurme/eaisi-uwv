"""
Centralized ML Configuration Module.

Single source of truth for:
- ORM Base class (unified metadata registry for all ML tables)
- ORM table definitions (tuning results, evaluation records)
- Domain-specific baseline forecaster (SectorQuarterRollingMean)
- Feature catalog (named feature groups with source metadata)
- Estimator configurations (ModelConfiguration catalog)

Feature catalog loading
-----------------------
``FEATURE_CATALOG`` is populated in one of two ways:

1. **From a preset JSON** (preferred): if ``DIR_FEATURE_SELECTION`` contains a
   file named in ``_ACTIVE_PRESET_NAME``, ``load_feature_catalog()`` reads it
   and constructs ``FeatureGroup`` instances.  This is the output of the
   ``feature_selection.py`` script.

2. **Hardcoded fallback**: if no preset file exists, the catalog falls back to
   a manually defined dict.  This ensures the pipeline runs even before any
   feature selection has been performed.
"""
import copy
import json
import logging
import numpy as np
import pandas as pd
import polars as pl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.compose import make_reduction
from sqlalchemy import String, Float, Integer, LargeBinary, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func

# Defensive import: DIR_FEATURE_SELECTION may not exist in config.py yet.
# If missing, fall back to the conventional path so the pipeline still runs.
try:
    from src.config import DIR_FEATURE_SELECTION
except ImportError:
    DIR_FEATURE_SELECTION = (
        Path(__file__).resolve().parent.parent.parent / "data" / "feature_selection"
    )

_logger = logging.getLogger(__name__)


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
    passed_gate: Mapped[Optional[int]] = mapped_column(Integer)
    model_blob: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    timestamp: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Domain-Specific Baseline Forecaster
# ---------------------------------------------------------------------------

class SectorQuarterRollingMean(BaseEstimator, RegressorMixin):
    """
    Domain-specific baseline forecaster for quarterly sick leave prediction.

    For each (SBI sector, quarter) group, predicts the mean of the same quarter
    over the previous n_years, using a no-leakage shift(1) partitioned by sector
    and quarter. Computed via Polars for performance.

    Formula:
        ŷ_t = mean(y_{t-1}, y_{t-2}, y_{t-3})   for the same quarter Q and sector S

    Using .over(["sbi_col", "quarter_col"]) ensures each (sector, quarter)
    combination is averaged independently. shift(1) prevents look-ahead bias
    — the current quarter's value is excluded from its own prediction.
    """

    def __init__(
        self,
        n_years: int = 3,
        sbi_col: str = "BedrijfstakkenBranchesSBI2008",
        quarter_col: str = "quarter",
    ):
        self.n_years = n_years
        self.sbi_col = sbi_col
        self.quarter_col = quarter_col
        self._lookup: Dict[tuple, float] = {}
        self._global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SectorQuarterRollingMean":
        """
        Learns the rolling mean lookup from training data.

        Groups by (sbi_col, quarter_col) when sbi_col is present in X;
        falls back to quarter_col-only grouping if sbi_col is absent
        (e.g. when SBI sector is encoded as OHE columns rather than a text column).

        Args:
            X: DataFrame containing at least quarter_col; sbi_col is optional.
            y: Target series (sick leave percentage).
        """
        has_sbi = self.sbi_col in X.columns
        group_cols = [self.sbi_col, self.quarter_col] if has_sbi else [self.quarter_col]

        df = X[group_cols].copy()
        df["_target"] = y.values

        over_cols = group_cols  # Polars .over() partition
        lf = (
            pl.from_pandas(df)
            .with_columns(
                pl.col("_target")
                .shift(1)                          # no look-ahead: exclude current row
                .rolling_mean(
                    window_size=self.n_years,
                    min_samples=self.n_years,      # match notebook: no prediction until n_years full observations exist
                )
                .over(over_cols)
                .alias("_rolling_mean")
            )
        )
        result = lf.select(group_cols + ["_rolling_mean"]).to_pandas()

        # Build lookup: (sbi_code, quarter) → predicted rolling mean
        # Falls back to (quarter,) key tuple when sbi_col is absent
        self._lookup = {}
        for _, row in result.dropna(subset=["_rolling_mean"]).iterrows():
            if has_sbi:
                key = (row[self.sbi_col], int(row[self.quarter_col]))
            else:
                key = (int(row[self.quarter_col]),)
            self._lookup[key] = float(row["_rolling_mean"])

        self._global_mean = float(y.mean())
        self._has_sbi = has_sbi
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts using the learned rolling mean lookup via a vectorised
        left-merge against a DataFrame-form of self._lookup.  Unseen key
        combinations fall back to the global training mean.
        """
        has_sbi = getattr(self, "_has_sbi", self.sbi_col in X.columns)

        # Edge case: empty lookup → everything falls back to global mean.
        if not self._lookup:
            return np.full(len(X), self._global_mean, dtype=float)

        # Build query keys frame in row-order of X (merge preserves left order).
        if has_sbi:
            keys = pd.DataFrame({
                self.sbi_col: X[self.sbi_col].values,
                self.quarter_col: X[self.quarter_col].astype(int).values,
            })
            lookup_df = pd.DataFrame(
                [(k[0], k[1], v) for k, v in self._lookup.items()],
                columns=[self.sbi_col, self.quarter_col, "_pred"],
            )
            on = [self.sbi_col, self.quarter_col]
        else:
            keys = pd.DataFrame({
                self.quarter_col: X[self.quarter_col].astype(int).values,
            })
            lookup_df = pd.DataFrame(
                [(k[0], v) for k, v in self._lookup.items()],
                columns=[self.quarter_col, "_pred"],
            )
            on = [self.quarter_col]

        merged = keys.merge(lookup_df, on=on, how="left")
        return merged["_pred"].fillna(self._global_mean).to_numpy(dtype=float)


class PLS1D(PLSRegression):
    """``PLSRegression`` that returns 1-D predictions for single-target y.

    sklearn's PLSRegression.predict returns shape (n_samples, n_targets), so
    for our 1-D quarterly sick-leave target it returns (n_samples, 1).  The
    sktime reducer wraps the inner regressor and expects 1-D output from
    ``predict``; passing a 2-D array breaks downstream metric computations
    and concatenation in walk-forward CV.

    This thin subclass squeezes the trailing singleton dimension when
    present.  It inherits everything else — ``fit``, params, ``set_params`` —
    so it composes correctly inside ``Pipeline`` and ``make_reduction``,
    and ``sklearn.base.clone`` works without changes.
    """

    def predict(self, X, copy=True):
        y_pred = super().predict(X, copy=copy)
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred


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


def load_feature_catalog(preset_path: Path) -> Dict[str, FeatureGroup]:
    """Build a ``FEATURE_CATALOG`` from a feature selection preset JSON.

    The JSON is produced by ``feature_selection.py``'s ``build_presets()``
    function.  Its ``feature_groups`` dict maps group names to dicts with
    ``columns``, ``source_table``, and ``description`` — exactly the fields
    of ``FeatureGroup``.

    In addition to the named groups, an ``all_survivors`` meta-group is
    created containing every feature that survived the filter pipeline.
    Models can reference ``feature_groups=["all_survivors"]`` to use the
    full preset without needing to list individual groups.

    Parameters
    ----------
    preset_path : Path
        Path to a preset JSON file (e.g. ``preset_domain_labor.json``).

    Returns
    -------
    dict[str, FeatureGroup]
        Feature catalog ready for use by ``DataExtractor._resolve_groups()``.

    Raises
    ------
    FileNotFoundError
        If the preset file does not exist.
    KeyError
        If the JSON is missing the ``feature_groups`` key.
    """
    with open(preset_path) as fh:
        preset = json.load(fh)

    groups = preset["feature_groups"]
    catalog: Dict[str, FeatureGroup] = {}
    for name, meta in groups.items():
        catalog[name] = FeatureGroup(
            name=name,
            columns=meta["columns"],
            source_table=meta.get("source_table", ""),
            description=meta.get("description", ""),
        )

    # Add a catch-all group containing every surviving feature, regardless
    # of which named group it belongs to.  This lets models use
    # feature_groups=["all_survivors"] to get the full preset.
    all_features = preset.get("surviving_features", [])
    if all_features:
        catalog["all_survivors"] = FeatureGroup(
            name="all_survivors",
            columns=all_features,
            source_table="",
            description=(
                f"All {len(all_features)} features that survived the "
                f"'{preset.get('preset', 'unknown')}' filter pipeline."
            ),
        )

    # Diagnostic group: empty columns list.  Combined with ml_1's
    # _KEEP_STRUCTURAL injection (period_enddate, year, quarter, trend_index,
    # covid_period, post_covid), this resolves to "structural features only" —
    # the feature set used by the structural_linear diagnostic model.
    catalog["structural_only"] = FeatureGroup(
        name="structural_only",
        columns=[],
        source_table="",
        description="Empty group; resolves to structural features only via ml_1 injection.",
    )

    return catalog


# ---------------------------------------------------------------------------
# FEATURE_CATALOG — loaded from preset JSON or hardcoded fallback
# ---------------------------------------------------------------------------

#: Name of the active preset file inside DIR_FEATURE_SELECTION.
#: Change this to switch between presets (e.g. "preset_minimal_stable.json").
_ACTIVE_PRESET_NAME: str = "preset_domain_labor.json"

_HARDCODED_CATALOG: Dict[str, FeatureGroup] = {
    "labor_volume": FeatureGroup(
        name="labor_volume",
        columns=[
            "GewerkteUren_3_A045285_4000",
            "GewerkteUren_3_A045285_3000",
            "GewerkteUren_3_A045286_3000",
            "GewerkteUren_3_A045286_4000",
            "GewerkteUren_8_A045285_3000",
            "GewerkteUren_8_A045286_3000",
            "GewerkteUren_8_A045286_4000",
            "GewerkteUren_5_A045285",
            "GewerkteUrenPerWerkzamePersoon_4_A045285_4000",
            "GewerkteUrenPerWerkzamePersoon_9_A045285_4000",
            "GewerkteUrenPerWerkzamePersoon_9_A045285_3000",
            "GewerkteUrenPerWerkzamePersoon_9_A045286_4000",
            "GewerkteUrenPerBaan_10_A045285_3000",
            "GewerkteUrenSeizoengecorrigeerd_6_A045285",
        ],
        source_table="85920NED",
        description="Hours worked, hours per employed person, and hours per job by sector and category (85920NED).",
    ),
    "workforce": FeatureGroup(
        name="workforce",
        columns=[
            "WerkzamePersonen_6_A045285_3000",
            "WerkzamePersonen_7_A045285",
            "WerkzamePersonen_7_A045286",
            "WerkzamePersonenSeizoengecorrigeerd_9_A045286",
            "Banen_7_A045285_4000",
            "Banen_7_A045286_3000",
            "Banen_2_A045286_4000",
            "Banen_8_A045285",
            "BanenSeizoengecorrigeerd_10_A045286",
            "Banen_8_A045286",
        ],
        source_table="85920NED",
        description="Number of jobs and employed persons, seasonally adjusted, by sector and category (85920NED).",
    ),
}

# Meta-groups derived from the hardcoded named groups.  Added post-hoc because
# FeatureGroup is frozen and cannot reference sibling entries during dict
# construction.  Ensures the fallback path supports the same group names
# (all_survivors, structural_only) as a preset-loaded catalog — otherwise
# models referencing those groups would crash if the preset file is missing.
_HARDCODED_CATALOG["all_survivors"] = FeatureGroup(
    name="all_survivors",
    columns=(
        _HARDCODED_CATALOG["labor_volume"].columns
        + _HARDCODED_CATALOG["workforce"].columns
    ),
    source_table="",
    description="Hardcoded fallback: union of all named groups (labor_volume + workforce).",
)
_HARDCODED_CATALOG["structural_only"] = FeatureGroup(
    name="structural_only",
    columns=[],
    source_table="",
    description="Empty group; resolves to structural features only via ml_1 injection.",
)

# Try loading from preset; fall back to hardcoded if the file doesn't exist
# or is malformed.  A bad preset file must never crash the pipeline at import.
_preset_path = DIR_FEATURE_SELECTION / _ACTIVE_PRESET_NAME
FEATURE_CATALOG: Dict[str, FeatureGroup]

if _preset_path.exists():
    try:
        FEATURE_CATALOG = load_feature_catalog(_preset_path)
        _logger.debug(
            "FEATURE_CATALOG loaded from preset: %s (%d groups)",
            _preset_path.name, len(FEATURE_CATALOG),
        )
    except Exception as _exc:
        FEATURE_CATALOG = _HARDCODED_CATALOG
        _logger.warning(
            "Failed to load preset %s: %s — using hardcoded fallback",
            _preset_path.name, _exc,
        )
else:
    FEATURE_CATALOG = _HARDCODED_CATALOG
    _logger.debug(
        "FEATURE_CATALOG using hardcoded fallback (no preset at %s)",
        _preset_path,
    )


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
        config = ModelConfiguration.get("baseline")
        config.estimator  # SectorQuarterRollingMean()
        config.feature_groups  # None = discovery mode

    Baseline:
        SectorQuarterRollingMean — rolling 3-year same-quarter mean per SBI sector.
        Domain-specific, interpretable, and leakage-free benchmark.

    Reducers (sktime make_reduction):
        Wraps sklearn regressors into recursive lag-window forecasters.
        For linear-family models (linear / ridge / elasticnet / structural_linear)
        the underlying regressor is wrapped in a sklearn Pipeline with a
        StandardScaler step.  This adds an extra nesting level to param_grid
        keys:  estimator__regressor__<param>  rather than  estimator__<param>.
        Scaling is critical for L1/L2-penalised models on CBS data because raw
        labour-volume features (hours, persons) dwarf scaled features in
        magnitude and would otherwise hijack the regularisation budget.
        Tree-family models (random_forest / hist_gbr) are scale-invariant and
        are NOT wrapped in a scaler.
    """

    _CATALOG: Dict[str, ModelExperiment] = {
        "baseline": ModelExperiment(
            name="SectorQuarterRollingMean",
            estimator=SectorQuarterRollingMean(n_years=3),
            param_grid={},
            feature_groups=None,  # uses sbi_col + quarter_col from X; no catalog groups needed
            description="Rolling 3-year same-quarter mean per SBI sector. No look-ahead bias.",
        ),
        "structural_linear": ModelExperiment(
            name="StructuralLinear",
            estimator=make_reduction(
                Pipeline([("scaler", StandardScaler()), ("regressor", Ridge())]),
                window_length=4, strategy="recursive",
            ),
            param_grid={
                "window_length": [4, 8],
                "estimator__regressor__alpha": [0.1, 1.0, 10.0, 100.0],
            },
            feature_groups=["structural_only"],
            description="Diagnostic floor: Ridge on structural features only (trend, regime flags, temporal).",
        ),
        "linear": ModelExperiment(
            name="LinearRegression_Reduced",
            estimator=make_reduction(
                Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())]),
                window_length=4, strategy="recursive",
            ),
            param_grid={"window_length": [4, 8]},
            feature_groups=["labor_volume"],
            description="LinearRegression in scaler pipeline via make_reduction recursive lag-window forecaster.",
        ),
        "random_forest": ModelExperiment(
            name="RandomForest_Reduced",
            estimator=make_reduction(
                RandomForestRegressor(random_state=42), window_length=12, strategy="recursive"
            ),
            param_grid={
                "window_length": [4, 8],
                "estimator__n_estimators": [100, 200],
                "estimator__max_depth": [5, 10, None],
                "estimator__min_samples_leaf": [5, 10],
            },
            feature_groups=["labor_volume", "workforce"],
            description="RandomForest via make_reduction (recursive). min_samples_leaf regularises small-N fits.",
        ),
        "gradient_boosting": ModelExperiment(
            name="GradientBoosting_Reduced",
            estimator=make_reduction(
                GradientBoostingRegressor(random_state=42), window_length=12, strategy="recursive"
            ),
            param_grid={
                "window_length": [4, 8],
                "estimator__n_estimators": [100, 200],
                "estimator__learning_rate": [0.05, 0.1],
            },
            feature_groups=["labor_volume", "workforce"],
            description="GradientBoosting via make_reduction (recursive).",
        ),
        "ridge": ModelExperiment(
            name="Ridge_Reduced",
            estimator=make_reduction(
                Pipeline([("scaler", StandardScaler()), ("regressor", Ridge())]),
                window_length=4, strategy="recursive",
            ),
            param_grid={
                "window_length": [4, 8],
                "estimator__regressor__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
            },
            feature_groups=["all_survivors"],
            description="Ridge in scaler pipeline via make_reduction. L2 regularization.",
        ),
        "elasticnet": ModelExperiment(
            name="ElasticNet_Reduced",
            estimator=make_reduction(
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", ElasticNet(max_iter=10_000)),
                ]),
                window_length=4, strategy="recursive",
            ),
            param_grid={
                "window_length": [4, 8],
                "estimator__regressor__alpha": [0.01, 0.1, 1.0, 10.0],
                "estimator__regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            feature_groups=["all_survivors"],
            description="ElasticNet in scaler pipeline via make_reduction. L1+L2 regularization.",
        ),
        "hist_gbr": ModelExperiment(
            name="HistGBR_Reduced",
            estimator=make_reduction(
                HistGradientBoostingRegressor(random_state=42),
                window_length=12, strategy="recursive",
            ),
            param_grid={
                "window_length": [4, 8],
                "estimator__max_iter": [100, 300],
                "estimator__max_depth": [3, 5],
                "estimator__learning_rate": [0.05, 0.1],
                "estimator__min_samples_leaf": [10, 20],
            },
            feature_groups=["all_survivors"],
            description="HistGradientBoosting via make_reduction. Non-linear. Preset-driven.",
        ),
        "pls": ModelExperiment(
            name="PLS_Reduced",
            estimator=make_reduction(
                Pipeline([
                    ("scaler", StandardScaler()),
                    # scale=False because StandardScaler already handles scaling;
                    # PLS1D fixes sklearn's 2-D predict-output quirk so this
                    # composes correctly inside the sktime reducer.
                    ("regressor", PLS1D(n_components=10, scale=False, max_iter=1_000)),
                ]),
                window_length=4, strategy="recursive",
            ),
            param_grid={
                "window_length": [4, 8],
                # n_components is the dimensionality-reduction knob: how many
                # supervised components PLS extracts.  Cap at 20 because
                # n_samples (~70 after lag burn-in) and the feature count
                # bound PLS's rank — values much above 20 add noise.
                "estimator__regressor__n_components": [2, 5, 10, 20],
            },
            feature_groups=["all_survivors"],
            description=(
                "Partial Least Squares regression in scaler pipeline. "
                "Supervised dimensionality reduction — components chosen "
                "to maximise covariance with y, not variance in X.  Often "
                "outperforms Ridge in p>>n regimes."
            ),
        ),
        # Removed: "hist_gradient_boosting" was a near-duplicate of "hist_gbr" and
        # was never referenced by run_overnight.MODELS.  Single source of truth.
        #
        # Prophet removed: unsuitable for quarterly 4Q-ahead forecasting.
        # Reasons: (1) designed for high-frequency data (daily/weekly), not 4 obs/year;
        # (2) 400 regressors on ~100 training rows causes extreme overfitting (R² ≈ -2300);
        # (3) PerformanceWarning from fragmented DataFrame at 400 columns;
        # (4) sktime Prophet adapter requires exact date-index alignment for future X
        #     which conflicts with freq=None DatetimeIndex used throughout the pipeline.
    }

    @classmethod
    def get(cls, key: str) -> ModelExperiment:
        """Retrieves an experiment by its catalog key, returning an isolated copy.

        Isolation prevents state pollution across the (sector × preset × model)
        sweep loop:
        - estimator is sklearn.base.clone()'d → unfitted, no leaked fitted state
        - param_grid is deepcopy()'d → mutations don't propagate across loops
        - feature_groups is a fresh list → independent of catalog entry's list
        """
        if key not in cls._CATALOG:
            available = ", ".join(cls._CATALOG.keys())
            raise ValueError(f"Experiment '{key}' not found. Available: {available}")
        src = cls._CATALOG[key]
        return ModelExperiment(
            name=src.name,
            estimator=clone(src.estimator),
            param_grid=copy.deepcopy(src.param_grid),
            # Preserve None vs [] distinction (ml_1 treats them differently):
            #   None  → discovery mode (all non-structural columns)
            #   []    → groups mode with zero groups (structural-only via injection)
            feature_groups=(
                list(src.feature_groups) if src.feature_groups is not None else None
            ),
            description=src.description,
        )

    @classmethod
    def get_all(cls) -> List[ModelExperiment]:
        """Returns all registered experiments (each as an isolated copy)."""
        return [cls.get(key) for key in cls._CATALOG.keys()]

    @classmethod
    def get_tuning_suite(cls) -> List[ModelExperiment]:
        """Returns only models that have a non-empty param_grid (isolated copies)."""
        return [cls.get(key) for key, exp in cls._CATALOG.items() if exp.param_grid]
