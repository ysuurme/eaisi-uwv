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

1. **From the canonical catalog JSON** (preferred): if ``DIR_FEATURE_SELECTION``
   contains the file named in ``_FEATURE_CATALOG_FILE``,
   ``load_feature_catalog()`` reads it and constructs ``FeatureGroup``
   instances.  This is the output of the feature-selection flow
   (``python main.py --select-features``).

2. **Hardcoded fallback**: if no catalog file exists, the catalog falls back to
   a manually defined dict so the pipeline can still run — but a warning is
   emitted because the fallback is not derived from the current gold dataset.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.compose import TransformedTargetForecaster, make_reduction
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import STLForecaster
from sktime.transformations.series.detrend import Deseasonalizer
from sqlalchemy import String, Float, Integer, DateTime
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
    mase: Mapped[Optional[float]] = mapped_column(Float)  # THE comparison metric (seasonal-naive m=4)
    r2: Mapped[Optional[float]] = mapped_column(Float)
    mae: Mapped[Optional[float]] = mapped_column(Float)
    mape: Mapped[Optional[float]] = mapped_column(Float)
    rmse: Mapped[Optional[float]] = mapped_column(Float)
    passed_gate: Mapped[Optional[int]] = mapped_column(Integer)
    timestamp: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.now())

class ModelPredictionRecord(Base):  # type: ignore  # Base imported from this module in the actual file
    """Stores per-row walk-forward predictions for cross-model comparison.

    One row per (run_id, origin_date, horizon).  For the standard 4Q
    walk-forward with 5 rolling origins, each run produces 20 rows
    (2 inner + 3 outer origins × 4 horizons each).
    Enables MASE, per-horizon decay, regime split, and Diebold-Mariano
    tests in the cross-model comparison notebook, which all require
    row-level y_true/y_pred (not aggregate metrics).

    The (run_id, model_name) pair links back to ModelEvaluationRecord and
    to MLflow's runs table for full provenance.  The fold_set column
    enables honest nested cross-validation: inner-fold rows are used by
    m_pipeline_loader to PICK the winning variant per sector, outer-fold
    rows are reported as the honest out-of-sample evaluation.
    """
    __tablename__ = "model_predictions"

    # Surrogate key — each prediction row is independent
    id:          Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Provenance — joins to model_evaluations and MLflow runs
    run_id:      Mapped[str]  = mapped_column(String, index=True)
    model_name:  Mapped[str]  = mapped_column(String, index=True)
    sector_code: Mapped[str]  = mapped_column(String, index=True)

    # Walk-forward coordinates
    origin_date: Mapped[str]  = mapped_column(DateTime)
    target_date: Mapped[str]  = mapped_column(DateTime, index=True)
    horizon:     Mapped[int]  = mapped_column(Integer)

    # Nested-CV labelling (added for honest variant selection)
    # "inner" = early walk-forward origins used by m_pipeline_loader to PICK
    #           the winning Pipeline variant per sector (variant selection).
    # "outer" = later walk-forward origins reported as the honest out-of-
    #           sample evaluation of the chosen variant.  These predictions
    #           are NEVER inspected during selection.
    # ml_5_model_evaluation.py splits the rolling origins 40/60 by default —
    # for n_test_points=20 (5 origins), this is 2 inner + 3 outer origins.
    # server_default ensures the SQL-level DEFAULT is 'outer' so that
    # ALTER TABLE migrations and CREATE TABLE produce equivalent DDL.
    fold_set:    Mapped[str]  = mapped_column(
        String(8), default="outer", server_default="outer",
    )

    # Prediction pair
    y_true:      Mapped[float] = mapped_column(Float)
    y_pred:      Mapped[float] = mapped_column(Float)

    # Auto-populated insert time
    timestamp:   Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.now())


class SectorPerformance(Base):
    """Denormalised read-model for visualizations / a future explorer app.

    One row per SBI sector, materialised (refreshed) FROM the single source of
    truth — the MLflow registry champion (which self-describes model_family /
    model_type / feature_groups / mase / mae / mape / r2), the baseline MASE from
    ``model_evaluations``, and the CBS SBI hierarchy (title + level).  This table
    is a projection: ``m_sector_quality.refresh_sector_performance`` is the only
    writer, so MLflow remains authoritative and this stays a clean, queryable
    surface for charts/apps (champion · model type · sector · performance vs
    baseline being the leading attributes).
    """
    __tablename__ = "sector_performance"

    sector_code:   Mapped[str] = mapped_column(String, primary_key=True)
    sbi_title:     Mapped[Optional[str]] = mapped_column(String)
    sbi_level:     Mapped[Optional[str]] = mapped_column(String)
    model_family:  Mapped[Optional[str]] = mapped_column(String)
    model_type:    Mapped[Optional[str]] = mapped_column(String)
    feature_groups: Mapped[Optional[str]] = mapped_column(String)
    mase:          Mapped[Optional[float]] = mapped_column(Float)  # THE comparison metric (seasonal-naive m=4)
    baseline_mase: Mapped[Optional[float]] = mapped_column(Float)  # baseline model's MASE (reference)
    champion_mae:  Mapped[Optional[float]] = mapped_column(Float)  # primary stakeholder metric (percentage points)
    champion_mape: Mapped[Optional[float]] = mapped_column(Float)  # foundational percentage-error metric
    r2:            Mapped[Optional[float]] = mapped_column(Float)
    tier:          Mapped[Optional[str]] = mapped_column(String)
    refreshed_at:  Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.now())


class ModelForecastRecord(Base):
    """Forward 4Q forecast produced by a sector's ``@prod`` champion (Step 7).

    Additive durable projection written by ``ml_orchestrator.run_forecast``
    (delete-then-insert per ``sector_code``), mirroring the
    ``model_predictions`` / ``sector_performance`` pattern: MLflow stays the
    single source of truth (the champion + its lineage), and this table is a
    queryable surface for charts/apps and downstream consumers.

    One row per ``(sector_code, target_date / horizon)``.  Unlike
    ``model_predictions`` — walk-forward ``y_true`` vs ``y_pred`` on observed
    HISTORY — these rows are genuine out-of-sample forecasts for quarters that
    have not happened yet, so there is no ``y_true``.  ``forecast_made_on`` is
    the origin (the last observed quarter end the champion was refit through);
    ``feature_catalog_hash`` ties the forecast back to the exact resolved feature
    set the champion was trained on (the producing run's ``feature_set_hash``
    tag), so a forecast is reproducible from MLflow metadata alone.
    """
    __tablename__ = "model_forecasts"

    # Surrogate key — each forecast row is independent
    id:                   Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Sector + champion provenance (resolved from the registry @prod alias)
    sector_code:          Mapped[str] = mapped_column(String, index=True)
    model_family:         Mapped[Optional[str]] = mapped_column(String)
    model_type:           Mapped[Optional[str]] = mapped_column(String)
    experiment_key:       Mapped[Optional[str]] = mapped_column(String)
    champion_version:     Mapped[Optional[str]] = mapped_column(String)

    # Forecast coordinates — origin (last observed quarter) → future target dates
    forecast_made_on:     Mapped[str] = mapped_column(DateTime)
    target_date:          Mapped[str] = mapped_column(DateTime, index=True)
    horizon:              Mapped[int] = mapped_column(Integer)

    # The forward prediction (no y_true — these quarters have not happened)
    y_pred:               Mapped[float] = mapped_column(Float)

    # Reproducibility: the champion's resolved feature-set hash (ml_4 run tag)
    feature_catalog_hash: Mapped[Optional[str]] = mapped_column(String)

    # Auto-populated insert time
    generated_at:         Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.now())


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


class QuarterlyPeriodForecaster(BaseForecaster):
    """Runs a wrapped sktime forecaster on a quarterly PeriodIndex.

    The pipeline hands sktime forecasters a DatetimeIndex with ``freq=None``
    (set deliberately in Step 3).  Reducer-style forecasters handle that fine,
    but decomposition-based estimators (``STLForecaster``, ``Deseasonalizer``,
    ``Detrender``) require an index with frequency information and fail with
    "You must pass a freq argument" / "frequency is missing".

    This wrapper converts y/X to a quarterly ``PeriodIndex`` before delegating
    to the wrapped forecaster, and converts predictions back to normalized
    quarter-end timestamps so the rest of the pipeline (walk-forward evaluation,
    prediction persistence, pyfunc serialization) sees the same index convention
    as every other estimator.

    It is a regular sktime forecaster: ``get_params``/``set_params`` expose the
    wrapped forecaster as the ``forecaster`` parameter, so grid search reaches
    inner hyperparameters via ``forecaster__<param>`` keys and
    ``sklearn.base.clone`` / pickling work unchanged.

    Args:
        forecaster: Any sktime forecaster requiring a frequency-aware index
            (e.g. ``STLForecaster`` or a ``TransformedTargetForecaster`` with
            ``Deseasonalizer``/``Detrender`` steps).
    """

    _tags = {
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:pred_int": False,
    }

    def __init__(self, forecaster: BaseForecaster):
        self.forecaster = forecaster
        super().__init__()

    def _fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None, fh=None):
        self.forecaster_ = self.forecaster.clone()
        self.forecaster_.fit(y=_to_period(y), X=_to_period(X), fh=fh)
        return self

    def _predict(self, fh, X: Optional[pd.DataFrame] = None) -> pd.Series:
        y_pred = self.forecaster_.predict(fh=fh, X=_to_period(X))
        if isinstance(y_pred.index, pd.PeriodIndex):
            y_pred = y_pred.copy()
            y_pred.index = pd.DatetimeIndex(
                y_pred.index.to_timestamp(how="end").normalize(), freq=None
            )
        return y_pred


def _to_period(data):
    """Return a copy of ``data`` re-indexed on a quarterly PeriodIndex."""
    if data is None:
        return None
    converted = data.copy()
    converted.index = pd.DatetimeIndex(converted.index).to_period("Q")
    return converted


# ---------------------------------------------------------------------------
# Chronos-Bolt — zero-shot univariate foundation-model forecaster
# ---------------------------------------------------------------------------
# Amazon Chronos-Bolt is a pretrained T5 time-series model: there is NO training
# on our data — ``fit`` stores the target context, ``predict`` runs a forward
# pass and returns the median (q=0.5) quantile.  Univariate: exogenous X is
# ignored (like the ETS models).  The multi-hundred-MB pipeline is loaded ONCE
# per process and cached, so cloning + the walk-forward refit across the
# 39-sector sweep reloads nothing and only the light (model_id, device, context)
# state is pickled for MLflow.  ``chronos``/``torch`` are imported lazily so
# importing this module stays light and never downloads weights.

_CHRONOS_CACHE: Dict[str, Any] = {}


def _load_chronos(model_id: str, device: str):
    """Load + cache the Chronos pipeline (one load per process; lazy import)."""
    key = f"{model_id}@{device}"
    if key not in _CHRONOS_CACHE:
        from chronos import BaseChronosPipeline
        _CHRONOS_CACHE[key] = BaseChronosPipeline.from_pretrained(model_id, device_map=device)
    return _CHRONOS_CACHE[key]


class ChronosForecaster(BaseForecaster):
    """Zero-shot univariate forecaster wrapping Amazon Chronos-Bolt.

    A pretrained foundation model — no training on our data: ``_fit`` retains the
    observed target as the forecast context, ``_predict`` runs Chronos on that
    context and returns the median (q=0.5) quantile for the requested horizon.
    Exogenous X is ignored (univariate, like the ETS models); ``param_grid`` is
    empty (nothing to tune).  The pipeline is module-cached via ``_load_chronos``
    so ``sklearn.base.clone`` and the walk-forward refit are cheap, and only
    ``model_id`` / ``device`` / the stored context are pickled — never the weights.

    Bolt is a quantile-regression model (no sampling) → the median forecast is
    deterministic, keeping the pipeline reproducible.
    """

    _tags = {
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:pred_int": False,
    }

    def __init__(self, model_id: str = "amazon/chronos-bolt-base", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        super().__init__()

    def _fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None, fh=None):
        # Zero-shot: retain the observed target as the forecast context.
        self._context = y.astype("float32")
        return self

    def _predict(self, fh, X: Optional[pd.DataFrame] = None) -> pd.Series:
        import torch

        rel = [int(step) for step in fh.to_relative(self.cutoff)]
        pipeline = _load_chronos(self.model_id, self.device)
        quantiles, _mean = pipeline.predict_quantiles(
            inputs=torch.tensor(self._context.to_numpy(dtype="float32")),
            prediction_length=max(rel),
            quantile_levels=[0.5],
        )
        median = quantiles[0, :, 0].cpu().numpy()  # [prediction_length]
        abs_index = fh.to_absolute(self.cutoff).to_pandas()
        values = [float(median[step - 1]) for step in rel]
        return pd.Series(values, index=abs_index, name=self._context.name)


# ---------------------------------------------------------------------------
# Deseasonalized feature-ML builder — remove quarterly seasonality from the
# target before the multivariate Ridge reducer (the notebook hypothesis:
# deseasonalizing makes feature-pipeline ML competitive with STL+ETS / AutoETS).
# Deseasonalizer needs a frequency-aware index, so the chain is wrapped in
# QuarterlyPeriodForecaster (PeriodIndex internally).
# ---------------------------------------------------------------------------

def _deseason_ridge():
    """Deseasonalized multivariate Ridge reducer on the selected features.

    ``Deseasonalizer(sp=4)`` strips quarterly seasonality from the target; the
    Ridge lag-window reducer then learns on the residual using the catalog
    features, and ``QuarterlyPeriodForecaster`` re-adds the frequency the
    transformer needs.
    """
    reducer = make_reduction(
        Pipeline([("scaler", StandardScaler()), ("regressor", Ridge())]),
        window_length=4, strategy="recursive",
    )
    return QuarterlyPeriodForecaster(TransformedTargetForecaster([
        ("deseason", Deseasonalizer(sp=4)),
        ("reduce", reducer),
    ]))


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
        Path to a catalog/preset JSON file (e.g. ``feature_catalog.json``).

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
# FEATURE_CATALOG — the SINGLE SOURCE OF TRUTH is the generated catalog file
# data/feature_selection/feature_catalog.json (produced by --select-features).
# There are NO hardcoded feature lists: named groups + ``all_survivors`` come
# exclusively from that file.  A fresh checkout (no catalog yet) exposes only
# ``structural_only`` until `python main.py --select-features` is run.
# ---------------------------------------------------------------------------

#: Canonical feature-catalog file inside DIR_FEATURE_SELECTION, generated by the
#: feature-selection flow (``python main.py --select-features``).  This file is
#: the single source of truth for which features each named group contains.
_FEATURE_CATALOG_FILE: str = "feature_catalog.json"


def _structural_only_catalog() -> Dict[str, FeatureGroup]:
    """Minimal catalog used ONLY when no ``feature_catalog.json`` exists yet.

    Contains just the ``structural_only`` meta-group (empty column list →
    resolves to the structural features ml_1 injects).  Deliberately holds no
    hardcoded CBS feature lists: named groups + ``all_survivors`` are produced
    exclusively by feature selection, so univariate/structural models
    (baseline / autoets / stl_ets / structural_linear) still run on a fresh
    checkout, while multivariate models referencing ``all_survivors`` require
    ``python main.py --select-features`` first.
    """
    return {
        "structural_only": FeatureGroup(
            name="structural_only", columns=[], source_table="",
            description="Empty group; resolves to structural features only via ml_1 injection.",
        )
    }


# Loaded from the canonical catalog at import; a missing/malformed file degrades
# to structural_only (never crashes the import — feature selection itself must be
# importable before the catalog exists).
_catalog_path = DIR_FEATURE_SELECTION / _FEATURE_CATALOG_FILE
FEATURE_CATALOG: Dict[str, FeatureGroup] = {}


def reload_feature_catalog() -> None:
    """(Re)populate ``FEATURE_CATALOG`` from the canonical catalog file — the
    single source of truth (``data/feature_selection/feature_catalog.json``).

    Mutates the module-level dict IN PLACE so existing imports (e.g.
    ``ml_1_data_extraction.FEATURE_CATALOG``) observe the refresh.  Called at
    import time and again by ``ml_orchestrator.run_feature_selection`` after a
    new catalog has been written, so a single process can select features and
    train without re-importing.  When the file is absent or unreadable,
    ``FEATURE_CATALOG`` falls back to ``structural_only`` ONLY — there are no
    hardcoded feature lists.
    """
    if _catalog_path.exists():
        try:
            loaded = load_feature_catalog(_catalog_path)
            FEATURE_CATALOG.clear()
            FEATURE_CATALOG.update(loaded)
            _logger.debug(
                "FEATURE_CATALOG loaded from %s (%d groups)",
                _catalog_path.name, len(FEATURE_CATALOG),
            )
            return
        except Exception as _exc:
            _logger.warning(
                "Failed to load feature catalog %s: %s — limiting FEATURE_CATALOG "
                "to 'structural_only'. Re-run `python main.py --select-features`.",
                _catalog_path.name, _exc,
            )
    else:
        _logger.warning(
            "No feature catalog at %s — FEATURE_CATALOG limited to 'structural_only'. "
            "Run `python main.py --select-features` to generate it from the gold "
            "dataset (the single source of truth for features).",
            _catalog_path,
        )
    FEATURE_CATALOG.clear()
    FEATURE_CATALOG.update(_structural_only_catalog())


reload_feature_catalog()


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
    """Curated catalog of estimator configurations for the sick-leave comparison.

    Seven models spanning the univariate-vs-multivariate question, each adding
    distinct value (redundant families + exploratory ablations were pruned;
    ``chronos_bolt`` added as a foundation-model contender):

    Baseline — ``baseline``:
        ``SectorQuarterRollingMean`` (rolling 3-year same-quarter mean per
        sector).  A leakage-free reference contender, run once per sector and
        scored under the same walk-forward MASE as every model — shown as the
        ``baseline_mase`` reference column.  (The MASE *denominator* itself is the
        in-sample seasonal-naive m=4 MAE, computed in Step 5, not this model.)

    Univariate — ``autoets`` / ``stl_ets`` / ``chronos_bolt``:
        Forecast the TARGET HISTORY ONLY — they ignore exogenous X
        (``ignores-exogeneous-X=True``), so feature selection does not affect
        them.  ``feature_groups=["structural_only"]`` passes structural X that
        the estimators drop internally.  ``autoets`` / ``stl_ets`` are
        sktime-native ETS (the notebook winners: STL+ETS MASE ≈ 0.93, AutoETS
        ≈ 1.17; STL composites are wrapped in ``QuarterlyPeriodForecaster``).
        ``chronos_bolt`` is a **zero-shot** Amazon Chronos-Bolt foundation model
        (pretrained T5; no training on our data — ``fit`` stores the context,
        ``predict`` returns the median quantile); ``param_grid={}``.

    Multivariate ML — ``ridge`` / ``random_forest`` (on ``all_survivors``):
        The models that actually LEVERAGE the carefully-selected CBS features
        (lagged target + the feature_catalog.json survivors), via sktime
        ``make_reduction`` recursive lag-window forecasters.  ``ridge`` is the
        regularised linear option (scaler pipeline → param keys
        ``estimator__regressor__alpha``); ``random_forest`` is the non-linear
        option capturing feature interactions (scale-invariant, no scaler).
        These test whether the exogenous drivers add value beyond the target's
        own past.

    Deseasonalized feature-ML — ``ridge_deseason`` (on ``all_survivors``):
        Strips quarterly seasonality from the target before the Ridge reducer
        (wrapped in ``QuarterlyPeriodForecaster``) — the feature-ML arm that beat
        the baseline for T001081.  ``ridge`` is its undeseasonalized control.

    Feature groups resolve from ``data/feature_selection/feature_catalog.json``
    (the single source of truth; ``all_survivors`` = the selected features).
    """

    _CATALOG: Dict[str, ModelExperiment] = {
        "baseline": ModelExperiment(
            name="SectorQuarterRollingMean",
            estimator=SectorQuarterRollingMean(n_years=3),
            param_grid={},
            feature_groups=None,  # uses sbi_col + quarter_col from X; no catalog groups needed
            description="Rolling 3-year same-quarter mean per SBI sector. No look-ahead bias.",
        ),
        "autoets": ModelExperiment(
            name="AutoETS_Stat",
            estimator=AutoETS(sp=4),
            # List-of-dicts grid: only VALID ETS combinations (no damped trend
            # without a trend; no additive error with multiplicative seasonal).
            # 11 combos — covers the per-sector winners of the notebook study
            # (MAdM / MNM / MNN / MNA).  Multiplicative forms require y > 0,
            # which holds for sick-leave percentages (≈ 1.4–10.1).
            param_grid=[
                {"error": ["add"], "trend": [None, "add"], "seasonal": [None, "add"]},
                {"error": ["mul"], "trend": [None], "seasonal": [None, "add", "mul"]},
                {"error": ["mul"], "trend": ["add"], "damped_trend": [False, True],
                 "seasonal": ["add", "mul"]},
            ],
            feature_groups=["structural_only"],
            description="Univariate ETS (statsmodels) — sktime-native equivalent of the notebook AutoETS winner; X ignored.",
        ),
        "stl_ets": ModelExperiment(
            name="STLETS_Stat",
            estimator=QuarterlyPeriodForecaster(STLForecaster(
                sp=4,
                forecaster_trend=AutoETS(error="add", trend="add", damped_trend=True, sp=1),
                forecaster_seasonal=NaiveForecaster(strategy="last", sp=4),
                forecaster_resid=AutoETS(error="add", sp=1),
            )),
            param_grid={
                "forecaster__seasonal": [7, 13],
                "forecaster__robust": [False, True],
                "forecaster__forecaster_trend__damped_trend": [False, True],
            },
            feature_groups=["structural_only"],
            description="STL decomposition (quarterly) + ETS trend/residual components — sktime-native STL+ETS, the notebook comparison winner.",
        ),
        "chronos_bolt": ModelExperiment(
            name="ChronosBolt_Stat",
            estimator=ChronosForecaster(model_id="amazon/chronos-bolt-base", device="cpu"),
            param_grid={},  # zero-shot foundation model — nothing to tune
            feature_groups=["structural_only"],
            description="Zero-shot Amazon Chronos-Bolt foundation model (univariate — target history only, X ignored).",
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
            feature_groups=["all_survivors"],
            description="RandomForest via make_reduction (recursive). min_samples_leaf regularises small-N fits.",
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
        # Deseasonalized multivariate feature-ML — strips quarterly seasonality
        # from the target before the Ridge reducer (the notebook finding: this
        # makes feature-pipeline ML competitive with the univariate stat models;
        # it was the feature-ML arm that beat the baseline for T001081).  `ridge`
        # above is the undeseasonalized multivariate control.
        "ridge_deseason": ModelExperiment(
            name="RidgeDeseason_Reduced",
            estimator=_deseason_ridge(),
            param_grid={
                "forecaster__reduce__window_length": [4, 8],
                "forecaster__reduce__estimator__regressor__alpha": [0.1, 1.0, 10.0, 100.0],
            },
            feature_groups=["all_survivors"],
            description="Deseasonalized (sp=4) Ridge reducer on the selected features — quarterly seasonality removed before lag-window reduction.",
        ),
        # Prophet removed: unsuitable for quarterly 4Q-ahead forecasting
        # (designed for high-frequency data; severe overfitting with many
        # regressors on ~100 rows; needs exact date-index alignment for future
        # X, incompatible with the freq=None DatetimeIndex used throughout).
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
    def get_all_keys(cls) -> List[str]:
        """Returns every catalog key — the full set of model families for a sweep."""
        return list(cls._CATALOG.keys())

    @classmethod
    def get_tuning_suite(cls) -> List[ModelExperiment]:
        """Returns only models that have a non-empty param_grid (isolated copies)."""
        return [cls.get(key) for key, exp in cls._CATALOG.items() if exp.param_grid]
