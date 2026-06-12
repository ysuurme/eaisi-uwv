"""
Step 7 — ML Model Inference (forward forecast production).

Steps 1-6 train, evaluate, and register a per-sector champion in the MLflow
registry (the ``@prod`` alias).  Nothing yet *uses* a champion to produce the
project's actual deliverable: a forward 4-quarter sick-leave forecast.  The
first six steps only ever score models on history — they never produce a value
for a quarter that has not happened.  This step closes that gap.

For one sector it:

1. Resolves the ``@prod`` champion of ``master_SickLeave_4Q_<sector>``.
2. Reads the champion's lineage from its MLflow run — the catalog key
   (``experiment_key``) and the tuned hyper-parameters (``best_params``).
3. Rebuilds the estimator from :class:`ModelConfiguration` and **refits it on
   the sector's FULL observed history**.  Champions are fitted on the train
   split only; a live forecast must be anchored at the latest observed quarter,
   so re-fitting on everything up to 2025Q3 is required for the forecast dates
   to be 2025Q4-2026Q3 rather than the train-split end.
4. Constructs production-honest future ``X`` via
   :func:`ml_5_model_evaluation.build_future_x` (deterministic structure
   extended from the dates, stochastic exogenous columns carried forward) — the
   same no-leakage rule the champion was *selected* under in Step 5.
5. Forecasts ``n_steps`` quarters ahead and returns a tidy frame of predicted
   values.

Design note
-----------
This module only *predicts values* — it returns DataFrames and does not persist
to the eval DB or render figures (those remain separate concerns).  It adds no
new tables and changes none of the existing pipeline files: it reuses the
established helpers so there is no duplicated logic —
:func:`ml_5_model_evaluation.build_future_x` (the Step 5/7 shared X-builder),
:func:`ml_5_model_evaluation._predict_origin` (the fit+predict estimator
dispatch), :func:`ml_3_data_preparation._extract_feature_matrix`, the
:class:`DataExtractor` (Step 1), and the :class:`ModelConfiguration` catalog.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import DIR_DB_EVAL, DIR_DB_GOLD, ML_TARGET_COLUMN
from src.ml_engineering.ml_1_data_extraction import DataExtractor
from src.ml_engineering.ml_3_data_preparation import DATE_COL, _extract_feature_matrix
from src.ml_engineering.ml_5_model_evaluation import (
    _predict_origin,
    _to_timestamp,
    build_future_x,
)
from src.ml_engineering.model_configs import ModelConfiguration, ModelExperiment
from src.utils.m_log import f_log

#: Forecast horizon — matches the project's 4-quarter-ahead objective.
_FH_STEPS = 4

#: MLflow alias that marks a sector's champion (mirrors ml_6 / ADR-002).
_PROD_ALIAS = "prod"

#: OHE column prefix used in the gold store for SBI sector indicators.
_OHE_PREFIX = "BedrijfskenmerkenSBI2008_"

#: Sector token of the CBS national total (all-industry mode → no SBI filter).
_NATIONAL_TOTAL = "T001081"


@dataclass(frozen=True)
class ChampionLineage:
    """The provenance needed to rebuild and refit a sector's champion."""
    sector_code: str
    version: str
    run_id: str
    experiment_key: str
    model_family: str
    model_type: str
    feature_groups: str = ""
    best_params: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure helpers (MLflow-free, independently testable)
# ---------------------------------------------------------------------------

def _sector_to_sbi_filter(sector_label: str) -> Optional[str]:
    """Map a sector token to the gold OHE filter column.

    ``T001081`` (national total) → ``None`` (all-industry mode, mirroring
    Step 1); any other sector → its ``BedrijfskenmerkenSBI2008_<sector>`` OHE
    column.
    """
    if sector_label == _NATIONAL_TOTAL:
        return None
    return f"{_OHE_PREFIX}{sector_label}"


def _experiment_prefix(gold_table: str) -> str:
    """Registry-name prefix for the gold table (mirrors ``run_pipeline``).

    ``master_data_ml_preprocessed`` → ``master_SickLeave_4Q_`` so that
    registered models ``master_SickLeave_4Q_<sector>`` can be enumerated.
    """
    dataset_id = gold_table.replace("master_data_ml_preprocessed", "master")
    return f"{dataset_id}_SickLeave_4Q_"


def _coerce_param(value: Any) -> Any:
    """Best-effort coerce a JSON-stringified hyper-parameter back to its type.

    ``best_params`` is logged with ``json.dumps(..., default=str)``, so numpy
    scalars survive as strings (e.g. ``"8"``, ``"1.0"``).  Plain ints/floats,
    booleans, and ``None`` round-trip natively; legitimate string params (ETS
    ``"add"``/``"mul"``) are left untouched.
    """
    if not isinstance(value, str):
        return value
    special = {"True": True, "False": False, "None": None}
    if value in special:
        return special[value]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _coerce_best_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply :func:`_coerce_param` to every value of a ``best_params`` dict."""
    return {key: _coerce_param(val) for key, val in params.items()}


def _rebuild_estimator(
    experiment_key: str,
    best_params: Dict[str, Any],
) -> Tuple[Any, ModelExperiment]:
    """Rebuild the champion estimator from the catalog + tuned params.

    Returns ``(estimator, config)`` where ``estimator`` is an unfitted clone of
    ``ModelConfiguration.get(experiment_key).estimator`` with the (coerced)
    ``best_params`` applied.  ``config`` carries the matching
    ``feature_groups`` so the inference feature matrix is reconstructed exactly
    as the champion was trained.
    """
    config = ModelConfiguration.get(experiment_key)
    estimator = config.estimator
    if best_params:
        estimator.set_params(**_coerce_best_params(best_params))
    return estimator, config


def _forecast_from_history(
    estimator: Any,
    x_hist: pd.DataFrame,
    y_hist: pd.Series,
    *,
    sector_code: str,
    model_family: str,
    model_type: str,
    experiment_key: str,
    champion_version: str,
    n_steps: int = _FH_STEPS,
) -> pd.DataFrame:
    """Refit ``estimator`` on full history and forecast ``n_steps`` quarters.

    Builds production-honest future ``X`` with :func:`build_future_x` and
    forecasts via the shared Step-5 dispatch :func:`_predict_origin` (which
    handles both the sklearn baseline and sktime forecaster APIs).  Returns one
    tidy row per forecast quarter.
    """
    x_future = build_future_x(x_hist, n_steps=n_steps)
    y_pred = np.asarray(_predict_origin(estimator, y_hist, x_hist, x_future)).ravel()

    origin = _to_timestamp(x_hist.index[-1])
    rows: List[Dict[str, Any]] = []
    for h in range(min(n_steps, len(x_future), len(y_pred))):
        rows.append({
            "sector_code":      sector_code,
            "model_family":     model_family,
            "model_type":       model_type,
            "experiment_key":   experiment_key,
            "champion_version": champion_version,
            "origin_date":      origin,
            "target_date":      _to_timestamp(x_future.index[h]),
            "horizon":          int(h + 1),
            "y_pred":           float(y_pred[h]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MLflow / gold-store integration
# ---------------------------------------------------------------------------

def _read_champion_lineage(client, registered_model_name: str, sector_code: str) -> ChampionLineage:
    """Read the ``@prod`` champion's rebuild lineage from MLflow.

    The catalog key + tuned params come from the producing run's params
    (logged by Step 4); the model family/type come from the version tags
    (stamped by Step 6), falling back to run params when absent.
    """
    mv = client.get_model_version_by_alias(registered_model_name, _PROD_ALIAS)
    run = client.get_run(mv.run_id)
    params = dict(run.data.params or {})
    tags = dict(getattr(mv, "tags", None) or {})

    try:
        best_params = json.loads(params.get("best_params", "{}")) or {}
    except (TypeError, ValueError):
        best_params = {}

    return ChampionLineage(
        sector_code=sector_code,
        version=str(mv.version),
        run_id=mv.run_id,
        experiment_key=params.get("experiment_key", ""),
        model_family=tags.get("model_family") or params.get("model_name", ""),
        model_type=tags.get("model_type") or params.get("base_estimator_class", ""),
        feature_groups=tags.get("feature_groups") or params.get("feature_groups", ""),
        best_params=best_params if isinstance(best_params, dict) else {},
    )


def _load_sector_history(
    gold_table: str,
    sbi_filter_col: Optional[str],
    feature_groups: Optional[List[str]],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a sector's FULL observed history as (X, y), quarterly DatetimeIndex.

    Mirrors Step 1 extraction + Step 3 feature-matrix construction, but keeps
    every row (no train/test split) so the estimator is refit on all data up to
    the latest observed quarter.
    """
    extractor = DataExtractor(db_path=DIR_DB_GOLD, table_name=gold_table)
    df = extractor.extract(
        target_column=ML_TARGET_COLUMN,
        feature_groups=feature_groups,
        sbi_filter_col=sbi_filter_col,
    )
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df.set_index(pd.DatetimeIndex(df[DATE_COL], freq=None))

    y = df[ML_TARGET_COLUMN].astype(float)
    x = _extract_feature_matrix(df, ML_TARGET_COLUMN)
    return x, y


def forecast_sector(
    client,
    gold_table: str,
    sector_label: str,
    n_steps: int = _FH_STEPS,
) -> pd.DataFrame:
    """Produce the forward forecast for one sector's ``@prod`` champion.

    Args:
        client: An ``mlflow.tracking.MlflowClient`` pointed at the eval DB.
        gold_table: Gold feature-store table name.
        sector_label: Sector token, e.g. ``"T001081"`` or ``"301000"``.
        n_steps: Forecast horizon in quarters (default 4).

    Returns:
        Tidy DataFrame, one row per forecast quarter (see
        :func:`_forecast_from_history`).
    """
    registered_model_name = f"{_experiment_prefix(gold_table)}{sector_label}"
    lineage = _read_champion_lineage(client, registered_model_name, sector_label)

    # Rebuild from the catalog; degrade to default params if the tuned params
    # cannot be applied, so a forecast is still produced (logged for audit).
    try:
        estimator, config = _rebuild_estimator(lineage.experiment_key, lineage.best_params)
    except Exception as exc:
        f_log(
            f"Inference | {sector_label}: tuned-param rebuild failed ({exc}); "
            f"falling back to catalog defaults for '{lineage.experiment_key}'.",
            c_type="warning",
        )
        estimator, config = _rebuild_estimator(lineage.experiment_key, {})

    sbi_filter_col = _sector_to_sbi_filter(sector_label)
    x_hist, y_hist = _load_sector_history(gold_table, sbi_filter_col, config.feature_groups)

    frame = _forecast_from_history(
        estimator, x_hist, y_hist,
        sector_code=sector_label,
        model_family=lineage.model_family or config.name,
        model_type=lineage.model_type,
        experiment_key=lineage.experiment_key,
        champion_version=lineage.version,
        n_steps=n_steps,
    )
    f_log(
        f"Forecast | {sector_label} | {lineage.model_family or config.name} "
        f"v{lineage.version} | {len(frame)}Q ahead "
        f"({frame['target_date'].min().date()}…{frame['target_date'].max().date()})",
        c_type="success",
    )
    return frame


def forecast_all_champions(
    client,
    gold_table: str = "master_data_ml_preprocessed",
    n_steps: int = _FH_STEPS,
) -> pd.DataFrame:
    """Forecast every registered sector that has a ``@prod`` champion.

    Enumerates ``master_SickLeave_4Q_<sector>`` registered models; sectors
    without a ``@prod`` alias (or that fail to forecast) are logged and skipped
    so one bad sector never aborts the run.  Returns the concatenated tidy
    forecast frame across all sectors (empty if none are registered).
    """
    prefix = _experiment_prefix(gold_table)
    frames: List[pd.DataFrame] = []
    n_registered = 0

    for rm in client.search_registered_models():
        if not rm.name.startswith(prefix):
            continue
        n_registered += 1
        sector_label = rm.name[len(prefix):]
        try:
            frames.append(forecast_sector(client, gold_table, sector_label, n_steps=n_steps))
        except Exception as exc:
            f_log(f"Forecast | {sector_label} skipped: {exc}", c_type="warning")

    f_log(
        f"Inference complete | {len(frames)}/{n_registered} sectors forecast",
        c_type="complete",
    )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_inference(
    gold_table: str = "master_data_ml_preprocessed",
    n_steps: int = _FH_STEPS,
) -> pd.DataFrame:
    """Entry point: point MLflow at the eval DB and forecast all champions.

    Returns the tidy forward-forecast frame.  Persistence and plotting are
    intentionally left to separate steps — this returns predicted values only.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(f"sqlite:///{DIR_DB_EVAL.as_posix()}?timeout=30")
    client = MlflowClient()
    f_log(f"Step 7 | Forward inference from @prod champions | table={gold_table}",
          c_type="start")
    return forecast_all_champions(client, gold_table=gold_table, n_steps=n_steps)


if __name__ == "__main__":
    from src.utils.m_log import setup_logging
    setup_logging()
    result = run_inference()
    if result.empty:
        print("No @prod champions found — run the training sweep first.")
    else:
        print(result.to_string(index=False))
