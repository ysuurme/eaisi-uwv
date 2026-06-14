"""
ML Pipeline Orchestrator — MLOps Level 0.

Integrates all pipeline steps into a single sequential flow:
    Step 1: Data Extraction     — pull features from Gold feature store
    Step 2: Data Validation     — schema, completeness, dtype checks
    Step 3: Data Preparation    — train/test split, type casting
    Step 4: Model Training      — fit/tune estimator with MLflow tracking
    Step 5: Model Evaluation    — compute test-set metrics + per-row predictions
                                  (with inner/outer fold labelling for honest CV)
    Step 6: Model Validation    — quality gate + MLflow registration

Cross-cutting entry points hosted here (the orchestrator is the hub):
    run_sector_sweep      — run_pipeline once per SBI sector
    run_feature_selection — gold panel (Step 1 extraction logic) → registry
                            frequency filter → statistical funnel →
                            feature_catalog.json (consumed by FEATURE_CATALOG)

Future: Step 7 — Model Deployment (inference service) [not yet implemented]
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from sqlalchemy import inspect as sa_inspect

from src.config import (
    CBS_TABLE_REGISTRY,
    DIR_DB_EVAL,
    DIR_DB_GOLD,
    DIR_FEATURE_SELECTION,
    FEATURE_SELECTION_FREQUENCIES,
    FEATURE_SELECTION_FUNNEL,
    ML_TARGET_COLUMN,
    PROJECT_ROOT,
    get_category_for_table,
)
from src.ml_engineering.model_configs import (
    _FEATURE_CATALOG_FILE,
    Base,
    ModelConfiguration,
    ModelForecastRecord,
    reload_feature_catalog,
)

from src.ml_engineering.ml_1_data_extraction import DataExtractor
from src.ml_engineering.ml_2_data_validation import DataValidator
from src.ml_engineering.ml_3_data_preparation import DataPreparator
from src.ml_engineering.ml_4_model_training import ModelTrainer, _base_estimator_name
from src.ml_engineering.ml_5_model_evaluation import ModelEvaluator
from src.ml_engineering.ml_6_model_validation import ModelValidator

from src.utils.feature_selection_utils import (
    apply_correlation_filter,
    apply_granger_filter,
    apply_lagged_correlation_filter,
    apply_lasso_stability_filter,
    apply_near_constant_filter,
    apply_redundancy_filter,
    save_preset_to_json,
)
from src.utils.m_log import f_log


# ---------------------------------------------------------------------------
# Infrastructure Setup
# ---------------------------------------------------------------------------

def _configure_mlflow(db_eval_path: Path) -> str:
    """Points MLflow tracking at the evaluation SQLite database."""
    relative_path = db_eval_path.relative_to(PROJECT_ROOT).as_posix()
    tracking_uri = f"sqlite:///{relative_path}?timeout=30"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def _ensure_eval_db(engine) -> None:
    """Creates the evaluation tables from the ORM and enables WAL mode.

    The eval DB is local and disposable (rebuilt from scratch), so there is no
    in-place migration: ``Base.metadata.create_all`` builds every table from
    the current ORM definitions on a fresh database.  If a stale eval DB from a
    previous schema lingers, delete ``data/4_eval/eval_data.db*`` and re-run.
    """
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.commit()
    Base.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Pipeline Entry Point
# ---------------------------------------------------------------------------

def run_pipeline(
    experiment_key: str,
    gold_table: str = "master_data_ml_preprocessed",
    feature_groups: Optional[List[str]] = None,
    sbi_filter_col: Optional[str] = None,
    n_test_points: int = 20,
    r2_floor: Optional[float] = None,
    max_mase: Optional[float] = None,
) -> None:
    """MLOps Level 0: Manual ML Pipeline.

    Args:
        experiment_key: Key in ModelConfiguration catalog (e.g. "baseline", "ridge", "random_forest").
        gold_table: Name of the preprocessed table in the gold feature store.
        feature_groups: Named feature groups from FEATURE_CATALOG. None = all columns.
        sbi_filter_col: OHE column for sector-specific mode, e.g.
            ``'BedrijfskenmerkenSBI2008_301000'``.  None = all-industry mode
            (aggregate across all sectors to one quarterly series).
        n_test_points: Number of quarterly evaluation points for walk-forward
            assessment.  Must be divisible by 4.  Default 20 = 5 origins × 4Q.
            ml_5 splits these into 2 inner + 3 outer origins for nested CV.
        r2_floor: Optional secondary R² floor for the quality gate.  None
            (default) disables it — promotion is decided purely by the MAPE
            champion/challenger gate (ADR-001).  R² is computed on OUTER folds.
    """
    # --- Select Estimator ---
    config = ModelConfiguration.get(experiment_key)
    dataset_id = gold_table.replace("master_data_ml_preprocessed", "master")
    # Derive a short sector label for run naming and MLflow tags
    sector_label = (
        sbi_filter_col.replace("BedrijfskenmerkenSBI2008_", "")
        if sbi_filter_col else "T001081"
    )
    experiment_name = f"{dataset_id}_SickLeave_4Q"
    # Eval-store / run identifier — keeps the {family}_{sector} convention that
    # model_predictions + m_pipeline_loader depend on (CONTEXT.md C3).
    model_name = f"{config.name}_{sector_label}"
    # MLflow registry key — sector-only, so all families compete for ONE
    # champion per sector (ADR-002).  Family is carried as a version tag.
    registered_model_name = f"{experiment_name}_{sector_label}"

    f_log(
        f"Pipeline start | Model: {config.name} | Sector: {sector_label} | Table: {gold_table}",
        c_type="start",
    )

    # --- Infrastructure ---
    _configure_mlflow(DIR_DB_EVAL)
    DIR_DB_EVAL.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{DIR_DB_EVAL.as_posix()}", connect_args={"timeout": 30})
    _ensure_eval_db(engine)

    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name, artifact_location="./mlruns")
    mlflow.set_experiment(experiment_name)
    # Disable autolog: sktime make_reduction wraps sklearn internals and sklearn
    # autolog would intercept those inner fits, producing duplicate/noisy run entries.
    # All params, metrics, and artifacts are logged explicitly in Steps 4 and 5.
    mlflow.sklearn.autolog(disable=True)

    # --- Step 1: Data Extraction ---
    f_log("Step 1 | Extracting data from gold feature store...", c_type="process")
    extractor = DataExtractor(db_path=DIR_DB_GOLD, table_name=gold_table)

    # Config-defined groups take precedence over CLI-passed groups
    active_groups = config.feature_groups if config.feature_groups is not None else feature_groups
    raw_df = extractor.extract(
        target_column=ML_TARGET_COLUMN,
        feature_groups=active_groups,
        sbi_filter_col=sbi_filter_col,
    )

    # --- Step 2: Data Validation (pre-preparation) ---
    f_log("Step 2 | Validating extracted data...", c_type="process")
    DataValidator.validate(raw_df, target_column=ML_TARGET_COLUMN, stage="pre_prep")

    # --- Step 3: Data Preparation ---
    f_log("Step 3 | Preparing train/test split...", c_type="process")
    x_train, x_test, y_train, y_test, lineage = DataPreparator.prepare(
        raw_df, target_column=ML_TARGET_COLUMN, n_test=n_test_points,
    )
    lineage["dataset"] = gold_table

    # --- Single DB Session for Steps 4-6 (Unit of Work) ---
    session_factory = sessionmaker(bind=engine)

    with session_factory() as session:
        try:
            # --- Step 4: Model Training ---
            f_log("Step 4 | Training model...", c_type="process")
            trainer = ModelTrainer(session=session)
            fitted_model, run_id = trainer.train(
                experiment=config, x_train=x_train, y_train=y_train,
                run_name=f"{config.name}_{sector_label}",
                lineage={**lineage, "sector": sector_label, "experiment_key": experiment_key},
            )

            # --- Step 5: Model Evaluation ---
            # sector_code is passed explicitly so per-row predictions in
            # model_predictions carry the sector identifier without relying
            # on the model_name parsing fallback.  The returned `metrics`
            # are computed on OUTER folds only (honest estimate).
            f_log("Step 5 | Evaluating on test set (nested CV)...", c_type="process")
            evaluator = ModelEvaluator(session=session)
            metrics = evaluator.evaluate(
                run_id=run_id,
                fitted_model=fitted_model,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                model_name=model_name,
                n_test_points=n_test_points,
                sector_code=sector_label,
            )

            # --- Commit training + evaluation records atomically ---
            session.commit()
            f_log("Unit of Work committed.", c_type="success")

            # --- Step 6: Model Validation & Registration ---
            f_log("Step 6 | Validating model quality...", c_type="process")
            validator = ModelValidator()
            passed = validator.validate_and_register(
                run_id=run_id,
                registered_model_name=registered_model_name,
                metrics=metrics,
                model_family=config.name,
                model_type=_base_estimator_name(config.estimator),
                feature_groups=(
                    json.dumps(active_groups) if active_groups is not None else "discovery"
                ),
                sector_code=sector_label,
                r2_floor=r2_floor,
                max_mase=max_mase,
                tags={"data_source": gold_table, "sector": sector_label, "status": "candidate"},
                description=f"{config.name} | sector={sector_label} | 4Q ahead.",
                session=session,
            )

            if passed:
                f_log("Pipeline completed | Model is Production", c_type="complete")
            else:
                f_log("Pipeline halted | Model did not meet quality gate", c_type="gate_fail")

        except Exception as exc:
            session.rollback()
            f_log(f"Pipeline failed. Session rolled back: {exc}", c_type="error")
            raise


def run_sector_sweep(
    experiment_key: str,
    gold_table: str = "master_data_ml_preprocessed",
    feature_groups: Optional[List[str]] = None,
    n_test_points: int = 20,
    r2_floor: Optional[float] = None,
    max_mase: Optional[float] = None,
) -> None:
    """Runs the pipeline for every OHE SBI sector column found in the gold table.

    Discovers all columns that match 'BedrijfskenmerkenSBI2008_*' at runtime
    and calls ``run_pipeline`` once per sector.  Results land in the same
    MLflow experiment, each run tagged with its sector label, enabling
    side-by-side comparison across all sectors in the MLflow UI.

    Args:
        experiment_key: Key in ModelConfiguration catalog.
        gold_table: Name of the preprocessed table in the gold feature store.
        feature_groups: Named feature groups from FEATURE_CATALOG.  None = all.
        r2_floor: Optional secondary R² floor per sector (None = MAPE gate only).
    """
    discovery_engine = create_engine(f"sqlite:///{DIR_DB_GOLD.as_posix()}")
    columns = sa_inspect(discovery_engine).get_columns(gold_table)
    sbi_cols = sorted(
        c["name"] for c in columns
        if c["name"].startswith("BedrijfskenmerkenSBI2008_")
    )

    f_log(
        f"Sector sweep | {len(sbi_cols)} sectors found | Model: {experiment_key}",
        c_type="start",
    )

    for i, sbi_col in enumerate(sbi_cols, 1):
        sector_label = sbi_col.replace("BedrijfskenmerkenSBI2008_", "")
        f_log(
            f"Sector {i}/{len(sbi_cols)}: {sector_label}",
            c_type="process",
        )
        try:
            run_pipeline(
                experiment_key=experiment_key,
                gold_table=gold_table,
                sbi_filter_col=sbi_col,
                feature_groups=feature_groups,
                n_test_points=n_test_points,
                r2_floor=r2_floor,
                max_mase=max_mase,
            )
        except Exception as exc:
            # Log and continue — one bad sector should not abort the sweep
            f_log(f"Sector {sector_label} failed: {exc}", c_type="error")

    f_log("Sector sweep complete.", c_type="complete")


def run_full_sweep(
    gold_table: str = "master_data_ml_preprocessed",
    model_keys: Optional[List[str]] = None,
    feature_groups: Optional[List[str]] = None,
    n_test_points: int = 20,
    r2_floor: Optional[float] = None,
    max_mase: Optional[float] = None,
) -> None:
    """Run every model family across all SBI sectors → one eval DB.

    The single "clean run" training entry point: loops ``run_sector_sweep`` over
    every ``ModelConfiguration`` key (default: all of them) so each family writes
    its walk-forward predictions/metrics to ``model_predictions`` /
    ``model_evaluations`` and competes for each sector's ``@prod`` champion.

    Pass ``model_keys`` to restrict the set — e.g. exclude the slow
    ``chronos_bolt`` (CPU forward passes of a foundation model across every
    sector × origin) for a fast iteration and run it separately overnight.
    """
    keys = model_keys or ModelConfiguration.get_all_keys()
    f_log(
        f"Full sweep | {len(keys)} model families × all sectors | {gold_table}",
        c_type="start",
    )
    for i, key in enumerate(keys, 1):
        f_log(f"Model family {i}/{len(keys)}: {key}", c_type="process")
        try:
            run_sector_sweep(
                experiment_key=key,
                gold_table=gold_table,
                feature_groups=feature_groups,
                n_test_points=n_test_points,
                r2_floor=r2_floor,
                max_mase=max_mase,
            )
        except Exception as exc:
            # One bad family should not abort the whole sweep
            f_log(f"Model family {key} failed: {exc}", c_type="error")

    f_log("Full sweep complete.", c_type="complete")


# ---------------------------------------------------------------------------
# Forecast Production Entry Point (Step 7 wiring)
# ---------------------------------------------------------------------------

def run_forecast(
    gold_table: str = "master_data_ml_preprocessed",
    n_steps: int = 4,
    render_figures: bool = True,
) -> int:
    """Forward 4Q forecast from every ``@prod`` champion → ``model_forecasts``.

    Step 7 (``ml_7_model_inference``) resolves each sector's registry champion,
    refits it on the sector's full observed history, and forecasts ``n_steps``
    quarters ahead.  This entry point wires that into the eval store and the
    figure set:

    * points MLflow at ``DIR_DB_EVAL`` and ensures the ORM tables exist
      (``model_forecasts`` is additive — no existing table is touched);
    * persists the tidy forecast frame with delete-then-insert **per
      ``sector_code``** (re-running for a subset of sectors replaces only those
      sectors' rows), mirroring the ``sector_performance`` refresh semantics;
    * logs each sector's forecast as an MLflow table artifact on its champion's
      run (``eval/forward_forecast.json``), so the forward forecast shows up in
      the run's Evaluation tab next to the backtest tables;
    * renders one best-effort ``reports/figures/forecast_<sector>.png`` overlay
      per sector (a failing sector is logged and skipped, never fatal).

    Champions must exist first — on an empty registry this logs a hint and
    writes nothing.  Returns the number of forecast rows written.
    """
    from mlflow.tracking import MlflowClient
    from src.ml_engineering import ml_7_model_inference as ml_7

    f_log("Step 7 | Forward forecast from @prod champions", c_type="start")

    _configure_mlflow(DIR_DB_EVAL)
    DIR_DB_EVAL.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{DIR_DB_EVAL.as_posix()}", connect_args={"timeout": 30}
    )
    _ensure_eval_db(engine)
    client = MlflowClient()

    try:
        forecasts = ml_7.forecast_all_champions(
            client, gold_table=gold_table, n_steps=n_steps,
        )
        if forecasts.empty:
            f_log(
                "No @prod champions found — run the training sweep first "
                "(e.g. `python main.py master_data_ml_preprocessed baseline "
                "--all-sectors`).",
                c_type="warning",
            )
            return 0
        n_written = _persist_forecasts(engine, forecasts)
    finally:
        engine.dispose()  # release the SQLite file handle (Windows-safe)

    # MLflow table artifacts on each champion's run (after the engine is closed,
    # so we don't hold two connections to the eval-store SQLite at once).
    n_logged = _log_forecast_tables(client, forecasts)

    f_log(
        f"Forecast stored | {n_written} rows across "
        f"{forecasts['sector_code'].nunique()} sectors → model_forecasts "
        f"| {n_logged} champion run(s) got a forward_forecast table",
        c_type="store",
    )

    if render_figures:
        _render_forecast_figures(forecasts, gold_table)

    f_log("Forecast production complete.", c_type="complete")
    return n_written


def _persist_forecasts(engine, forecasts) -> int:
    """Delete-then-insert the forecast frame into ``model_forecasts`` per sector.

    Replace-by-key semantics keyed on ``sector_code`` (the same projection
    pattern as ``m_sector_quality.write_sector_performance``): each sector's
    prior rows are deleted before its fresh forecast is inserted, so a partial
    run never leaves stale rows for the sectors it covered nor wipes the ones it
    did not.  The frame's ``origin_date`` maps to the table's ``forecast_made_on``.
    """
    records = forecasts.to_dict("records")
    sectors = sorted({str(r["sector_code"]) for r in records})

    with sessionmaker(bind=engine)() as session:
        for sector in sectors:
            session.query(ModelForecastRecord).filter(
                ModelForecastRecord.sector_code == sector
            ).delete()
        for r in records:
            session.add(ModelForecastRecord(
                sector_code=str(r["sector_code"]),
                model_family=r.get("model_family"),
                model_type=r.get("model_type"),
                experiment_key=r.get("experiment_key"),
                champion_version=str(r.get("champion_version", "")),
                forecast_made_on=r.get("origin_date"),
                target_date=r.get("target_date"),
                horizon=int(r["horizon"]),
                y_pred=float(r["y_pred"]),
                feature_catalog_hash=r.get("feature_catalog_hash") or "",
            ))
        session.commit()
    return len(records)


def _log_forecast_tables(client, forecasts) -> int:
    """Log each sector's forward forecast as an MLflow table on its champion's run.

    Writes ``eval/forward_forecast.json`` to the run that produced the sector's
    ``@prod`` champion, so the forecast shows in that run's Evaluation tab beside
    the backtest tables.  ``mlflow.log_table`` APPENDS, so this is made idempotent
    by skipping a run that already carries the artifact — the forecast is
    deterministic per champion run, and the eval-DB ``model_forecasts`` table is
    the always-refreshed source of truth.  Best-effort per run (never fatal).
    Returns the number of champion runs that received a fresh table.
    """
    if forecasts is None or forecasts.empty or "champion_run_id" not in forecasts.columns:
        return 0
    logged = 0
    for run_id, grp in forecasts.groupby("champion_run_id"):
        if not run_id:
            continue
        try:
            existing = {a.path for a in client.list_artifacts(run_id, "eval")}
            if "eval/forward_forecast.json" in existing:
                continue  # deterministic per champion run → don't append a duplicate
            tbl = grp.drop(columns=["champion_run_id"]).copy()
            tbl["origin_date"] = tbl["origin_date"].astype(str)
            tbl["target_date"] = tbl["target_date"].astype(str)
            with mlflow.start_run(run_id=run_id):
                mlflow.log_table(tbl, artifact_file="eval/forward_forecast.json")
            logged += 1
        except Exception as exc:
            f_log(f"Forecast table log skipped for run {run_id[:8]}: {exc}", c_type="warning")
    return logged


def _render_forecast_figures(forecasts, gold_table: str) -> None:
    """Best-effort per-sector forecast overlays — never aborts the run.

    Loads each sector's observed target history (Step 7 helper) and overlays the
    forward forecast via ``m_model_viz.plot_forecast``, saving
    ``reports/figures/forecast_<sector>.png``.  Any per-sector failure (or a
    missing matplotlib backend) is logged and skipped.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        from src.ml_engineering import ml_7_model_inference as ml_7
        from src.utils.m_model_viz import plot_forecast, save_figure
    except Exception as exc:  # pragma: no cover - import/backend env issue
        f_log(f"Forecast figures skipped (setup failed): {exc}", c_type="warning")
        return

    out_dir = PROJECT_ROOT / "reports" / "figures"
    for sector in sorted(forecasts["sector_code"].astype(str).unique()):
        try:
            history = ml_7.load_sector_target_history(gold_table, sector)
            save_figure(
                plot_forecast(history, forecasts, sector),
                out_dir / f"forecast_{sector}.png",
            )
        except Exception as exc:
            f_log(f"Forecast figure for {sector} skipped: {exc}", c_type="warning")


# ---------------------------------------------------------------------------
# Reporting Entry Point (Phase 5 — main.py --report)
# ---------------------------------------------------------------------------

def run_report(gold_table: str = "master_data_ml_preprocessed") -> Dict[str, Any]:
    """Champion-based reporting: regenerate every chart/CSV + a narrative summary.

    One command, all read from the single sources of truth (the MLflow registry
    + the eval-DB read tables):

    1. refresh the ``sector_performance`` read-model (+ ``reports/sector_quality.csv``);
    2. regenerate the standard figure set (leaderboard + predicted-vs-actual);
    3. build the model-family × feature-group experiment matrix (+ horizon curve,
       forecast overlay, champion-importance bars) and write ``experiment_matrix.csv``;
    4. write the human-readable summary to ``.claude/`` (NOT the versioned tree).

    Returns a small dict of counts/paths.  Each stage is best-effort — a missing
    dimension file or empty source degrades that stage, never aborts the run.
    """
    from src.utils import m_model_viz, m_sector_quality

    f_log("Reporting | regenerating champion-based report", c_type="start")
    _configure_mlflow(DIR_DB_EVAL)

    # 1. Refresh the sector_performance read-model + CSV (hierarchy enrichment is
    #    the most env-fragile step — degrade to MLflow-only quality if it fails).
    n_sectors = 0
    try:
        n_sectors = m_sector_quality.refresh_sector_performance(eval_db_path=DIR_DB_EVAL)
    except Exception as exc:
        f_log(f"sector_performance refresh degraded ({exc}); reporting from MLflow only.",
              c_type="warning")
    quality = m_sector_quality.load_sector_performance(DIR_DB_EVAL)
    m_sector_quality.write_report(quality, PROJECT_ROOT / "reports" / "sector_quality.csv")

    # 2. Full figure bundle (leaderboard, predicted-vs-actual, experiment matrix,
    #    horizon decay, forecast overlay, champion importances) + matrix CSV.
    figs = m_model_viz.generate_all(eval_db_path=DIR_DB_EVAL, gold_table=gold_table)

    # 3. Narrative summary → .claude (non-versioned), aggregated by m_sector_quality.
    summary_md = m_sector_quality.build_narrative_markdown(
        eval_db_path=DIR_DB_EVAL, gold_table=gold_table,
    )
    summary_path = PROJECT_ROOT / ".claude" / "week_2026-06-19_model_report.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_md, encoding="utf-8")

    f_log(
        f"Report complete | {n_sectors} sectors | {len(figs)} figures | "
        f"summary → {summary_path.relative_to(PROJECT_ROOT)}",
        c_type="complete",
    )
    return {"sectors": n_sectors, "figures": len(figs), "summary": str(summary_path)}


def run_comparison(
    eval_db_path: Optional[Path] = None,
    gold_table: str = "master_data_ml_preprocessed",
    output_dir: Optional[Path] = None,
    variant_selection: str = "per_sector_honest",
) -> Dict[str, Any]:
    """Cross-method statistical comparison from the eval DB → ``reports/comparison/``.

    Reads every model family's honest OUTER-fold predictions straight from the
    eval DB (no parquets) via ``m_pipeline_loader.load_families_from_eval_db``,
    aligns them on the shared (sector, target_date, horizon) keys, and runs the
    full ``m_evaluation`` comparison: point / regime / per-sector / per-horizon
    metrics, skill vs the baseline, prediction-interval calibration where
    available, pairwise Diebold-Mariano + Friedman/Nemenyi tests, and the
    operational + decision matrices.  Every table is written under
    ``reports/comparison/``.

    Needs ≥2 model families in ``model_predictions`` — run a sweep first
    (``python main.py --full-sweep``).  Returns a small dict of counts/paths.
    """
    from src.utils import m_evaluation, m_pipeline_loader

    f_log("Comparison | cross-method statistical comparison", c_type="start")
    eval_db_path = eval_db_path or DIR_DB_EVAL
    out_dir = output_dir or (PROJECT_ROOT / "reports" / "comparison")

    family_dfs, baseline_df, _winners = m_pipeline_loader.load_families_from_eval_db(
        eval_db_path, variant_selection=variant_selection,
    )
    if len(family_dfs) < 2:
        f_log(
            f"Comparison needs ≥2 model families in model_predictions; found "
            f"{len(family_dfs)}. Run a sweep first (e.g. python main.py --full-sweep).",
            c_type="warning",
        )
        return {"families": len(family_dfs), "output_dir": str(out_dir)}

    baseline = baseline_df if (baseline_df is not None and not baseline_df.empty) else None
    m_evaluation.compare_all_models(
        family_dfs, baseline_df=baseline, output_dir=out_dir, verbose=True,
    )

    family_names = [d["model_name"].iloc[0] for d in family_dfs]
    f_log(
        f"Comparison complete | {len(family_dfs)} families "
        f"({', '.join(family_names)}) → {out_dir.relative_to(PROJECT_ROOT)}",
        c_type="complete",
    )
    return {"families": len(family_dfs), "models": family_names, "output_dir": str(out_dir)}


# ---------------------------------------------------------------------------
# Feature Selection Entry Point
# ---------------------------------------------------------------------------

#: Prefix marking yearly-derived columns in the gold layer.
_YEARLY_PREFIX = "y_"

#: Synthetic sector column added by DataExtractor.load_full_panel().
_SECTOR_COL = "sector"

#: Quarter-end date column in the gold panel.
_DATE_COL = "period_enddate"

#: Default feature-selection holdout — the last N quarters are dropped before the
#: funnel so target-dependent filters never see the walk-forward evaluation
#: window.  Matches ml_3/ml_5's evaluation window (n_test_points=20 = 5 origins × 4Q).
_FS_HOLDOUT_QUARTERS = 20


def run_feature_selection(
    gold_table: str = "master_data_ml_preprocessed",
    funnel_params: Optional[Dict[str, Dict[str, Any]]] = None,
    holdout_last_n_quarters: int = _FS_HOLDOUT_QUARTERS,
) -> Path:
    """Derives the canonical feature catalog from the gold feature store.

    Flow (mirrors the numbered pipeline structure):
        Step 0 (leakage guard) — drop the last ``holdout_last_n_quarters``
            quarters from the panel BEFORE any target-dependent filtering, so the
            funnel (correlation / Granger / LASSO, all of which use the target)
            never sees the walk-forward evaluation window.  The panel is balanced
            (every sector shares the same quarterly calendar), so one global
            cutoff matches the per-sector test split in ml_3.  Pass ``0`` to
            disable (e.g. to regenerate a catalog over all history for diagnostics).
        Step 1 (DataExtractor) — load the full panel and derive candidate
            feature columns from the gold table (``derive_feature_columns`` is
            the single source of truth for what counts as a feature column);
        Registry filter — keep only columns originating from CBS tables whose
            frequency is in ``FEATURE_SELECTION_FREQUENCIES``; yearly-derived
            ``y_*`` columns are excluded belt-and-braces;
        Statistical funnel — sequential filters (near-constant → correlation →
            lagged correlation → Granger → LASSO stability → redundancy)
            reduce the candidates to a feature set with explanatory power;
        Catalog write — survivors are grouped by registry category and
            persisted as ``feature_catalog.json``, the file that
            ``model_configs.FEATURE_CATALOG`` loads.

    Args:
        gold_table: Name of the preprocessed table in the gold feature store.
        funnel_params: Optional per-filter overrides merged over
            ``config.FEATURE_SELECTION_FUNNEL``,
            e.g. ``{"granger": {"max_lag": 2}}``.

    Returns:
        Path to the written ``feature_catalog.json``.
    """
    f_log("Feature Selection | registry-driven statistical funnel", c_type="start")

    # --- Step 0: leakage guard — hold out the evaluation window ---
    df = DataExtractor.load_full_panel(DIR_DB_GOLD, gold_table)
    df, holdout_info = _apply_feature_selection_holdout(df, holdout_last_n_quarters)
    if holdout_info["held_out_quarters"]:
        f_log(
            f"Holdout | dropped last {holdout_info['held_out_quarters']} quarters "
            f"(cutoff {holdout_info['cutoff_date']}) before selection: "
            f"{holdout_info['rows_before']} → {holdout_info['rows_after']} rows "
            f"— funnel cannot see the evaluation window (no leakage)",
            c_type="process",
        )

    # --- Step 1: candidate columns (DataExtractor logic) ---
    all_features = DataExtractor.derive_feature_columns(df, ML_TARGET_COLUMN)

    # --- Registry filter: frequency-eligible origins only ---
    origin = _load_column_origin()
    allowed_tables = {
        tid for tid, meta in CBS_TABLE_REGISTRY.items()
        if meta["frequency"] in FEATURE_SELECTION_FREQUENCIES
        and meta["category"] != "target"
    }
    candidates, exclusions = _partition_candidates(all_features, origin, allowed_tables)
    f_log(
        f"Candidate pool | {len(candidates)} eligible of {len(all_features)} "
        f"(excluded: {exclusions['yearly_prefix']} yearly-prefixed, "
        f"{exclusions['frequency']} non-{'/'.join(FEATURE_SELECTION_FREQUENCIES)} origin, "
        f"{exclusions['unknown_origin']} unknown origin)",
        c_type="process",
    )
    if not candidates:
        raise RuntimeError(
            f"No candidate features after registry filtering. Exclusions: {exclusions}"
        )

    # --- Statistical funnel (sequential) ---
    params = {
        name: {**defaults, **(funnel_params or {}).get(name, {})}
        for name, defaults in FEATURE_SELECTION_FUNNEL.items()
    }
    chain = _apply_funnel(df, candidates, params)
    survivors = chain[-1]["retained"]
    if not survivors:
        raise RuntimeError(
            "Feature-selection funnel eliminated every candidate — refusing to "
            "write an empty catalog. Inspect FEATURE_SELECTION_FUNNEL thresholds."
        )

    # --- Group by registry category + write the canonical catalog ---
    groups, ungrouped = _group_by_registry_category(survivors, origin)
    path = save_preset_to_json(
        preset_name="feature_catalog",
        output_dir=DIR_FEATURE_SELECTION,
        survivors=survivors,
        feature_groups=groups,
        filter_chain=chain,
        input_shape=df.shape,
        description=(
            "Canonical feature catalog generated by run_feature_selection. "
            f"Candidates restricted to registry frequencies "
            f"{list(FEATURE_SELECTION_FREQUENCIES)}; survivors grouped by "
            "registry category."
        ),
        ungrouped_survivors=ungrouped,
        filename=_FEATURE_CATALOG_FILE,
        extra_metadata={
            "gold_table": gold_table,
            "target": ML_TARGET_COLUMN,
            "frequencies_included": list(FEATURE_SELECTION_FREQUENCIES),
            "feature_selection_holdout": holdout_info,
            "candidate_pool": {
                "n_candidates": len(candidates),
                "allowed_tables": sorted(allowed_tables),
                **{f"excluded_{k}": v for k, v in exclusions.items()},
            },
        },
    )

    # Refresh the in-process catalog so selection + training can run in one process.
    reload_feature_catalog()

    f_log(
        f"Feature catalog written: {path.name} | {len(candidates)} candidates → "
        f"{len(survivors)} survivors in {len(groups)} groups "
        f"({', '.join(sorted(groups))})",
        c_type="complete",
    )
    return path


def _apply_feature_selection_holdout(df, n_quarters: int) -> tuple:
    """Drop the last ``n_quarters`` unique quarters from the panel (leakage guard).

    Feature selection is target-dependent, so it must not see the quarters that
    later form the walk-forward evaluation window.  The gold panel is balanced
    (all sectors share one quarterly calendar), so a single global cutoff aligns
    with the per-sector test split that ml_3 applies at training time.

    Returns ``(train_df, holdout_info)``.  ``n_quarters <= 0`` disables the guard
    (returns the panel unchanged) — used only for all-history diagnostic catalogs.
    """
    if n_quarters <= 0:
        return df, {"held_out_quarters": 0, "cutoff_date": None,
                    "rows_before": int(len(df)), "rows_after": int(len(df))}

    dates = pd.to_datetime(df[_DATE_COL])
    unique_dates = sorted(dates.unique())
    if len(unique_dates) <= n_quarters:
        raise RuntimeError(
            f"Feature-selection holdout ({n_quarters}q) ≥ available quarters "
            f"({len(unique_dates)}). Reduce holdout_last_n_quarters."
        )
    cutoff = unique_dates[-n_quarters]
    train_df = df[dates.to_numpy() < cutoff].copy()
    info = {
        "held_out_quarters": int(n_quarters),
        "cutoff_date": str(pd.Timestamp(cutoff).date()),
        "rows_before": int(len(df)),
        "rows_after": int(len(train_df)),
    }
    return train_df, info


def _load_column_origin() -> Dict[str, str]:
    """Column → CBS table ID mapping, written by the gold loader."""
    origin_path = DIR_FEATURE_SELECTION / "column_origin.json"
    if not origin_path.exists():
        raise FileNotFoundError(
            f"column_origin.json not found at {origin_path}. It is written by "
            "the gold loader — run `python main.py --refresh-data` first."
        )
    with open(origin_path) as fh:
        return json.load(fh)


def _partition_candidates(
    feature_cols: List[str],
    origin: Dict[str, str],
    allowed_tables: set,
) -> tuple:
    """Split feature columns into funnel candidates and exclusion counts."""
    candidates: List[str] = []
    exclusions = {"yearly_prefix": 0, "frequency": 0, "unknown_origin": 0}
    unknown: List[str] = []
    for col in feature_cols:
        if col.startswith(_YEARLY_PREFIX):
            exclusions["yearly_prefix"] += 1
        elif col not in origin:
            exclusions["unknown_origin"] += 1
            unknown.append(col)
        elif origin[col] in allowed_tables:
            candidates.append(col)
        else:
            exclusions["frequency"] += 1
    if unknown:
        f_log(
            "Unknown-origin columns excluded (not in column_origin.json): "
            f"{unknown[:10]}{' …' if len(unknown) > 10 else ''}",
            c_type="warning",
        )
    return candidates, exclusions


def _apply_funnel(
    df, candidates: List[str], params: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sequential filter chain — each filter consumes the previous survivors."""
    chain: List[Dict[str, Any]] = []

    def _step(result: Dict[str, Any]) -> List[str]:
        chain.append(result)
        f_log(
            f"Filter {result['filter']:<19} | "
            f"{len(result['input']):>3} → {len(result['retained'])}",
            c_type="process",
        )
        return result["retained"]

    current = _step(apply_near_constant_filter(candidates, df, **params["near_constant"]))
    if current:
        current = _step(apply_correlation_filter(
            current, df, ML_TARGET_COLUMN,
            sector_col=_SECTOR_COL, **params["correlation"],
        ))
    if current:
        current = _step(apply_lagged_correlation_filter(
            current, df, ML_TARGET_COLUMN,
            sector_col=_SECTOR_COL, **params["lagged_correlation"],
        ))
    if current:
        current = _step(apply_granger_filter(
            current, df, ML_TARGET_COLUMN,
            sector_col=_SECTOR_COL, **params["granger"],
        ))
    if current:
        current = _step(apply_lasso_stability_filter(
            current, df, ML_TARGET_COLUMN,
            sector_col=_SECTOR_COL, **params["lasso_stability"],
        ))
    if current:
        current = _step(apply_redundancy_filter(
            current, df, ML_TARGET_COLUMN,
            sector_col=_SECTOR_COL, **params["redundancy"],
        ))
    return chain


def _group_by_registry_category(
    survivors: List[str],
    origin: Dict[str, str],
) -> tuple:
    """Group survivors by registry category — the model-facing vocabulary."""
    by_category: Dict[str, Dict[str, Any]] = {}
    ungrouped: List[str] = []
    for col in sorted(survivors):
        category = get_category_for_table(origin.get(col, ""))
        if category is None:
            ungrouped.append(col)
            continue
        entry = by_category.setdefault(category, {"columns": [], "tables": set()})
        entry["columns"].append(col)
        entry["tables"].add(origin[col])

    groups = {
        name: {
            "columns": meta["columns"],
            "source_table": ", ".join(sorted(meta["tables"])),
            "description": (
                f"Registry category '{name}': {len(meta['columns'])} surviving "
                f"features from {', '.join(sorted(meta['tables']))}."
            ),
        }
        for name, meta in sorted(by_category.items())
    }
    if ungrouped:
        f_log(
            f"{len(ungrouped)} survivors without a registry category stay "
            f"ungrouped (still reachable via all_survivors): {ungrouped}",
            c_type="warning",
        )
    return groups, ungrouped


if __name__ == "__main__":
    from src.utils.m_log import setup_logging
    setup_logging()

    # All-industry mode (default):
    run_pipeline(experiment_key="ridge", gold_table="master_data_ml_preprocessed")
    # Sector-specific mode (example):
    # run_pipeline(experiment_key="ridge", gold_table="master_data_ml_preprocessed",
    #              sbi_filter_col="BedrijfskenmerkenSBI2008_301000")
