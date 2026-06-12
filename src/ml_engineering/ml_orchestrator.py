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

Future: Step 7 — Model Deployment (inference service) [not yet implemented]
"""
import os
from pathlib import Path
from typing import List, Optional

import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from sqlalchemy import inspect as sa_inspect

from src.config import DIR_DB_GOLD, DIR_DB_EVAL, ML_TARGET_COLUMN, PROJECT_ROOT
from src.ml_engineering.model_configs import Base, ModelConfiguration

from src.ml_engineering.ml_1_data_extraction import DataExtractor
from src.ml_engineering.ml_2_data_validation import DataValidator
from src.ml_engineering.ml_3_data_preparation import DataPreparator
from src.ml_engineering.ml_4_model_training import ModelTrainer
from src.ml_engineering.ml_5_model_evaluation import ModelEvaluator
from src.ml_engineering.ml_6_model_validation import ModelValidator

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
    """Creates evaluation tables, enables WAL mode, and validates schema.

    On first run the per-row prediction table is created from scratch.

    SCHEMA CHECK (fail-fast):
        If ``model_predictions`` already exists but is missing the ``fold_set``
        column required by the nested-CV evaluator (ml_5_model_evaluation.py),
        this function raises ``RuntimeError`` with explicit migration SQL.
        ``Base.metadata.create_all`` is a no-op for existing tables — it does
        NOT add columns that were introduced after the table's first creation.
        So we must check explicitly.
    """
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.commit()
    Base.metadata.create_all(engine)

    # ── Schema check: enforce fold_set column on model_predictions ──────
    insp = sa_inspect(engine)
    if "model_predictions" in insp.get_table_names():
        existing_cols = {c["name"] for c in insp.get_columns("model_predictions")}
        if "fold_set" not in existing_cols:
            db_path = engine.url.database  # e.g. "data/4_eval/eval_data.db"
            raise RuntimeError(
                "\n"
                "═════════════════════════════════════════════════════════════════════\n"
                "  SCHEMA MISMATCH — model_predictions lacks 'fold_set' column\n"
                "═════════════════════════════════════════════════════════════════════\n"
                "  The updated ml_5_model_evaluation.py writes inner/outer fold labels\n"
                "  via a 'fold_set' column.  Your existing model_predictions table was\n"
                "  created before this column existed and SQLAlchemy's create_all() is\n"
                "  a no-op for existing tables.\n"
                "\n"
                "  Apply ONE of these migrations and re-run:\n"
                "\n"
                "  (A) DROP + recreate (simplest; deletes old predictions):\n"
                f"        sqlite3 {db_path}\n"
                "        DROP TABLE model_predictions;\n"
                "\n"
                "  (B) ALTER TABLE (preserves old rows, marks them as 'outer'):\n"
                f"        sqlite3 {db_path}\n"
                "        ALTER TABLE model_predictions\n"
                "        ADD COLUMN fold_set VARCHAR(8) NOT NULL DEFAULT 'outer';\n"
                "\n"
                "  Note: under option (B), old rows will NOT have inner-fold predictions,\n"
                "  so the loader's per_sector_honest mode will treat them as ineligible\n"
                "  for variant selection (correct behaviour — only re-runs with the new\n"
                "  ml_5 will produce eligible inner-fold data).\n"
                "═════════════════════════════════════════════════════════════════════\n"
            )


# ---------------------------------------------------------------------------
# Pipeline Entry Point
# ---------------------------------------------------------------------------

def run_pipeline(
    experiment_key: str,
    gold_table: str = "master_data_ml_preprocessed",
    feature_groups: Optional[List[str]] = None,
    sbi_filter_col: Optional[str] = None,
    n_test_points: int = 20,
    threshold_r2: float = 0.2,
) -> None:
    """MLOps Level 0: Manual ML Pipeline.

    Args:
        experiment_key: Key in ModelConfiguration catalog (e.g. "linear", "random_forest").
        gold_table: Name of the preprocessed table in the gold feature store.
        feature_groups: Named feature groups from FEATURE_CATALOG. None = all columns.
        sbi_filter_col: OHE column for sector-specific mode, e.g.
            ``'BedrijfskenmerkenSBI2008_301000'``.  None = all-industry mode
            (aggregate across all sectors to one quarterly series).
        n_test_points: Number of quarterly evaluation points for walk-forward
            assessment.  Must be divisible by 4.  Default 20 = 5 origins × 4Q.
            ml_5 splits these into 2 inner + 3 outer origins for nested CV.
        threshold_r2: Minimum R² score to pass the quality gate.  Note: with
            the inner/outer split, R² is now computed on OUTER folds only.
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
    model_name = f"{config.name}_{sector_label}"

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
                lineage={**lineage, "sector": sector_label},
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
                run_id=run_id, model_name=model_name, metrics=metrics,
                threshold_r2=threshold_r2,
                tags={"data_source": gold_table, "sector": sector_label, "status": "candidate"},
                description=f"Model {model_name} | sector={sector_label} | 4Q ahead.",
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
    threshold_r2: float = 0.2,
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
        threshold_r2: Minimum R² to pass the quality gate per sector.
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
                threshold_r2=threshold_r2,
            )
        except Exception as exc:
            # Log and continue — one bad sector should not abort the sweep
            f_log(f"Sector {sector_label} failed: {exc}", c_type="error")

    f_log("Sector sweep complete.", c_type="complete")


if __name__ == "__main__":
    from src.utils.m_log import setup_logging
    setup_logging()

    # All-industry mode (default):
    run_pipeline(experiment_key="linear", gold_table="master_data_ml_preprocessed", threshold_r2=0.0)
    # Sector-specific mode (example):
    # run_pipeline(experiment_key="linear", gold_table="master_data_ml_preprocessed",
    #              sbi_filter_col="BedrijfskenmerkenSBI2008_301000", threshold_r2=0.0)
