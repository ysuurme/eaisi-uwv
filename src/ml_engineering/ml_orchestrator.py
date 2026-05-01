"""
ML Pipeline Orchestrator — MLOps Level 0.

Integrates all pipeline steps into a single sequential flow:
    Step 1: Data Extraction     — pull features from Gold feature store
    Step 2: Data Validation     — schema, completeness, dtype checks
    Step 3: Data Preparation    — train/test split, type casting
    Step 4: Model Training      — fit/tune estimator with MLflow tracking
    Step 5: Model Evaluation    — compute test-set metrics
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
    """Creates evaluation tables and enables WAL mode for SQLite concurrency."""
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
    features: Optional[List[str]] = None,
    threshold_r2: float = 0.2,
) -> None:
    """MLOps Level 0: Manual ML Pipeline.

    Args:
        experiment_key: Key in ModelConfiguration catalog (e.g. "linear", "random_forest").
        gold_table: Name of the preprocessed table in the gold feature store.
        features: Optional list of feature column names. None = all numeric columns.
        threshold_r2: Minimum R² score to pass the quality gate.
    """
    # --- Select Estimator ---
    config = ModelConfiguration.get(experiment_key)
    dataset_id = gold_table.replace("master_data_ml_preprocessed", "master")
    experiment_name = f"{dataset_id}_SickLeave"
    model_name = f"{config.name}_{dataset_id}"

    f_log(f"Pipeline start | Model: {model_name} | Table: {gold_table}", c_type="start")

    # --- Infrastructure ---
    _configure_mlflow(DIR_DB_EVAL)
    DIR_DB_EVAL.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{DIR_DB_EVAL.as_posix()}", connect_args={"timeout": 30})
    _ensure_eval_db(engine)

    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name, artifact_location="./mlruns")
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(log_models=False, log_datasets=False, log_input_examples=False)

    # --- Step 1: Data Extraction ---
    f_log("Step 1 | Extracting data from gold feature store...", c_type="process")
    extractor = DataExtractor(db_path=DIR_DB_GOLD, table_name=gold_table)
    raw_df = extractor.extract(target_column=ML_TARGET_COLUMN, features=features)

    # --- Step 2: Data Validation (pre-preparation) ---
    f_log("Step 2 | Validating extracted data...", c_type="process")
    DataValidator.validate(raw_df, target_column=ML_TARGET_COLUMN, stage="pre_prep")

    # --- Step 3: Data Preparation ---
    f_log("Step 3 | Preparing train/test split...", c_type="process")
    x_train, x_test, y_train, y_test, lineage = DataPreparator.prepare(
        raw_df, target_column=ML_TARGET_COLUMN,
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
                run_name=f"{config.name}_Run", lineage=lineage,
            )

            # --- Step 5: Model Evaluation ---
            f_log("Step 5 | Evaluating on test set...", c_type="process")
            evaluator = ModelEvaluator(session=session)
            metrics = evaluator.evaluate(
                run_id=run_id, fitted_model=fitted_model,
                x_test=x_test, y_test=y_test, model_name=model_name,
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
                tags={"data_source": gold_table, "status": "candidate"},
                description=f"Model {model_name} trained on {gold_table}.",
            )

            if passed:
                f_log("Pipeline completed | Model is Production", c_type="complete")
            else:
                f_log("Pipeline halted | Model did not meet quality gate", c_type="gate_fail")

        except Exception as exc:
            session.rollback()
            f_log(f"Pipeline failed. Session rolled back: {exc}", c_type="error")
            raise


if __name__ == "__main__":
    from src.utils.m_log import setup_logging
    setup_logging()

    run_pipeline(experiment_key="linear", gold_table="master_data_ml_preprocessed", threshold_r2=0.0)
