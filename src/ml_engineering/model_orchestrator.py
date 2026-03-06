"""
Model Orchestrator for the ML Layer.
Ties together:
1. Training (SQLite Tracking, CV Logging)
2. Evaluation (mlflow.models.evaluate, .pkl persistence in eval.db)
3. Governance (Alias-based promotion)
"""
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session, sessionmaker
from src.ml_engineering.model_training import DatasetLoader, ModelTrainer
from src.ml_engineering.model_evaluation import ModelEvaluator
from src.ml_engineering.model_registry import ModelRegistry

# --- Configuration ---
try:
    from config import DIR_DB_GOLD, ML_TARGET_COLUMN, DIR_DB_EVAL
except ImportError:
    raise ImportError("Configuration file 'config.py' not found.")

logger = logging.getLogger(__name__)

class ModelOrchestrator:
    """Orchestrates the full ML lifecycle with a persistent Session (Unit of Work)."""

    def __init__(self, experiment_name: str, model_name: str):
        self.experiment_name = experiment_name
        self.model_name = model_name
        
        self.trainer = ModelTrainer(experiment_name=self.experiment_name)
        self.evaluator = ModelEvaluator()
        self.registry = ModelRegistry()
        
        # Session factory
        self.Session = sessionmaker(bind=self.evaluator.engine)

    def run_experiment(
        self,
        gold_table: str,
        experiment_config: Any,
        run_name: Optional[str] = None,
        threshold_r2: float = 0.5,
        features: Optional[List[str]] = None
    ):
        """Executes the pipeline within a single persistent Session."""
        logger.info(f"🚀 Starting ML Pipeline for {self.model_name}...")

        with self.Session() as session:
            try:
                # 1. Load Data
                loader = DatasetLoader(db_path=DIR_DB_GOLD, table_name=gold_table)
                x_tr, x_te, y_tr, y_te, lineage = loader.load_and_split(
                    target_column=ML_TARGET_COLUMN,
                    features=features
                )

                # 2. Train & Tune (Active Session)
                best_model, run_id = self.trainer.train_experiment(
                    session=session,
                    experiment=experiment_config,
                    x_train=x_tr,
                    y_train=y_tr,
                    run_name=run_name or f"{experiment_config.name}_Run",
                    lineage=lineage
                )

                # 3. Evaluate Gate (Active Session)
                passed_gate = self.evaluator.evaluate_candidate(
                    session=session,
                    run_id=run_id,
                    best_model=best_model,
                    x_test=x_te,
                    y_test=y_te,
                    model_name=self.model_name,
                    threshold_r2=threshold_r2
                )

                # 4. Finalise Unit of Work
                session.commit()
                logger.info("Unit of Work committed successfully.")

                # 5. Register & Promote (Metadata-only API)
                if passed_gate:
                    version = self.registry.register_candidate(
                        run_id=run_id,
                        model_name=self.model_name,
                        tags={"data_source": gold_table, "status": "candidate"},
                        description=f"Model {self.model_name} passed gate on {gold_table}."
                    )
                    self.registry.promote_to_alias(self.model_name, version, alias="prod")
                    logger.info("🎉 Pipeline completed: Model is Production (@prod).")
                else:
                    logger.warning("🚫 Pipeline halted: Model did not meet the quality gate.")
            
            except Exception as e:
                session.rollback()
                logger.error(f"Pipeline failed. Session rolled back: {e}")
                raise e

if __name__ == "__main__":
    from src.ml_engineering.model_configs import ModelRegistry as ConfigRegistry
    logging.basicConfig(level=logging.INFO)
    
    orch = ModelOrchestrator("80072ned_SickLeave", "RF_SickLeave")
    orch.run_experiment(
        gold_table="80072ned_gold",
        experiment_config=ConfigRegistry.get("random_forest"),
        threshold_r2=0.0
    )
