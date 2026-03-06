"""
Model Evaluator for the ML Layer.
Responsible for:
- Standardized benchmarks using mlflow.models.evaluate().
- Persisting metrics AND model .pkl blobs to eval.db.
- Gatekeeper logic for model registration.
"""
import logging
import pickle
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from sqlalchemy import create_engine, text

# --- Configuration ---
try:
    from config import DIR_DB_EVAL
except ImportError:
    raise ImportError("Configuration file 'config.py' not found.")

# Centralised MLflow tracking in eval_data.db
mlflow.set_tracking_uri(f"sqlite:///{DIR_DB_EVAL}")

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates models and persists artifacts to eval.db."""

    def __init__(self, db_eval_path: Path = DIR_DB_EVAL):
        self.db_eval_path = db_eval_path
        self.engine = create_engine(f"sqlite:///{self.db_eval_path}")
        self._init_db()

    def _init_db(self):
        """Ensures the evaluation table can store BLOBs for models."""
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_evaluations (
                    run_id TEXT PRIMARY KEY,
                    model_name TEXT,
                    r2 REAL,
                    mae REAL,
                    rmse REAL,
                    passed_gate INTEGER,
                    model_blob BLOB,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()

    def evaluate_candidate(
        self,
        run_id: str,
        best_model: Any,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        threshold_r2: float = 0.5
    ) -> bool:
        """
        Runs mlflow.models.evaluate and persists results + pkl to eval.db.
        """
        model_uri = f"runs:/{run_id}/model"
        eval_data = x_test.copy()
        eval_data["target"] = y_test.values

        with mlflow.start_run(run_id=run_id):
            # Using mlflow.models.evaluate for traditional ML (regression)
            result = mlflow.models.evaluate(
                model=model_uri,
                data=eval_data,
                targets="target",
                model_type="regressor",
                evaluators="default"
            )
            
            metrics = result.metrics
            r2 = metrics.get("r2_score", 0)
            mae = metrics.get("mean_absolute_error", 0)
            rmse = metrics.get("root_mean_squared_error", 0)

            passed_gate = r2 >= threshold_r2
            
            # Serialize model to bytes
            model_blob = pickle.dumps(best_model)
            
            # Persist to eval.db (including the model itself)
            self._log_to_db(run_id, model_name, r2, mae, rmse, passed_gate, model_blob)

            logger.info(f"Gate: {'PASS' if passed_gate else 'FAIL'} (R2: {r2:.4f})")
            return passed_gate

    def _log_to_db(self, run_id, model_name, r2, mae, rmse, passed_gate, model_blob):
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT OR REPLACE INTO model_evaluations 
                    (run_id, model_name, r2, mae, rmse, passed_gate, model_blob)
                    VALUES (:run_id, :model_name, :r2, :mae, :rmse, :passed_gate, :model_blob)
                """),
                {
                    "run_id": run_id, 
                    "model_name": model_name, 
                    "r2": r2, 
                    "mae": mae, 
                    "rmse": rmse, 
                    "passed_gate": int(passed_gate),
                    "model_blob": model_blob
                }
            )
            conn.commit()
