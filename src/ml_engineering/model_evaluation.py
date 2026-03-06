"""
Model Evaluator for the ML Layer.
Responsible for:
- Manual evaluation logic (Zero-Artifact, no mlflow.models.evaluate).
- Persisting metrics AND model secure blobs (skops) to eval.db.
- Gatekeeper logic for model registration.
"""
import logging
import json
import skops.io as sio
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
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
    """Evaluates models and persists artifacts securely to eval.db (Zero-Artifact)."""

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
        Calculates metrics manually and persists results + secure blob to eval.db.
        Avoids mlflow.models.evaluate to prevent artifact creation.
        """
        with mlflow.start_run(run_id=run_id):
            # 1. Prediction & Manual Metrics (Zero-Artifact)
            # Use .values to prevent 'X has feature names' warnings
            y_pred = best_model.predict(x_test.values)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)

            # 2. Log Metrics to MLflow (Metadata only in DB)
            mlflow.log_metrics({
                "r2_score": r2,
                "mean_absolute_error": mae,
                "root_mean_squared_error": rmse
            })

            passed_gate = r2 >= threshold_r2
            
            # 3. Secure serialization using skops
            model_blob = sio.dumps(best_model)
            
            # 4. Persist to eval.db
            self._log_to_db(run_id, model_name, r2, mae, rmse, passed_gate, model_blob)

            logger.info(f"Gate: {'PASS' if passed_gate else 'FAIL'} (R2: {r2:.4f}). Zero artifacts created.")
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
