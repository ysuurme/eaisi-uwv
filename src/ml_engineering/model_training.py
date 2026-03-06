"""
Model Trainer for the ML Layer.
Responsible for:
- Loading Gold data.
- Training models with MLflow Tracking (Metadata in eval_data.db).
- Capturing full Tuning Results as JSON in eval_data.db (No CSV files).
"""
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sqlalchemy import create_engine, text

from src.ml_engineering.model_configs import ModelExperiment

# --- Configuration ---
try:
    from config import DIR_DB_GOLD, ML_TARGET_COLUMN, DIR_DB_EVAL
except ImportError:
    raise ImportError("Configuration file 'config.py' not found.")

# Centralised MLflow tracking in eval_data.db
mlflow.set_tracking_uri(f"sqlite:///{DIR_DB_EVAL}")

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Loads and splits Gold SQLite data with lineage tracking and float enforcement."""
    
    def __init__(self, db_path: Path, table_name: str):
        self.db_path = db_path
        self.table_name = table_name
        self.engine = create_engine(f"sqlite:///{self.db_path}")

    def load_and_split(
        self, 
        target_column: str, 
        n_splits: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        df = pd.read_sql_table(self.table_name, self.engine)
        df = df.dropna(subset=[target_column]).sort_values("Perioden_dt").reset_index(drop=True)
        y = df[target_column].astype(float)
        
        # Select numeric features and enforce float64 for MLflow schema stability
        x = df.select_dtypes(include="number").drop(columns=[target_column], errors="ignore")
        if "silver_id" in x.columns:
            x = x.drop(columns=["silver_id"])
        x = x.astype("float64")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        *_, (train_index, test_index) = tscv.split(x)
        lineage = {"dataset": self.table_name, "target": target_column}
        return x.iloc[train_index], x.iloc[test_index], y.iloc[train_index], y.iloc[test_index], lineage

class ModelTrainer:
    """Trains estimators, logging metadata to MLflow and full tuning results to eval_data.db."""
    
    def __init__(self, experiment_name: str, db_eval_path: Path = DIR_DB_EVAL):
        self.experiment_name = experiment_name
        self.engine_eval = create_engine(f"sqlite:///{db_eval_path}")
        mlflow.set_experiment(self.experiment_name)
        # Autolog basic params and metrics, but skip models (we log them explicitly)
        mlflow.sklearn.autolog(log_models=False, log_datasets=True)
        self._init_db()

    def _init_db(self):
        """Ensures the tuning results table exists."""
        with self.engine_eval.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_tuning_results (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT,
                    cv_results_json TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()

    def _log_tuning_to_db(self, run_id: str, cv_results: dict):
        """Serialises and stores the full tuning grid in eval_data.db."""
        # Convert numpy types to native Python for JSON serialisation
        serializable_results = {
            k: v.tolist() if hasattr(v, "tolist") else v 
            for k, v in cv_results.items()
        }
        with self.engine_eval.connect() as conn:
            conn.execute(
                text("""
                    INSERT OR REPLACE INTO model_tuning_results (run_id, experiment_name, cv_results_json)
                    VALUES (:run_id, :experiment_name, :cv_results_json)
                """),
                {
                    "run_id": run_id, 
                    "experiment_name": self.experiment_name, 
                    "cv_results_json": json.dumps(serializable_results)
                }
            )
            conn.commit()

    def train_experiment(
        self, 
        experiment: ModelExperiment, 
        x_train: pd.DataFrame, 
        y_train: pd.Series, 
        run_name: str, 
        lineage: Dict
    ) -> Tuple[Any, str]:
        """Performs training and ensures no artifacts are left on the file system."""
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            mlflow.set_tags({"data_source": lineage["dataset"], "target": lineage["target"]})
            
            if experiment.param_grid:
                logger.info(f"Tuning {experiment.name}...")
                tscv = TimeSeriesSplit(n_splits=5)
                grid_search = GridSearchCV(
                    experiment.estimator, 
                    experiment.param_grid, 
                    cv=tscv, 
                    scoring="neg_mean_squared_error", 
                    n_jobs=-1
                )
                grid_search.fit(x_train, y_train)
                best_model = grid_search.best_estimator_
                
                # Store full tuning grid in DB (No CSV creation)
                self._log_tuning_to_db(run_id, grid_search.cv_results_)
            else:
                best_model = experiment.estimator
                best_model.fit(x_train, y_train)

            # Explicit Model Logging (Must-have metadata)
            signature = infer_signature(x_train, best_model.predict(x_train.head(5)))
            mlflow.sklearn.log_model(
                sk_model=best_model, 
                artifact_path="model", 
                signature=signature, 
                input_example=x_train.head(1),
                conda_env=mlflow.pyfunc.get_default_conda_env()
            )

            logger.info(f"✅ Training complete for {run_name}. Results stored in eval_data.db.")
            return best_model, run_id
