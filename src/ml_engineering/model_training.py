"""
Model Trainer for the ML Layer.
Responsible for:
- Loading Gold data.
- Training models with PURE DB Tracking (No artifacts, no mlruns).
- Capturing signatures and environment as JSON Tags in MLflow SQLite.
"""
import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sqlalchemy import create_engine, text, String, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.sql import func

from src.ml_engineering.model_configs import ModelExperiment

# --- Configuration ---
try:
    from config import DIR_DB_GOLD, ML_TARGET_COLUMN, DIR_DB_EVAL
except ImportError:
    raise ImportError("Configuration file 'config.py' not found.")

# Global enforcement of SQLite tracking
db_uri = f"sqlite:///{DIR_DB_EVAL}?timeout=30"
os.environ["MLFLOW_TRACKING_URI"] = db_uri
mlflow.set_tracking_uri(db_uri)

logger = logging.getLogger(__name__)

# --- ORM Model Definitions ---
class Base(DeclarativeBase):
    pass

class ModelTuningRecord(Base):
    """ORM representation of hyperparameter tuning results."""
    __tablename__ = "model_tuning_results"
    
    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    experiment_name: Mapped[str] = mapped_column(String)
    cv_results_json: Mapped[str] = mapped_column(String)
    timestamp: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.now())

class DatasetLoader:
    """Loads and splits Gold SQLite data with lineage tracking and float enforcement."""
    
    def __init__(self, db_path: Path, table_name: str):
        self.db_path = db_path
        self.table_name = table_name
        self.engine = create_engine(f"sqlite:///{self.db_path}")

    def load_and_split(
        self, 
        target_column: str, 
        features: Optional[List[str]] = None,
        n_splits: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        df = pd.read_sql_table(self.table_name, self.engine)
        df = df.dropna(subset=[target_column]).sort_values("Perioden_dt").reset_index(drop=True)
        y = df[target_column].astype(float)
        
        if features:
            x = df[features].copy()
        else:
            x = df.select_dtypes(include="number").drop(columns=[target_column], errors="ignore")
            if "silver_id" in x.columns:
                x = x.drop(columns=["silver_id"])
        
        x.columns = [str(col) for col in x.columns]
        x = x.astype("float64")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        *_, (train_index, test_index) = tscv.split(x)
        lineage = {"dataset": self.table_name, "target": target_column, "feature_count": x.shape[1]}
        return x.iloc[train_index], x.iloc[test_index], y.iloc[train_index], y.iloc[test_index], lineage

class ModelTrainer:
    """Trains estimators with Zero-Artifact logging (Metadata only in DB)."""
    
    def __init__(self, experiment_name: str, db_eval_path: Path = DIR_DB_EVAL, engine: Optional[Any] = None):
        self.experiment_name = experiment_name
        self.db_eval_path = db_eval_path
        # Use provided engine or create a new high-concurrency one
        self.engine_eval = engine or create_engine(
            f"sqlite:///{self.db_eval_path}",
            connect_args={"timeout": 30}
        )
        
        mlflow.set_tracking_uri(f"sqlite:///{self.db_eval_path}")
        # Ensure experiment exists in DB
        if not mlflow.get_experiment_by_name(self.experiment_name):
            mlflow.create_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)
        
        # Autolog params and metrics to DB, but strictly no artifacts
        mlflow.sklearn.autolog(log_models=False, log_datasets=False, log_input_examples=False)
        self._init_db()

    def _init_db(self):
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

    def _log_tuning_to_db(self, session: Session, run_id: str, cv_results: dict):
        """Serialises and stores the full tuning grid in eval_data.db via ORM."""
        serializable_results = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in cv_results.items()}
        record = ModelTuningRecord(
            run_id=run_id,
            experiment_name=self.experiment_name,
            cv_results_json=json.dumps(serializable_results)
        )
        session.merge(record)
        session.commit() # Commit immediately to release SQLite write lock for MLflow

    def train_experiment(
        self, 
        session: Session,
        experiment: ModelExperiment, 
        x_train: pd.DataFrame, 
        y_train: pd.Series, 
        run_name: str, 
        lineage: Dict
    ) -> Tuple[Any, str]:
        """Performs training within an active ORM Session."""
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            mlflow.set_tags({"data_source": lineage["dataset"], "target": lineage["target"]})
            
            if experiment.param_grid:
                logger.info(f"Tuning {experiment.name}...")
                tscv = TimeSeriesSplit(n_splits=5)
                grid_search = GridSearchCV(experiment.estimator, experiment.param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
                grid_search.fit(x_train, y_train)
                best_model = grid_search.best_estimator_
                self._log_tuning_to_db(session, run_id, grid_search.cv_results_)
            else:
                best_model = experiment.estimator
                best_model.fit(x_train, y_train)

            # --- RESTORE UI VISIBILITY ---
            # Explicit Model Logging (Enables Model Registry and UI visibility)
            signature = infer_signature(x_train, best_model.predict(x_train.head(5)))
            
            # Manual environment to bypass MLflow's failing pip discovery (solved via UV)
            conda_env = {
                "name": "eaisi-uwv-env",
                "channels": ["conda-forge"],
                "dependencies": [
                    "python=3.10.11",
                    {
                        "pip": [
                            "mlflow>=3.10.1",
                            "scikit-learn>=1.6.0",
                            "pandas>=2.3.3",
                            "sqlalchemy>=2.0.45",
                            "skops>=0.13.0"
                        ]
                    },
                ],
            }

            # We use skops for the internal MLflow log as well to match our DB storage
            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="model",
                signature=signature,
                input_example=x_train.head(1),
                conda_env=conda_env,
                serialization_format="skops"
            )

            logger.info(f"✅ Training complete for {run_name}. Model logged to MLflow UI.")
            return best_model, run_id
