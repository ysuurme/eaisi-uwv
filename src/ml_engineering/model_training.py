"""
Model Trainer for the ML Layer.
Responsible for:
- Loading Gold data.
- Training scikit-learn models with MLflow SQLite Tracking (No mlruns).
- Capturing CV results and high-value metadata.
"""
import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sqlalchemy import create_engine

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
    """Loads and splits Gold SQLite data with lineage tracking."""
    
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
        x = df.select_dtypes(include="number").drop(columns=[target_column], errors="ignore")
        if "silver_id" in x.columns:
            x = x.drop(columns=["silver_id"])
        tscv = TimeSeriesSplit(n_splits=n_splits)
        *_, (train_index, test_index) = tscv.split(x)
        lineage = {"dataset": self.table_name, "target": target_column}
        return x.iloc[train_index], x.iloc[test_index], y.iloc[train_index], y.iloc[test_index], lineage

class ModelTrainer:
    """Trains estimators with explicit CV result logging in MLflow."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)
        mlflow.sklearn.autolog(log_models=False, log_datasets=True)

    def train_experiment(self, experiment: ModelExperiment, x_train: pd.DataFrame, y_train: pd.Series, run_name: str, lineage: Dict) -> Tuple[Any, str]:
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({"data_source": lineage["dataset"], "target": lineage["target"]})
            
            if experiment.param_grid:
                tscv = TimeSeriesSplit(n_splits=5)
                grid_search = GridSearchCV(experiment.estimator, experiment.param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
                grid_search.fit(x_train, y_train)
                best_model = grid_search.best_estimator_
                
                # Log CV results as artifact following tuning
                cv_results = pd.DataFrame(grid_search.cv_results_)
                cv_results.to_csv("cv_results.csv", index=False)
                mlflow.log_artifact("cv_results.csv", artifact_path="tuning")
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

            logger.info(f"✅ Training complete for {run_name}")
            return best_model, run.info.run_id
