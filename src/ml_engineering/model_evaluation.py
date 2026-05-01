"""
Model Evaluator for the ML Layer.
Uses SQLAlchemy ORM and Persistent Sessions to resolve e3q8 (DetachedInstanceError).
"""
import skops.io as sio
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sqlalchemy import create_engine, text, String, Float, LargeBinary, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, sessionmaker
from sqlalchemy.sql import func

# --- Configuration ---
try:
    from src.config import DIR_DB_EVAL, PROJECT_ROOT
except ImportError:
    raise ImportError("Configuration file 'src/config.py' not found.")

# --- Logging ---
from src.utils.m_log import f_log

# MLflow tracking configuration is injected dynamically to avoid side effects.


# --- ORM Model Definitions ---
class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass

class ModelEvaluationRecord(Base):
    """ORM representation of a model evaluation result."""
    __tablename__ = "model_evaluations"
    
    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_name: Mapped[Optional[str]] = mapped_column(String)
    r2: Mapped[Optional[float]] = mapped_column(Float)
    mae: Mapped[Optional[float]] = mapped_column(Float)
    rmse: Mapped[Optional[float]] = mapped_column(Float)
    passed_gate: Mapped[Optional[int]] = mapped_column(Float)
    model_blob: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    timestamp: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.now())

class ModelEvaluator:
    """Evaluates models using ORM Sessions with high-concurrency SQLite settings."""

    def __init__(self, db_eval_path: Path = DIR_DB_EVAL):
        self.db_eval_path = db_eval_path
        
        # Ensure the evaluation data directory exists so SQLite doesn't crash structurally
        if isinstance(self.db_eval_path, Path):
            self.db_eval_path.parent.mkdir(parents=True, exist_ok=True)
            
        self.engine = create_engine(
            f"sqlite:///{self.db_eval_path.as_posix()}",
            connect_args={"timeout": 30} 
        )
        
        # Configure MLflow tracking for evaluation
        rel_db_eval = self.db_eval_path.relative_to(PROJECT_ROOT).as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{rel_db_eval}?timeout=30")
        
        self._init_db()

    def _init_db(self):
        """Ensures the evaluation table exists and WAL mode is enabled."""
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.commit()
            
        Base.metadata.create_all(self.engine)
        # Migration check for model_blob (in case table was created with Core previously)
        with self.engine.connect() as conn:
            try:
                conn.execute(text("ALTER TABLE model_evaluations ADD COLUMN model_blob BLOB"))
                conn.commit()
            except Exception:
                pass 

    def evaluate_candidate(
        self,
        session: Session,
        run_id: str,
        best_model: Any,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        threshold_r2: float = 0.5
    ) -> bool:
        """
        Calculates metrics and persists to DB using an active ORM Session.
        The session is NOT closed here to allow lazy loading by the caller.
        """
        with mlflow.start_run(run_id=run_id):
            y_pred = best_model.predict(x_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)

            mlflow.log_metrics({
                "r2_score": r2,
                "mean_absolute_error": mae,
                "root_mean_squared_error": rmse
            })

            passed_gate = r2 >= threshold_r2
            model_blob = sio.dumps(best_model)
            
            record = ModelEvaluationRecord(
                run_id=run_id,
                model_name=model_name,
                r2=r2,
                mae=mae,
                rmse=rmse,
                passed_gate=int(passed_gate),
                model_blob=model_blob
            )
            
            session.merge(record)
            session.commit()

            gate_result = "PASS" if passed_gate else "FAIL"
            f_log(f"Gate: {gate_result} | R2: {r2:.4f}", c_type="success" if passed_gate else "gate_fail")
            f_log(f"Model stored | eval_data.db", c_type="store")
            return passed_gate
