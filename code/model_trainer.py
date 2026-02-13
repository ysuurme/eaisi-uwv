"""
Model Trainer for the ML Layer.
Reads feature-engineered data from Gold, trains scikit-learn models,
tunes hyperparameters, evaluates performance, and persists results.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

# --- Third Party Libraries ---
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sqlalchemy import create_engine

# --- Configuration ---
try:
    from config import DIR_DB_GOLD, DIR_MODELS, ML_TARGET_COLUMN
except ImportError:
    raise ImportError("Configuration file 'config.py' not found or missing required variables.")

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- Dataset Loader ---
class DatasetLoader:
    """
    Loads a Gold SQLite table into feature / target arrays
    and splits them into train and test sets.
    """

    def __init__(self, db_path: Path, table_name: str):
        self.db_path = Path(db_path)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.table_name = table_name

    def load_and_split(
        self,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Reads the Gold table, separates features from target,
        drops rows with missing target values, and returns a train/test split.
        """
        df = pd.read_sql_table(self.table_name, self.engine)
        logger.info(f"Loaded {len(df)} rows from '{self.table_name}'")

        # Drop rows where the target is missing
        df = df.dropna(subset=[target_column])

        # Separate features and target
        y = df[target_column]
        x = df.drop(columns=[target_column])

        # Keep only numeric columns for sklearn compatibility
        x = x.select_dtypes(include="number")

        # Fill remaining NaN feature values with column medians
        x = x.fillna(x.median())

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        logger.info(f"Train size: {len(x_train)} | Test size: {len(x_test)}")
        return x_train, x_test, y_train, y_test


# --- Hyperparameter Tuner ---
class HyperparameterTuner:
    """
    Wraps sklearn GridSearchCV for systematic hyperparameter search.
    Returns plain dicts so results can be forwarded to MLflow later.
    """

    def __init__(
        self,
        estimator,
        param_grid: dict,
        scoring: str = "neg_mean_squared_error",
        cv_folds: int = 5,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.grid_search: GridSearchCV | None = None

    def run_search(self, x_train: pd.DataFrame, y_train: pd.Series):
        """Performs cross-validated grid search and returns the best estimator."""
        self.grid_search = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.cv_folds,
            n_jobs=-1,
        )
        self.grid_search.fit(x_train, y_train)

        logger.info(f"Best params: {self.grid_search.best_params_}")
        logger.info(f"Best CV score ({self.scoring}): {self.grid_search.best_score_:.4f}")
        return self.grid_search.best_estimator_

    def get_results_summary(self) -> dict:
        """Returns a plain dict summarising the grid search results."""
        if self.grid_search is None:
            raise RuntimeError("run_search() must be called before get_results_summary()")

        return {
            "best_params": self.grid_search.best_params_,
            "best_score": self.grid_search.best_score_,
            "scoring": self.scoring,
            "cv_folds": self.cv_folds,
        }


# --- Model Trainer ---
class ModelTrainer:
    """
    Evaluates a fitted scikit-learn estimator on held-out data
    and persists the model, metrics, and config to disk.
    """

    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Computes regression metrics on the test set. Returns a plain dict."""
        predictions = self.model.predict(x_test)
        metrics = {
            "mse": mean_squared_error(y_test, predictions),
            "rmse": root_mean_squared_error(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
        }
        logger.info(f"Evaluation — MSE: {metrics['mse']:.4f} | R²: {metrics['r2']:.4f} | MAE: {metrics['mae']:.4f}")
        return metrics

    def save_run(self, identifier: str, output_dir: Path) -> Path:
        """
        Persists a training run to a timestamped folder:
          model.pkl   — serialized estimator
          metrics.json — evaluation scores
          config.json  — hyperparameters and metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(output_dir) / f"{identifier}_{self.model_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = run_dir / "model.pkl"
        joblib.dump(self.model, model_path)

        # Save config (model hyperparameters + metadata)
        config = {
            "model_name": self.model_name,
            "identifier": identifier,
            "timestamp": timestamp,
            "hyperparameters": self.model.get_params(),
        }
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4, default=str)

        logger.info(f"Run saved to {run_dir}")
        return run_dir

    def save_metrics(self, metrics: dict, run_dir: Path) -> None:
        """Writes evaluation metrics to the given run directory."""
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)


# --- Main execution ---
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor

    # 1. Load Gold data
    dataset = DatasetLoader(db_path=DIR_DB_GOLD, table_name="80072ned_gold")
    x_train, x_test, y_train, y_test = dataset.load_and_split(target_column=ML_TARGET_COLUMN)

    # 2. Hyperparameter tuning via cross-validated grid search
    param_grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
    tuner = HyperparameterTuner(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
    )
    best_model = tuner.run_search(x_train, y_train)
    tuning_summary = tuner.get_results_summary()

    # 3. Evaluate best model on held-out test set
    trainer = ModelTrainer(model=best_model, model_name="RandomForest")
    metrics = trainer.evaluate(x_test, y_test)

    # 4. Persist model, config, and metrics
    run_dir = trainer.save_run(identifier="80072ned", output_dir=DIR_MODELS)
    trainer.save_metrics(metrics, run_dir)
