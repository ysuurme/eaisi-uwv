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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine

# --- Configuration ---
try:
    from config import DIR_DB_GOLD, DIR_MODELS, ML_TARGET_COLUMN
except ImportError:
    raise ImportError("Configuration file 'config.py' not found or missing required variables.")

# Add or remove entries to compare different scikit-learn regressors.
MODEL_CONFIGS: list[tuple[str, object, dict]] = [
    (
        "Baseline_Mean",
        DummyRegressor(strategy="mean"),
        {},  # Naïve baseline — predicts training-set mean as a baseline
    ),
    (
        "LinearRegression",
        LinearRegression(),
        {},  # Most simple regressor that captures the linear trend
    ),
    (
        "RandomForest",
        RandomForestRegressor(random_state=42),
        {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
        },  # 6 combos × 5 folds = 30 fits
    ),
    (
        "GradientBoosting",
        GradientBoostingRegressor(random_state=42),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
        },  # 6 combos × 5 folds = 30 fits
    ),
    (
        "HistGradientBoosting",
        HistGradientBoostingRegressor(random_state=42),
        {
            "learning_rate": [0.05, 0.1, 0.2],
            "max_iter": [100, 300],
            "max_leaf_nodes": [30, 60],
        },  # 12 combos × 5 folds = 60 fits
    ),
]

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
    and splits them using TimeSeriesSplit to preserve temporal order.
    """

    def __init__(self, db_path: Path, table_name: str):
        self.db_path = Path(db_path)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.table_name = table_name

    def load_and_split(
        self,
        target_column: str,
        n_splits: int = 5,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Reads the Gold table, separates features from target,
        and returns a chronological train/test split using
        the last fold of TimeSeriesSplit.
        """
        df = pd.read_sql_table(self.table_name, self.engine)
        logger.info(f"Loaded {len(df)} rows from '{self.table_name}'")

        # Remove rows with NaN in target column
        df = df.dropna(subset=[target_column])
        logger.warning(f"Removed rows with NaN in target column, processing {len(df)} rows")

        # Sort chronologically so TimeSeriesSplit produces valid temporal folds
        df = df.sort_values("Perioden_dt").reset_index(drop=True)

        # Separate features and target
        y = df[target_column].astype(float)
        x = df.drop(columns=[target_column, "silver_id"])  # todo: drop silver_id in data_loader_gold instead
        logger.warning("Dropped silver_id from features (to be removed in data_loader_gold in future versions)")

        # Keep only numeric columns and convert pandas nullable dtypes to numpy float64
        x = x.select_dtypes(include="number").astype(float)
        logger.info(f"Selected {x.shape[1]} numeric feature columns")

        # Use the last fold so training = earliest data, test = latest data
        tscv = TimeSeriesSplit(n_splits=n_splits)
        *_, (train_index, test_index) = tscv.split(x)

        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        logger.info(f"Train size: {len(x_train)} | Test size: {len(x_test)}")
        return x_train, x_test, y_train, y_test


# --- Hyperparameter Tuner ---
class ModelHyperparameterTuner:
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
        """Performs time-series-aware cross-validated grid search."""
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        self.grid_search = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=tscv,
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


# --- Model Evaluation ---
class ModelEvaluation:
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


def run_model_training(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    configs: list[tuple[str, object, dict]],
) -> list[dict]:
    """Based on the model config, tunes, evaluates, and returns collected results."""
    results: list[dict] = []

    for name, estimator, param_grid in configs:
        logger.info(f"{'=' * 50}")
        logger.info(f"Training model: {name}")

        # Skip grid search when there are no hyperparameters to tune
        if param_grid:
            tuner = ModelHyperparameterTuner(
                estimator=estimator,
                param_grid=param_grid,
                scoring="neg_mean_squared_error",
            )
            best_model = tuner.run_search(x_train, y_train)
        else:
            best_model = estimator
            best_model.fit(x_train, y_train)

        trainer = ModelEvaluation(model=best_model, model_name=name)
        metrics = trainer.evaluate(x_test, y_test)

        run_dir = trainer.save_run(identifier="80072ned", output_dir=DIR_MODELS)
        trainer.save_metrics(metrics, run_dir)

        results.append({"name": name, **metrics})

    return results


def log_model_metrics(results: list[dict]) -> None:
    """Logs a ranked summary table so the best model is immediately visible."""
    ranked = sorted(results, key=lambda row: row["r2"], reverse=True)

    logger.info(f"\n{'=' * 70}")
    logger.info("MODEL COMPARISON (ranked by R²)")
    logger.info(f"{'Model':<25} {'R²':>8} {'MAE':>10} {'RMSE':>10} {'MSE':>12}")
    logger.info("-" * 70)
    for row in ranked:
        logger.info(
            f"{row['name']:<25} {row['r2']:>8.4f} {row['mae']:>10.4f} "
            f"{row['rmse']:>10.4f} {row['mse']:>12.4f}"
        )
    logger.info("=" * 70)


# --- Main execution ---
if __name__ == "__main__":

    # 1. Load Gold data
    dataset = DatasetLoader(db_path=DIR_DB_GOLD, table_name="80072ned_gold")
    x_train, x_test, y_train, y_test = dataset.load_and_split(target_column=ML_TARGET_COLUMN)

    # 2. Fit, evaluate, and persist each estimator in the estimator config
    results = run_model_training(
        x_train, y_train, x_test, y_test, MODEL_CONFIGS
    )

    # 3. Log ranked comparison
    log_model_metrics(results)
