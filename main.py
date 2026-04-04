"""
Main Entry Point for EAISI UWV ML Pipeline.
Usage: python main.py <gold_table> <model_key>
"""
import sys
import subprocess

# --- Local Application Imports ---
try:
    from src.config import START_MLFLOW_UI
    from src.ml_engineering.model_configs import ModelRegistry
    from src.ml_engineering.model_orchestrator import ModelOrchestrator
    from src.utils.m_mlflow_ui import ensure_mlflow_ui
    from src.utils.m_log import setup_logging, f_log
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Ensure you are running from the project root and have run 'uv pip install -e .'")
    sys.exit(1)

setup_logging()


def run_data_pipeline():
    """Sequentially triggers the full architectural Data Pipeline (Raw -> Bronze -> Silver -> Gold)."""
    f_log("Initiating Full Data Engineering Pipeline...", c_type="start")
    scripts = [
        "src/data_engineering/data_loader_raw.py",
        "src/data_engineering/data_loader_bronze.py",
        "src/data_engineering/data_loader_silver.py",
        "src/data_engineering/data_loader_gold.py"
    ]
    for script in scripts:
        f_log(f"Executing DataLoader: {script}", c_type="process")
        try:
            subprocess.run([sys.executable, script], check=True)
        except subprocess.CalledProcessError as e:
            f_log(f"Data Pipeline execution crashed violently at: {script}. Aborting process.", c_type="error")
            sys.exit(1)
    f_log("Data Pipeline successfully refreshed! All Gold Tables are natively synchronized.", c_type="success")


def run_ml_pipeline(gold_table: str, model_key: str, features: list = None):
    """Triggers the full ML lifecycle for a specific Gold table and estimator."""
    if START_MLFLOW_UI:
        ensure_mlflow_ui()

    try:
        dataset_id = gold_table.replace("_gold", "")
        config = ModelRegistry.get(model_key)

        orchestrator = ModelOrchestrator(
            experiment_name=f"{dataset_id}_SickLeave",
            model_name=f"{config.name}_{dataset_id}"
        )

        orchestrator.run_experiment(
            gold_table=gold_table,
            experiment_config=config,
            threshold_r2=0.2,
            features=features
        )
    except Exception as e:
        f_log(f"Pipeline failed for table '{gold_table}' with model '{model_key}': {e}", c_type="error")
        raise e


def main():
    # Intercept data-pipeline trigger
    if "--refresh-data" in sys.argv:
        sys.argv.remove("--refresh-data")
        run_data_pipeline()

    gold_table = sys.argv[1] if len(sys.argv) > 1 else "80072ned_gold"
    model_key = sys.argv[2] if len(sys.argv) > 2 else "random_forest"
    features = sys.argv[3].split(",") if len(sys.argv) > 3 else None

    f_log(f"Starting ML Lifecycle | Table: {gold_table} | Model: {model_key} | Features: {features or 'ALL'}", c_type="start")

    try:
        run_ml_pipeline(gold_table, model_key, features=features)
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()
