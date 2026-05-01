"""
Main Entry Point for EAISI UWV ML Pipeline.
Usage: python main.py <gold_table> <model_key> [group1,group2,...]
"""
import sys

from src.config import START_MLFLOW_UI
from src.ml_engineering.ml_orchestrator import run_pipeline
from src.utils.m_log import setup_logging, f_log
from src.utils.m_mlflow_ui import ensure_mlflow_ui

setup_logging()


def run_data_pipeline() -> None:
    """Sequentially triggers the full architectural Data Pipeline (Raw -> Bronze -> Silver -> Gold)."""
    import subprocess
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
        except subprocess.CalledProcessError:
            f_log(f"Data Pipeline execution crashed at: {script}. Aborting.", c_type="error")
            sys.exit(1)
    f_log("Data Pipeline successfully refreshed!", c_type="success")


def main() -> None:
    """Parses CLI arguments and runs the ML pipeline."""
    if "--refresh-data" in sys.argv:
        sys.argv.remove("--refresh-data")
        run_data_pipeline()

    gold_table = sys.argv[1] if len(sys.argv) > 1 else "master_data_ml_preprocessed"
    model_key = sys.argv[2] if len(sys.argv) > 2 else "linear"
    feature_groups = sys.argv[3].split(",") if len(sys.argv) > 3 else None

    f_log(
        f"Starting ML Lifecycle | Table: {gold_table} | Model: {model_key} | Groups: {feature_groups or 'ALL'}",
        c_type="start",
    )

    if START_MLFLOW_UI:
        ensure_mlflow_ui()

    try:
        run_pipeline(
            experiment_key=model_key,
            gold_table=gold_table,
            feature_groups=feature_groups,
        )
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
