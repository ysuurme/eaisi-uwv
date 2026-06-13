"""
Main Entry Point for EAISI UWV ML Pipeline.

Usage
-----
Data engineering (raw → bronze → silver → gold):
    python main.py --refresh-data

Feature selection (gold → statistical funnel → feature_catalog.json):
    python main.py --select-features

Forward forecast (every @prod champion → model_forecasts + figures):
    python main.py --forecast

Champion-based report (refresh read-models + regenerate all figures/CSVs + summary):
    python main.py --report

All-industry mode (default):
    python main.py <gold_table> <model_key> [sbi_filter_col] [group1,group2,...]

Sector-specific mode:
    python main.py master_data_ml_preprocessed ridge BedrijfskenmerkenSBI2008_301000

Sector sweep (all SBI sectors, one run each):
    python main.py master_data_ml_preprocessed baseline --all-sectors

Model keys (ModelConfiguration catalog): baseline, autoets, stl_ets, chronos_bolt,
ridge, random_forest, ridge_deseason.

Examples:
    python main.py master_data_ml_preprocessed baseline
    python main.py master_data_ml_preprocessed autoets BedrijfskenmerkenSBI2008_301000
    python main.py master_data_ml_preprocessed random_forest - labor_structure,wages
    (use '-' as sbi_filter_col placeholder to skip it and specify feature groups)
"""
import sys

from src.config import START_MLFLOW_UI
from src.ml_engineering.ml_orchestrator import (
    run_feature_selection,
    run_forecast,
    run_pipeline,
    run_report,
    run_sector_sweep,
)
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

    if "--select-features" in sys.argv:
        sys.argv.remove("--select-features")
        gold_table = sys.argv[1] if len(sys.argv) > 1 else "master_data_ml_preprocessed"
        run_feature_selection(gold_table=gold_table)
        return

    if "--forecast" in sys.argv:
        sys.argv.remove("--forecast")
        gold_table = sys.argv[1] if len(sys.argv) > 1 else "master_data_ml_preprocessed"
        run_forecast(gold_table=gold_table)
        return

    if "--report" in sys.argv:
        sys.argv.remove("--report")
        gold_table = sys.argv[1] if len(sys.argv) > 1 else "master_data_ml_preprocessed"
        run_report(gold_table=gold_table)
        return

    gold_table    = sys.argv[1] if len(sys.argv) > 1 else "master_data_ml_preprocessed"
    model_key     = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    # arg 3: sbi_filter_col — use '-' as a placeholder to skip (means all-industry)
    sbi_raw       = sys.argv[3] if len(sys.argv) > 3 else None
    sbi_filter_col = sbi_raw if (sbi_raw and sbi_raw != "-") else None
    feature_groups = sys.argv[4].split(",") if len(sys.argv) > 4 else None

    f_log(
        f"Starting ML Lifecycle | Table: {gold_table} | Model: {model_key} | "
        f"SBI: {sbi_filter_col or 'all-industry'} | Groups: {feature_groups or 'ALL'}",
        c_type="start",
    )

    if START_MLFLOW_UI:
        ensure_mlflow_ui()

    try:
        if "--all-sectors" in sys.argv:
            run_sector_sweep(
                experiment_key=model_key,
                gold_table=gold_table,
                feature_groups=feature_groups,
            )
        else:
            run_pipeline(
                experiment_key=model_key,
                gold_table=gold_table,
                sbi_filter_col=sbi_filter_col,
                feature_groups=feature_groups,
            )
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
