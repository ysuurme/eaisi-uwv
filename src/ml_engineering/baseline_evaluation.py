"""
Baseline Evaluation — MLflow Integration
Reads pre-computed baseline predictions from the Gold DB and logs
metrics to MLflow as a named reference run for model comparison.

Assumptions:
- Baseline predictions are already stored in the Gold DB by the
  baseline marimo notebook (prediction_baseline_total,
  prediction_baseline_sbi, prediction_baseline_compsize).
- This module only reads and logs; it never re-computes or re-exports.
"""
import logging

import mlflow
import polars as pl

try:
    from src.config import DIR_DB_GOLD
except ImportError:
    raise ImportError("Configuration file 'config.py' not found.")

from src.utils.m_query_database import f_query_database

logger = logging.getLogger(__name__)

# Gold tables written by the baseline notebook
_BASELINE_TABLES = {
    "total":    "prediction_baseline_total",
    "sbi":      "prediction_baseline_sbi",
    "compsize": "prediction_baseline_compsize",
}


def _load_baseline(table_name: str) -> pl.DataFrame:
    """Reads a baseline table from the Gold DB."""
    query = f'SELECT * FROM "{table_name}"'
    import typing
    return typing.cast(pl.DataFrame, f_query_database(DIR_DB_GOLD, query, "polars"))


def _compute_metrics(df: pl.DataFrame) -> dict:
    """Computes MAE and RMSE from a baseline DataFrame."""
    return {
        "mae":  df.select(pl.col("abs_error").mean()).item(),
        "rmse": df.select((pl.col("residual_error") ** 2).mean().sqrt()).item(),
    }


def log_baseline_to_mlflow(experiment_name: str) -> None:
    """
    Logs baseline metrics to MLflow as three child runs under one parent run.

    Creates:
      - Baseline_RollingMean (parent run, tagged as baseline)
        ├── Baseline_total
        ├── Baseline_sbi
        └── Baseline_compsize

    Args:
        experiment_name: Must match the experiment name used in ModelOrchestrator
                         so all runs appear in the same MLflow experiment.
    """
    mlflow.set_experiment(experiment_name)

    logger.info("Logging baseline metrics to MLflow...")

    with mlflow.start_run(run_name="Baseline_RollingMean") as parent_run:
        mlflow.set_tag("type", "baseline")
        mlflow.set_tag("strategy", "rolling_3yr_same_quarter_per_sbi")
        mlflow.set_tag("source_tables", ", ".join(_BASELINE_TABLES.values()))

        for segment, table_name in _BASELINE_TABLES.items():
            df = _load_baseline(table_name)
            metrics = _compute_metrics(df)

            with mlflow.start_run(
                run_name=f"Baseline_{segment}",
                nested=True,
                tags={"type": "baseline", "segment": segment},
            ):
                mlflow.log_param("strategy", "rolling_3yr_same_quarter_per_sbi")
                mlflow.log_param("segment",  segment)
                mlflow.log_param("source_table", table_name)
                mlflow.log_metric("mae",  metrics["mae"])
                mlflow.log_metric("rmse", metrics["rmse"])

                logger.info(
                    f"  [{segment}] MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f}"
                )

            # Also log total-level metrics on the parent for easy top-level comparison
            if segment == "total":
                mlflow.log_metric("mae",  metrics["mae"])
                mlflow.log_metric("rmse", metrics["rmse"])

    logger.info(f"Baseline logged. Parent run ID: {parent_run.info.run_id}")