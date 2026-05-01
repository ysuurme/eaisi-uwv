"""
Step 3 — ML Data Preparation.

Prepares validated data for model training:
- Parses period_enddate as DatetimeIndex
- Performs a time-aware train/test split (last n_test unique quarters = test window)
- Builds X (feature matrix) and y (target series) from the quarterly time series

Data contract (from Step 1)
---------------------------
Step 1 always delivers exactly one row per unique quarter date (either
all-industry aggregate or a single-sector filtered series).  Step 3 therefore
performs a simple temporal split — no panel-level logic is required.
"""
from typing import Any, Dict, Tuple

import pandas as pd

from src.utils.m_log import f_log


DATE_COL = "period_enddate"


class DataPreparator:
    """Prepares validated quarterly data for ML training."""

    @staticmethod
    def prepare(
        df: pd.DataFrame,
        target_column: str,
        n_test: int = 4,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        """Performs a time-aware split and sets a DatetimeIndex.

        Args:
            df: Validated DataFrame with target + features.
                Must contain exactly one row per unique quarter date
                (guaranteed by Step 1's aggregation).
            target_column: Name of the ML target column.
            n_test: Number of unique quarter dates to hold out as the test
                    window.  Default 4 = 1 year = the 4-quarter forecast horizon.

        Returns:
            (X_train, X_test, y_train, y_test, lineage_metadata)
        """
        df = df.copy()
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df = df.sort_values(DATE_COL).reset_index(drop=True)

        # Temporal split: last n_test unique quarters form the test window
        unique_dates = sorted(df[DATE_COL].unique())
        if len(unique_dates) <= n_test:
            raise ValueError(
                f"Dataset has only {len(unique_dates)} unique dates but n_test={n_test}. "
                "Reduce n_test or provide more historical data."
            )
        cutoff_date = unique_dates[-n_test]   # first date of the test window

        train_mask = df[DATE_COL] < cutoff_date
        test_mask  = df[DATE_COL] >= cutoff_date

        # Set DatetimeIndex (unique per quarter in the 1-row-per-quarter output)
        df = df.set_index(pd.DatetimeIndex(df[DATE_COL], freq=None))

        y = df[target_column].astype(float)
        X = _extract_feature_matrix(df, target_column)

        y_train, y_test = y[train_mask.values], y[test_mask.values]
        X_train, X_test = X[train_mask.values], X[test_mask.values]

        lineage = {
            "dataset": "gold_feature_store",
            "target": target_column,
            "feature_count": X.shape[1],
            "train_size": len(y_train),
            "test_size": len(y_test),
            "cutoff_date": str(cutoff_date),
        }

        f_log(
            f"Prepared | Temporal split | cutoff={cutoff_date.date()} | "
            f"Features: {X.shape[1]}, Train: {len(y_train)}, Test: {len(y_test)}",
            c_type="success",
        )
        return X_train, X_test, y_train, y_test, lineage


def _extract_feature_matrix(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Isolates feature columns for ML, excluding target and date artefacts.

    - Selects all numeric columns (includes year, quarter, and all CBS features)
    - Excludes: target column, period_enddate, silver_id
    - All OHE SBI columns were already dropped by Step 1
    """
    exclude_cols = {target_column, DATE_COL, "silver_id"}
    x = df.select_dtypes(include="number").drop(
        columns=[c for c in exclude_cols if c in df.columns],
        errors="ignore",
    )
    x.columns = [str(col) for col in x.columns]
    return x
