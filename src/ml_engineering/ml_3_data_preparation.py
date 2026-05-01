"""
Step 3 — ML Data Preparation.

Prepares validated data for model training:
- Separates features from target
- Casts all features to float64
- Performs time-series aware train/test split
- Applies imputation if needed (delegates to m_imputation utilities)
"""
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.utils.m_log import f_log


DATE_COL = "period_enddate"
SBI_COL = "BedrijfstakkenBranchesSBI2008"


class DataPreparator:
    """Prepares validated data for ML training."""

    @staticmethod
    def prepare(
        df: pd.DataFrame,
        target_column: str,
        n_splits: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        """Separates features from target and performs a time-series split.

        Args:
            df: Validated DataFrame with target + features.
            target_column: Name of the ML target column.
            n_splits: Number of TimeSeriesSplit folds (last fold used).

        Returns:
            (X_train, X_test, y_train, y_test, lineage_metadata)
        """
        y = df[target_column].astype(float)
        x = _extract_feature_matrix(df, target_column)
        train_idx, test_idx = _time_series_split(x, n_splits)

        lineage = {
            "dataset": "gold_feature_store",
            "target": target_column,
            "feature_count": x.shape[1],
            "train_size": len(train_idx),
            "test_size": len(test_idx),
        }

        f_log(
            f"Prepared | Features: {x.shape[1]}, Train: {len(train_idx)}, Test: {len(test_idx)}",
            c_type="success",
        )
        return x.iloc[train_idx], x.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], lineage


def _extract_feature_matrix(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Isolates numeric feature columns, excludes structural keys, casts to float64."""
    exclude_cols = {target_column, DATE_COL, SBI_COL, "silver_id"}
    x = df.select_dtypes(include="number").drop(
        columns=[c for c in exclude_cols if c in df.columns],
        errors="ignore",
    )
    x.columns = [str(col) for col in x.columns]
    return x.astype("float64")


def _time_series_split(x: pd.DataFrame, n_splits: int) -> Tuple[Any, Any]:
    """Returns the last fold of a TimeSeriesSplit as (train_indices, test_indices)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    *_, (train_index, test_index) = tscv.split(x)
    return train_index, test_index
