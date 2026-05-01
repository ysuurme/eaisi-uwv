"""
Step 2 — ML Data Validation.

Validates the extracted dataset before ML preparation.
Acts as a quality gate ensuring schema integrity, completeness, and type safety.
"""
from typing import List, Literal

import numpy as np
import pandas as pd

from src.utils.m_log import f_log


# Structural columns that are not ML features
DATE_COL = "period_enddate"
SBI_COL = "BedrijfstakkenBranchesSBI2008"


class DataValidator:
    """Validates extracted data before it enters the ML preparation step."""

    @staticmethod
    def validate(
        df: pd.DataFrame,
        target_column: str,
        stage: Literal["pre_prep", "post_prep"] = "pre_prep",
    ) -> pd.DataFrame:
        """Runs integrity checks on the extracted dataset.

        Args:
            df: The extracted DataFrame to validate.
            target_column: Name of the ML target column.
            stage: 'pre_prep' (before imputation, soft checks) or
                   'post_prep' (after preparation, strict zero-null enforcement).

        Returns:
            The same DataFrame unchanged (enables call-chaining).

        Raises:
            ValueError: If any strict check fails.
        """
        f_log(f"--- Data Validation Gate [{stage.upper()}] ---", c_type="process")

        _check_target_present(df, target_column)
        _check_date_column(df)
        _check_duplicates(df)
        _check_missing_values(df, stage)

        if stage == "post_prep":
            _check_float64_enforcement(df, target_column)

        f_log(f"Validation [{stage.upper()}] passed | Shape: {df.shape}", c_type="success")
        return df


def _check_target_present(df: pd.DataFrame, target_column: str) -> None:
    """Target column must exist in the dataset."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' missing from dataset.")
    f_log(f"✅ Target '{target_column}' present.", c_type="success")


def _check_date_column(df: pd.DataFrame) -> None:
    """Temporal key must be present and parseable as datetime for time-series splitting."""
    if DATE_COL not in df.columns:
        raise ValueError(f"Required column '{DATE_COL}' missing from dataset.")
    try:
        pd.to_datetime(df[DATE_COL])
        f_log(f"✅ '{DATE_COL}' present and parseable as datetime.", c_type="success")
    except Exception as e:
        raise ValueError(f"'{DATE_COL}' cannot be parsed as datetime: {e}")


def _check_duplicates(df: pd.DataFrame) -> None:
    """Check for duplicate rows on available key columns."""
    has_sbi = SBI_COL in df.columns
    key_cols = [DATE_COL, SBI_COL] if has_sbi else [DATE_COL]
    duplicate_count = df.duplicated(subset=key_cols).sum()

    if duplicate_count == 0:
        f_log(f"✅ No duplicate key rows ({key_cols}).", c_type="success")
        return

    if has_sbi:
        raise ValueError(f"{duplicate_count} duplicate rows on composite key {key_cols}.")

    # Date-only key: duplicates expected (multiple SBI sectors per quarter)
    f_log(
        f"⚠️ {duplicate_count} duplicate rows on {key_cols}. "
        f"Expected: multiple SBI sectors share the same date key.",
        c_type="warning",
    )


def _check_missing_values(df: pd.DataFrame, stage: str) -> None:
    """Soft check pre-prep, strict zero-null enforcement post-prep."""
    total_nulls = df.isna().sum().sum()

    if stage == "pre_prep":
        if total_nulls > 0:
            nan_pct = (df.isna().sum() / len(df) * 100).round(1)
            top_missing = nan_pct[nan_pct > 0].sort_values(ascending=False).head(10)
            f_log(f"⚠️ {total_nulls} NaN values. Top: {top_missing.to_dict()}", c_type="warning")
        else:
            f_log("✅ No missing values.", c_type="success")
        return

    if total_nulls > 0:
        raise ValueError(f"Post-preparation dataset still contains {total_nulls} NaN values.")
    f_log("✅ Zero missing values confirmed.", c_type="success")


def _check_float64_enforcement(df: pd.DataFrame, target_column: str) -> None:
    """All feature columns must be float64 for ML compatibility."""
    structural_cols = {DATE_COL, SBI_COL, target_column}
    feature_cols = [c for c in df.columns if c not in structural_cols]
    non_float = [c for c in feature_cols if df[c].dtype != np.float64]

    if non_float:
        raise ValueError(f"Columns not float64: {non_float[:10]}")
    f_log("✅ All feature columns are float64.", c_type="success")
