"""
Preprocessing module for the ML-ready master dataset.

Provides two public functions:
- validate_master_dataset(): schema and integrity checks before/after imputation
- impute_missing_values(): deterministic null-safe imputation with missing indicators

All operations are pure (no side-effects beyond logging) and type-annotated.
"""

from __future__ import annotations

from typing import Literal, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.utils.m_log import f_log


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATE_COL = "period_enddate"
SBI_COL = "BedrijfstakkenBranchesSBI2008"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_master_dataset(
    df: pd.DataFrame,
    *,
    stage: Literal["raw", "clean"],
) -> pd.DataFrame:
    """
    Run integrity checks on the master dataset at a given pipeline stage.

    Args:
        df: The master DataFrame to validate.
        stage: 'raw' (pre-imputation, soft checks) or 'clean' (post-imputation, strict).

    Returns:
        The same DataFrame unchanged (enables call-chaining).

    Raises:
        ValueError: If any strict check fails (clean stage only).
    """
    f_log(f"--- Validation Gate [{stage.upper()}] ---", c_type="process")

    # 1. Temporal key must be present
    if DATE_COL not in df.columns:
        raise ValueError(f"Required column '{DATE_COL}' is missing from the dataset.")
    f_log(f"✅ '{DATE_COL}' column present.", c_type="success")

    # 2. Duplicate key rows
    has_full_key = SBI_COL in df.columns
    key_cols = [DATE_COL, SBI_COL] if has_full_key else [DATE_COL]
    duplicate_count = df.duplicated(subset=key_cols).sum()
    if duplicate_count > 0:
        if has_full_key:
            # Full composite key available but still duplicates → real data issue
            raise ValueError(
                f"{duplicate_count} duplicate rows detected on composite key {key_cols}."
            )
        # Date-only key: duplicates are expected (multiple SBI sectors per quarter)
        f_log(f"⚠️ {duplicate_count} duplicate rows on {key_cols}. "
              f"Expected: multiple SBI sectors share the same date key.",
              c_type="warning")
    else:
        f_log(f"✅ No duplicate key rows ({key_cols}).", c_type="success")

    # 3. Missing-value assessment
    total_nulls = df.isna().sum().sum()
    if stage == "raw":
        # Soft log: report NaN density per column for engineer visibility
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            pct_missing = (df.isna().sum() / len(df) * 100).round(1)
            top_missing = pct_missing[pct_missing > 0].sort_values(ascending=False).head(10)
            f_log(f"⚠️ {len(nan_cols)} columns contain NaNs ({total_nulls} total). "
                  f"Top missing %: {top_missing.to_dict()}", c_type="warning")
        else:
            f_log("✅ No missing values.", c_type="success")
    else:
        # Strict: zero NaNs required after preprocessing
        if total_nulls > 0:
            raise ValueError(
                f"Post-imputation dataset still contains {total_nulls} NaN values."
            )
        f_log("✅ Zero missing values confirmed.", c_type="success")

    # 4. Float64 enforcement (clean stage only, excludes structural keys)
    if stage == "clean":
        feature_cols = [c for c in df.columns if c not in (DATE_COL, SBI_COL)]
        non_float = [c for c in feature_cols if df[c].dtype != np.float64]
        if non_float:
            raise ValueError(
                f"Columns not float64: {non_float[:10]}{'...' if len(non_float) > 10 else ''}"
            )
        f_log("✅ All feature columns are float64.", c_type="success")

    f_log(f"✅ Validation gate [{stage.upper()}] passed. Shape: {df.shape}", c_type="success")
    return df



