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
    key_cols = [DATE_COL, SBI_COL] if SBI_COL in df.columns else [DATE_COL]
    duplicate_count = df.duplicated(subset=key_cols).sum()
    if duplicate_count > 0:
        raise ValueError(
            f"{duplicate_count} duplicate rows detected on key columns {key_cols}."
        )
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

    # 4. Float64 enforcement (clean stage only)
    if stage == "clean":
        non_float = [c for c in df.columns if df[c].dtype != np.float64]
        if non_float:
            raise ValueError(
                f"Columns not float64: {non_float[:10]}{'...' if len(non_float) > 10 else ''}"
            )
        f_log("✅ All columns are float64.", c_type="success")

    f_log(f"✅ Validation gate [{stage.upper()}] passed. Shape: {df.shape}", c_type="success")
    return df


def _identify_ohe_columns(df: pd.DataFrame) -> List[str]:
    """Return columns that are binary OHE flags (values exclusively in {0, 1, NaN})."""
    ohe_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0}):
            ohe_cols.append(col)
    return ohe_cols


def impute_missing_values(
    df: pd.DataFrame,
    *,
    numeric_strategy: str = "median",
    add_missing_indicator: bool = True,
) -> pd.DataFrame:
    """
    Return a fully imputed copy of *df* with all columns cast to float64.

    Imputation tiers:
    - Binary OHE flags ({0, 1, NaN}) → filled with 0 (absent = not observed)
    - Remaining numeric columns → SimpleImputer(strategy=numeric_strategy)
    - Non-numeric columns → dropped (should not exist in a Gold table)

    Args:
        df: Input DataFrame (typically master_data_ml_joined).
        numeric_strategy: 'median' (default, robust to skew) or 'mean'.
        add_missing_indicator: If True, creates <col>_is_missing binary flags.

    Returns:
        A new DataFrame with zero NaNs, all float64.
    """
    result = df.copy()
    f_log(f"Starting imputation. Input shape: {result.shape}, "
          f"total NaNs: {result.isna().sum().sum()}", c_type="process")

    # 1. Capture missing-indicator flags BEFORE any imputation
    if add_missing_indicator:
        cols_with_nans = result.columns[result.isna().any()].tolist()
        missing_flags = result[cols_with_nans].isna().astype(float)
        missing_flags.columns = [f"{c}_is_missing" for c in cols_with_nans]
        f_log(f"Created {len(cols_with_nans)} missing-indicator columns.", c_type="info")

    # 2. Drop non-numeric columns (should already be absent in Gold, but safety net)
    non_numeric_cols = result.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        f_log(f"Dropping {len(non_numeric_cols)} non-numeric columns: "
              f"{non_numeric_cols[:5]}{'...' if len(non_numeric_cols) > 5 else ''}", c_type="warning")
        result = result.drop(columns=non_numeric_cols)

    # 3. Separate OHE flags from continuous numeric columns
    ohe_cols = _identify_ohe_columns(result)
    numeric_cols = [c for c in result.columns if c not in ohe_cols]

    # 4. Impute OHE flags with 0 (absent = not observed)
    if ohe_cols:
        result[ohe_cols] = result[ohe_cols].fillna(0.0)
        f_log(f"Imputed {len(ohe_cols)} OHE columns with 0.", c_type="info")

    # 5. Impute continuous numeric columns with median/mean
    cols_needing_imputation = [c for c in numeric_cols if result[c].isna().any()]
    if cols_needing_imputation:
        imputer = SimpleImputer(strategy=numeric_strategy)
        result[cols_needing_imputation] = imputer.fit_transform(
            result[cols_needing_imputation]
        )
        f_log(f"Imputed {len(cols_needing_imputation)} numeric columns "
              f"with {numeric_strategy}.", c_type="info")

    # 6. Append missing-indicator columns
    if add_missing_indicator and len(cols_with_nans) > 0:
        # Only keep indicators for columns that survived the drop step
        surviving_indicators = [c for c in missing_flags.columns
                                if c.replace("_is_missing", "") in result.columns]
        result = pd.concat([result, missing_flags[surviving_indicators]], axis=1)

    # 7. Cast everything to float64 (MLflow / Gold Policy)
    result = result.astype("float64")

    f_log(f"Imputation complete. Output shape: {result.shape}, "
          f"remaining NaNs: {result.isna().sum().sum()}", c_type="success")
    return result
