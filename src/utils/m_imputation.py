"""
Centralized Imputation Methodology.
Maintains all strategies for handling missing data across the pipeline.
This ensures our imputing logic is transparent, mathematically sound, and easily auditable by business users.
"""
from typing import List
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from src.utils.m_log import f_log

def impute_target_variable(df: pd.DataFrame, target_col: str, branch_col: str, date_col: str) -> pd.DataFrame:
    """
    Imputes missing values in the target variable to ensure a continuous time-series history.
    
    Methodology:
    1. Grouped Sort: Sorts chronologically by Branch and Date.
    2. Forward Fill (ffill): Bridges short temporal gaps within each Branch using the most recent valid observation.
    3. Median Imputation: For any remaining structural gaps (e.g. missing start-of-series), assigns the median central tendency.
    """
    df = df.copy()
    if target_col not in df.columns:
        return df
        
    initial_nans = df[target_col].isna().sum()
    if initial_nans == 0:
        return df
        
    f_log(f"Starting target imputation for '{target_col}'. Initial NaNs: {initial_nans}", c_type="process")
    
    # 1. Sort by Branch and Date
    if branch_col in df.columns and date_col in df.columns:
        df = df.sort_values(by=[branch_col, date_col]).reset_index(drop=True)
        # 2. Grouped Forward Fill
        df[target_col] = df.groupby(branch_col)[target_col].ffill()
        f_log(f"Applied grouped ffill. Remaining NaNs: {df[target_col].isna().sum()}", c_type="info")
    else:
        if date_col in df.columns:
            df = df.sort_values(by=[date_col]).reset_index(drop=True)
        df[target_col] = df[target_col].ffill()
        
    # 3. Median Imputation for remaining gaps
    if df[target_col].isna().any():
        imputer = SimpleImputer(strategy='median')
        df[[target_col]] = imputer.fit_transform(df[[target_col]])
        f_log(f"Applied median imputation. Remaining NaNs: {df[target_col].isna().sum()}", c_type="info")
        
    df[target_col] = df[target_col].astype('float64')
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
    """
    result = df.copy()
    DATE_COL = "period_enddate"
    SBI_COL = "BedrijfstakkenBranchesSBI2008"
    
    f_log(f"Starting imputation. Input shape: {result.shape}, "
          f"total NaNs: {result.isna().sum().sum()}", c_type="process")

    # 0. Extract structural keys
    structural_keys = [c for c in [DATE_COL, SBI_COL] if c in result.columns]
    key_data = result[structural_keys].copy() if structural_keys else None
    result = result.drop(columns=structural_keys, errors="ignore")

    # 1. Capture missing-indicator flags BEFORE any imputation
    if add_missing_indicator:
        cols_with_nans = result.columns[result.isna().any()].tolist()
        missing_flags = result[cols_with_nans].isna().astype(float)
        missing_flags.columns = [f"{c}_is_missing" for c in cols_with_nans]

    # 2. Drop non-numeric columns
    non_numeric_cols = result.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        result = result.drop(columns=non_numeric_cols)

    # 3. Separate OHE flags from continuous numeric columns
    ohe_cols = _identify_ohe_columns(result)
    numeric_cols = [c for c in result.columns if c not in ohe_cols]

    # 4. Impute OHE flags with 0
    if ohe_cols:
        result[ohe_cols] = result[ohe_cols].fillna(0.0)

    # 5. Impute continuous numeric columns with median/mean
    cols_needing_imputation = [c for c in numeric_cols if result[c].isna().any()]
    if cols_needing_imputation:
        imputer = SimpleImputer(strategy=numeric_strategy)
        result[cols_needing_imputation] = imputer.fit_transform(
            result[cols_needing_imputation]
        )

    # 6. Append missing-indicator columns
    if add_missing_indicator and len(cols_with_nans) > 0:
        surviving_indicators = [c for c in missing_flags.columns
                                if c.replace("_is_missing", "") in result.columns]
        result = pd.concat([result, missing_flags[surviving_indicators]], axis=1)

    # 7. Cast feature columns to float64
    result = result.astype("float64")

    # 8. Re-attach structural keys at the front
    if key_data is not None:
        result = pd.concat([key_data.reset_index(drop=True), result.reset_index(drop=True)], axis=1)

    f_log(f"Imputation complete. Output shape: {result.shape}, "
          f"remaining NaNs: {result.isna().sum().sum()}", c_type="success")
    return result
