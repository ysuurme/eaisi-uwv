"""
Step 1 — ML Data Extraction.

Extracts feature data from the Gold database (feature store).
The gold DB acts as the centralized feature store in this MLOps framework.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine

from src.utils.m_log import f_log


class DataExtractor:
    """Extracts a feature subset from the Gold feature store."""

    def __init__(self, db_path: Path, table_name: str):
        self.db_path = db_path
        self.table_name = table_name
        self.engine = create_engine(f"sqlite:///{self.db_path.as_posix()}")

    def extract(
        self,
        target_column: str,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Loads the gold table and selects target + requested features.

        Args:
            target_column: Name of the ML target column.
            features: Explicit feature list. If None, all numeric columns are used.

        Returns:
            DataFrame sorted by period_enddate with target + features.
        """
        df = pd.read_sql_table(self.table_name, self.engine)
        df = df.sort_values("period_enddate").reset_index(drop=True)

        if features:
            f_log(f"Selecting {len(features)} features from config: {features[:5]}...", c_type="process")
            columns_to_keep = [target_column] + features
            if "period_enddate" in df.columns:
                columns_to_keep = ["period_enddate"] + columns_to_keep
            df = df[[c for c in columns_to_keep if c in df.columns]]
        else:
            f_log("No feature subset defined. Using Discovery Mode (all numeric columns).", c_type="process")
            # Keep all numeric columns + period_enddate, drop structural keys
            non_feature_cols = ["silver_id"]
            df = df.drop(columns=[c for c in non_feature_cols if c in df.columns])

        f_log(
            f"Extracted {df.shape[1]} columns, {df.shape[0]} rows from '{self.table_name}'",
            c_type="success",
        )
        return df
