"""
Step 1 — ML Data Extraction.

Extracts feature data from the Gold database (feature store).
The gold DB acts as the centralized feature store in this MLOps framework.

Feature selection modes:
    groups    — resolves named groups from FEATURE_CATALOG to concrete column names
    discovery — uses all columns in the table except the target and structural keys
"""
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import create_engine

from src.ml_engineering.model_configs import FEATURE_CATALOG
from src.utils.m_log import f_log

# Columns that are never treated as ML features regardless of selection mode
_STRUCTURAL_COLUMNS = {"silver_id", "period_enddate", "BedrijfstakkenBranchesSBI2008"}


class DataExtractor:
    """Extracts a feature subset from the Gold feature store."""

    def __init__(self, db_path: Path, table_name: str):
        self.db_path = db_path
        self.table_name = table_name
        self.engine = create_engine(f"sqlite:///{self.db_path.as_posix()}")

    def extract(
        self,
        target_column: str,
        feature_groups: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Loads the gold table and selects target + features.

        Args:
            target_column: Name of the ML target column.
            feature_groups: Named groups from FEATURE_CATALOG. If None, all columns
                except the target and structural keys are used (discovery mode).

        Returns:
            DataFrame sorted by period_enddate with target + selected features.
        """
        df = pd.read_sql_table(self.table_name, self.engine)
        df = df.sort_values("period_enddate").reset_index(drop=True)

        available_columns = set(df.columns)

        if feature_groups is not None:
            feature_columns = self._resolve_groups(feature_groups, available_columns)
            mode = "groups"
        else:
            feature_columns = [
                c for c in df.columns
                if c != target_column and c not in _STRUCTURAL_COLUMNS
            ]
            mode = "discovery"

        structural_present = [c for c in df.columns if c in _STRUCTURAL_COLUMNS]
        columns_to_keep = structural_present + [target_column] + feature_columns
        df = df[[c for c in columns_to_keep if c in available_columns]]

        f_log(
            f"Extraction complete | mode={mode} | features={len(feature_columns)} | rows={df.shape[0]}",
            c_type="success",
        )
        return df

    def _resolve_groups(self, group_names: List[str], available_columns: set) -> List[str]:
        """Resolves feature group names to column names, logging any gaps."""
        resolved: List[str] = []
        for group_name in group_names:
            if group_name not in FEATURE_CATALOG:
                f_log(f"Feature group '{group_name}' not found in FEATURE_CATALOG — skipped.", c_type="warning")
                continue
            group = FEATURE_CATALOG[group_name]
            for col in group.columns:
                if col in available_columns:
                    resolved.append(col)
                else:
                    f_log(f"Column '{col}' from group '{group_name}' not in table — skipped.", c_type="warning")
        return resolved
