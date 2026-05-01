"""
Step 1 — ML Data Extraction.

Extracts feature data from the Gold database (feature store) and reduces the
sector-level panel to exactly one row per unique quarter date before passing
data downstream.

Operational modes
-----------------
all-industry (default, sbi_filter_col=None)
    Filters on BedrijfskenmerkenSBI2008_T001081 == 1, which is the national
    total row already present in the gold table.  This mirrors the notebook's
    ``df[df.sbi_code == "T001081"]`` approach — no averaging is required
    because T001081 is a pre-computed aggregate in the CBS source data.

sector-specific (sbi_filter_col='BedrijfskenmerkenSBI2008_301000')
    Filters rows where the named OHE column equals 1, isolating one SBI sector.
    Exactly one row per quarter remains after the filter.

In both modes the 39 OHE SBI indicator columns are dropped before output, and
the result has exactly one row per unique quarter date.

Feature selection modes
-----------------------
groups    — resolves named groups from FEATURE_CATALOG to concrete column names
discovery — uses all remaining columns after structural and OHE columns are
            removed
"""
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import create_engine

from src.ml_engineering.model_configs import FEATURE_CATALOG
from src.utils.m_log import f_log

_DATE_COL = "period_enddate"

# OHE column for the CBS national total (T001081) — mirrors notebook's sbi_code == "T001081"
_NATIONAL_TOTAL_COL = "BedrijfskenmerkenSBI2008_T001081"

# Columns kept as structural context (date + temporal indices)
# Note: BedrijfstakkenBranchesSBI2008 does NOT exist in the gold feature store;
# sector identity is encoded via OHE columns (BedrijfskenmerkenSBI2008_*).
_KEEP_STRUCTURAL = {"period_enddate", "year", "quarter"}

# Silently dropped — pipeline artefacts with no ML or context value
_DROP_ALWAYS = {"silver_id"}

_STRUCTURAL_COLUMNS = _KEEP_STRUCTURAL | _DROP_ALWAYS


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
        sbi_filter_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Loads the gold table, applies SBI mode, and collapses to 1 row per quarter.

        Args:
            target_column: Name of the ML target column.
            feature_groups: Named groups from FEATURE_CATALOG.  If None, all
                columns after structural and OHE removal are used (discovery).
            sbi_filter_col: OHE column name to use as a sector filter, e.g.
                ``'BedrijfskenmerkenSBI2008_301000'``.  When None the pipeline
                runs in all-industry mode (aggregate across all sectors).

        Returns:
            DataFrame with one row per quarter, sorted by period_enddate,
            containing the target + selected features.
        """
        df = pd.read_sql_table(self.table_name, self.engine)
        df = df.sort_values(_DATE_COL).reset_index(drop=True)

        # Apply SBI mode: filter / aggregate → 1 row per quarter
        df = self._apply_sbi_mode(df, sbi_filter_col, target_column)

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

        structural_present = [c for c in df.columns if c in _KEEP_STRUCTURAL]
        columns_to_keep = structural_present + [target_column] + feature_columns
        df = df[[c for c in columns_to_keep if c in available_columns]]

        sbi_label = sbi_filter_col or "all-industry"
        f_log(
            f"Extraction complete | mode={mode} | sbi={sbi_label} | "
            f"features={len(feature_columns)} | rows={df.shape[0]}",
            c_type="success",
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_sbi_mode(
        df: pd.DataFrame,
        sbi_filter_col: Optional[str],
        target_column: str,
    ) -> pd.DataFrame:
        """Reduces the sector panel to exactly one row per quarter.

        All-industry mode (sbi_filter_col=None)
            Filters on BedrijfskenmerkenSBI2008_T001081 == 1.  T001081 is the
            CBS national total — a pre-computed aggregate already present as a
            row in the gold table.  This exactly mirrors the notebook's
            ``df[df.sbi_code == "T001081"]``.  No averaging is performed.

        Sector-specific mode (sbi_filter_col='BedrijfskenmerkenSBI2008_301000')
            Filters rows where the named OHE flag equals 1, isolating one SBI
            sector.  The result is already one row per quarter.

        In both modes all OHE SBI columns are then dropped because they carry
        no predictive signal once the series is isolated to a single entity.
        """
        ohe_cols = [c for c in df.columns if c.startswith("BedrijfskenmerkenSBI2008_")]

        # Determine which OHE column identifies the desired series
        effective_col = sbi_filter_col if sbi_filter_col is not None else _NATIONAL_TOTAL_COL
        mode_label = f"sector:{sbi_filter_col}" if sbi_filter_col else f"all-industry ({_NATIONAL_TOTAL_COL})"

        if effective_col not in df.columns:
            raise ValueError(
                f"SBI column '{effective_col}' not found in dataset.  "
                f"OHE columns start with 'BedrijfskenmerkenSBI2008_'."
            )

        df = df[df[effective_col] == 1].copy()
        if df.empty:
            raise ValueError(
                f"No rows remain after filtering on {effective_col} == 1."
            )

        # Drop all OHE SBI columns — sector identity is now implicit in the filter
        df = df.drop(columns=ohe_cols, errors="ignore")
        df = df.sort_values(_DATE_COL).reset_index(drop=True)

        f_log(
            f"SBI mode={mode_label} | {len(df)} quarterly rows",
            c_type="process",
        )
        return df

    def _resolve_groups(self, group_names: List[str], available_columns: set) -> List[str]:
        """Resolves feature group names to column names, logging any gaps."""
        resolved: List[str] = []
        for group_name in group_names:
            if group_name not in FEATURE_CATALOG:
                f_log(
                    f"Feature group '{group_name}' not found in FEATURE_CATALOG — skipped.",
                    c_type="warning",
                )
                continue
            group = FEATURE_CATALOG[group_name]
            for col in group.columns:
                if col in available_columns:
                    resolved.append(col)
                else:
                    f_log(
                        f"Column '{col}' from group '{group_name}' not in table — skipped.",
                        c_type="warning",
                    )
        return resolved
