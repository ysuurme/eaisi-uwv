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

Full-panel loading
------------------
``load_full_panel()`` loads the complete gold table without SBI filtering,
keeping all sectors and all OHE columns.  It reconstructs a categorical
``sector`` column from the OHE encoding.  This is used by the feature
selection module, which needs cross-sector analysis.
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

# OHE column prefix for all SBI sector indicators
_OHE_PREFIX = "BedrijfskenmerkenSBI2008_"

# Columns kept as structural context (date + temporal indices + regime indicators).
# Note: BedrijfstakkenBranchesSBI2008 does NOT exist in the gold feature store;
# sector identity is encoded via OHE columns (BedrijfskenmerkenSBI2008_*).
#
# trend_index / covid_period / post_covid / covid_depth / recovery_quarters /
# trend_x_post_covid / quarter_x_post_covid are theoretically motivated regime
# indicators that must always reach the model regardless of preset filtering.
# trend_index in particular is declared STRUCTURAL in feature_selection_utils.py
# and is therefore never in any preset's surviving_features list — without this
# injection it would never reach the model.
#
# The continuous (covid_depth, recovery_quarters) and interaction
# (trend_x_post_covid, quarter_x_post_covid) features give linear models the
# lever to fit regime-dependent slope and seasonality without expanding
# parameter count uncontrollably.
_KEEP_STRUCTURAL = {
    "period_enddate", "year", "quarter",
    "trend_index",
    "covid_period", "post_covid",
    "covid_depth", "recovery_quarters",
    "trend_x_post_covid", "quarter_x_post_covid",
}

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
            feature_columns = self.derive_feature_columns(df, target_column)
            mode = "discovery"

        structural_present = [c for c in df.columns if c in _KEEP_STRUCTURAL]
        # Order-preserving dedup: a structural column may also appear in
        # feature_columns if a preset's group lists it explicitly.  Without
        # dedup, df[[col, col]] would return duplicate columns and break
        # downstream numeric dtype assertions.
        columns_to_keep = list(dict.fromkeys(
            structural_present + [target_column] + feature_columns
        ))
        df = df[[c for c in columns_to_keep if c in available_columns]]

        sbi_label = sbi_filter_col or "all-industry"
        f_log(
            f"Extraction complete | mode={mode} | sbi={sbi_label} | "
            f"features={len(feature_columns)} | rows={df.shape[0]}",
            c_type="success",
        )
        return df

    # ------------------------------------------------------------------
    # Feature-column derivation (single source of truth)
    # ------------------------------------------------------------------

    @classmethod
    def derive_feature_columns(
        cls,
        df: pd.DataFrame,
        target_column: str,
    ) -> List[str]:
        """Derive candidate ML feature columns from a gold-table frame.

        Single source of truth for "what is a feature column": every column
        except the target, the structural context (date/temporal keys, regime
        indicators, pipeline artefacts), the OHE sector indicators, and the
        synthetic ``sector`` label added by ``load_full_panel()``.

        Used by ``extract()`` in discovery mode (where OHE columns are already
        dropped) and by the feature-selection flow
        (``ml_orchestrator.run_feature_selection``), which operates on the full
        panel.  Column order follows the frame (deterministic).
        """
        excluded = _STRUCTURAL_COLUMNS | {target_column, "sector"}
        return [
            c for c in df.columns
            if c not in excluded and not c.startswith(_OHE_PREFIX)
        ]

    # ------------------------------------------------------------------
    # Full-panel loader for feature selection
    # ------------------------------------------------------------------

    @classmethod
    def load_full_panel(
        cls,
        db_path: Path,
        table_name: str = "master_data_ml_preprocessed",
    ) -> pd.DataFrame:
        """Load the complete gold table without SBI filtering.

        Returns the full panel (all sectors × all quarters) with a
        reconstructed categorical ``sector`` column derived from the OHE
        columns.  The OHE columns are retained so that downstream code
        can identify and exclude them from the feature set.

        This method is used by ``feature_selection.py``, which needs
        cross-sector analysis (per-sector correlation, Granger tests, etc.)
        that requires the complete panel — unlike ``extract()``, which
        filters to a single sector for the ML pipeline.

        Args:
            db_path: Path to the Gold SQLite database.
            table_name: Name of the preprocessed table in the gold store.

        Returns:
            DataFrame with all rows, all columns, plus a synthetic
            ``sector`` column (e.g. ``"T001081"``, ``"301000"``).
        """
        engine = create_engine(f"sqlite:///{db_path.as_posix()}")
        df = pd.read_sql_table(table_name, engine)
        df = df.sort_values(_DATE_COL).reset_index(drop=True)

        # Reconstruct sector identity from OHE columns.
        # Each row has exactly one OHE column == 1; idxmax finds it.
        ohe_cols = [c for c in df.columns if c.startswith(_OHE_PREFIX)]
        if not ohe_cols:
            raise ValueError(
                f"No OHE columns with prefix '{_OHE_PREFIX}' found in "
                f"table '{table_name}'.  Cannot reconstruct sector identity."
            )
        df["sector"] = (
            df[ohe_cols]
            .idxmax(axis=1)
            .str.replace(_OHE_PREFIX, "", regex=False)
        )

        f_log(
            f"Full panel loaded | {df.shape[0]} rows × {df.shape[1]} cols | "
            f"{df['sector'].nunique()} sectors | "
            f"{df[_DATE_COL].nunique()} quarters",
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
        ohe_cols = [c for c in df.columns if c.startswith(_OHE_PREFIX)]

        # Determine which OHE column identifies the desired series
        effective_col = sbi_filter_col if sbi_filter_col is not None else _NATIONAL_TOTAL_COL
        mode_label = f"sector:{sbi_filter_col}" if sbi_filter_col else f"all-industry ({_NATIONAL_TOTAL_COL})"

        if effective_col not in df.columns:
            raise ValueError(
                f"SBI column '{effective_col}' not found in dataset.  "
                f"OHE columns start with '{_OHE_PREFIX}'."
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
