"""
Unit tests for Step 1 — ML Data Extraction.

All mock DataFrames include BedrijfskenmerkenSBI2008_T001081 = 1 because
_apply_sbi_mode always filters on that column in all-industry mode (default).
In production the gold table always contains this national-total row.
"""

import unittest
from unittest.mock import patch
import pandas as pd
from pathlib import Path

from src.ml_engineering.ml_1_data_extraction import DataExtractor
from src.ml_engineering.model_configs import FeatureGroup


_MOCK_CATALOG = {
    "test_group": FeatureGroup(
        name="test_group",
        columns=["feat1"],
        source_table="test_table",
        description="Test group with a single feature.",
    )
}

_N = 5  # rows in every mock DataFrame


def _base_df(**extra_cols) -> pd.DataFrame:
    """Minimal gold-table-like DataFrame with the national total OHE flag set."""
    data = {
        "period_enddate": pd.date_range("2020-01-01", periods=_N, freq="QE"),
        "year": [2020, 2020, 2020, 2020, 2021],
        "quarter": [1, 2, 3, 4, 1],
        "target": list(range(1, _N + 1)),
        # National total flag — required by _apply_sbi_mode default path
        "BedrijfskenmerkenSBI2008_T001081": [1] * _N,
    }
    data.update(extra_cols)
    return pd.DataFrame(data)


class TestDataExtraction(unittest.TestCase):

    @patch("src.ml_engineering.ml_1_data_extraction.create_engine")
    @patch("src.ml_engineering.ml_1_data_extraction.pd.read_sql_table")
    @patch("src.ml_engineering.ml_1_data_extraction.FEATURE_CATALOG", _MOCK_CATALOG)
    def test_extract_by_group(self, mock_read_sql, mock_engine):
        """Groups mode: only columns declared in the resolved group are kept."""
        mock_read_sql.return_value = _base_df(
            feat1=[10, 20, 30, 40, 50],
            feat2=[0.1, 0.2, 0.3, 0.4, 0.5],
            silver_id=range(_N),
        )

        extractor = DataExtractor(Path("fake.db"), "gold_table")
        extracted_df = extractor.extract(target_column="target", feature_groups=["test_group"])

        self.assertIn("feat1", extracted_df.columns)
        self.assertIn("target", extracted_df.columns)
        self.assertNotIn("feat2", extracted_df.columns)
        self.assertNotIn("silver_id", extracted_df.columns)
        # OHE column dropped after filtering
        self.assertNotIn("BedrijfskenmerkenSBI2008_T001081", extracted_df.columns)

    @patch("src.ml_engineering.ml_1_data_extraction.create_engine")
    @patch("src.ml_engineering.ml_1_data_extraction.pd.read_sql_table")
    def test_extract_discovery_mode(self, mock_read_sql, mock_engine):
        """Discovery mode: all columns except target and structural keys are kept."""
        mock_read_sql.return_value = _base_df(
            feat1=[10, 20, 30, 40, 50],
            silver_id=range(_N),
        )

        extractor = DataExtractor(Path("fake.db"), "gold_table")
        extracted_df = extractor.extract(target_column="target", feature_groups=None)

        self.assertIn("feat1", extracted_df.columns)
        self.assertNotIn("silver_id", extracted_df.columns)
        # OHE column dropped after filtering
        self.assertNotIn("BedrijfskenmerkenSBI2008_T001081", extracted_df.columns)

    @patch("src.ml_engineering.ml_1_data_extraction.create_engine")
    @patch("src.ml_engineering.ml_1_data_extraction.pd.read_sql_table")
    @patch("src.ml_engineering.ml_1_data_extraction.FEATURE_CATALOG", _MOCK_CATALOG)
    def test_extract_unknown_group_logs_warning(self, mock_read_sql, mock_engine):
        """Unknown group name logs a warning and is skipped — no crash."""
        mock_read_sql.return_value = _base_df(feat1=[10, 20, 30, 40, 50])

        extractor = DataExtractor(Path("fake.db"), "gold_table")
        extracted_df = extractor.extract(
            target_column="target", feature_groups=["nonexistent_group"]
        )

        self.assertIn("target", extracted_df.columns)

    @patch("src.ml_engineering.ml_1_data_extraction.create_engine")
    @patch("src.ml_engineering.ml_1_data_extraction.pd.read_sql_table")
    @patch("src.ml_engineering.ml_1_data_extraction.FEATURE_CATALOG", _MOCK_CATALOG)
    def test_extract_missing_column_in_group_logs_warning(self, mock_read_sql, mock_engine):
        """Column declared in a group but absent from the table is skipped — no crash."""
        mock_read_sql.return_value = _base_df()  # feat1 intentionally absent

        extractor = DataExtractor(Path("fake.db"), "gold_table")
        extracted_df = extractor.extract(target_column="target", feature_groups=["test_group"])

        self.assertNotIn("feat1", extracted_df.columns)
        self.assertIn("target", extracted_df.columns)

    @patch("src.ml_engineering.ml_1_data_extraction.create_engine")
    @patch("src.ml_engineering.ml_1_data_extraction.pd.read_sql_table")
    def test_extract_sector_specific_mode(self, mock_read_sql, mock_engine):
        """Sector-specific mode: filters on the given OHE column and drops all OHE cols."""
        mock_read_sql.return_value = _base_df(
            feat1=[10, 20, 30, 40, 50],
            # Add a second sector column; all rows belong to T001081 only
            BedrijfskenmerkenSBI2008_301000=[0] * _N,
        )

        extractor = DataExtractor(Path("fake.db"), "gold_table")
        # Filter on T001081 explicitly (same result as default, but tests the sector path)
        extracted_df = extractor.extract(
            target_column="target",
            sbi_filter_col="BedrijfskenmerkenSBI2008_T001081",
        )

        self.assertIn("feat1", extracted_df.columns)
        self.assertIn("target", extracted_df.columns)
        self.assertNotIn("BedrijfskenmerkenSBI2008_T001081", extracted_df.columns)
        self.assertNotIn("BedrijfskenmerkenSBI2008_301000", extracted_df.columns)

    @patch("src.ml_engineering.ml_1_data_extraction.create_engine")
    @patch("src.ml_engineering.ml_1_data_extraction.pd.read_sql_table")
    def test_extract_missing_sbi_filter_col_raises(self, mock_read_sql, mock_engine):
        """Passing a non-existent sbi_filter_col raises a clear ValueError."""
        mock_read_sql.return_value = _base_df()

        extractor = DataExtractor(Path("fake.db"), "gold_table")
        with self.assertRaises(ValueError):
            extractor.extract(
                target_column="target",
                sbi_filter_col="BedrijfskenmerkenSBI2008_NONEXISTENT",
            )

    def test_derive_feature_columns_excludes_structural_ohe_and_sector(self):
        """derive_feature_columns is the single source of truth for feature columns:
        target, structural context, OHE indicators, and the synthetic sector
        label are excluded; everything else (incl. y_* yearly columns — frequency
        filtering happens at the registry layer, not here) is a feature."""
        df = _base_df(
            feat1=[10, 20, 30, 40, 50],
            y_feat_yearly=[1, 1, 1, 1, 1],
            silver_id=range(_N),
            trend_index=range(_N),
            covid_period=[0] * _N,
            sector=["T001081"] * _N,
        )

        cols = DataExtractor.derive_feature_columns(df, target_column="target")

        self.assertIn("feat1", cols)
        self.assertIn("y_feat_yearly", cols)
        self.assertNotIn("target", cols)
        self.assertNotIn("period_enddate", cols)
        self.assertNotIn("trend_index", cols)
        self.assertNotIn("covid_period", cols)
        self.assertNotIn("silver_id", cols)
        self.assertNotIn("sector", cols)
        self.assertNotIn("BedrijfskenmerkenSBI2008_T001081", cols)


if __name__ == "__main__":
    unittest.main()
