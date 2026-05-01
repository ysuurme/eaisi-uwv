"""
Unit tests for Step 1 — ML Data Extraction.
"""

import unittest
from unittest.mock import patch, MagicMock
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


class TestDataExtraction(unittest.TestCase):
    @patch("src.ml_engineering.ml_1_data_extraction.create_engine")
    @patch("src.ml_engineering.ml_1_data_extraction.pd.read_sql_table")
    @patch("src.ml_engineering.ml_1_data_extraction.FEATURE_CATALOG", _MOCK_CATALOG)
    def test_extract_by_group(self, mock_read_sql, mock_engine):
        """Groups mode: only columns declared in the resolved group are kept."""
        mock_df = pd.DataFrame({
            "period_enddate": pd.date_range("2020-01-01", periods=5, freq="ME"),
            "target": [1, 2, 3, 4, 5],
            "feat1": [10, 20, 30, 40, 50],
            "feat2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "silver_id": range(5),
        })
        mock_read_sql.return_value = mock_df

        extractor = DataExtractor(Path("fake.db"), "gold_table")
        extracted_df = extractor.extract(target_column="target", feature_groups=["test_group"])

        self.assertIn("feat1", extracted_df.columns)
        self.assertIn("target", extracted_df.columns)
        self.assertNotIn("feat2", extracted_df.columns)
        self.assertNotIn("silver_id", extracted_df.columns)

    @patch("src.ml_engineering.ml_1_data_extraction.create_engine")
    @patch("src.ml_engineering.ml_1_data_extraction.pd.read_sql_table")
    def test_extract_discovery_mode(self, mock_read_sql, mock_engine):
        """Discovery mode: all columns except target and structural keys are kept."""
        mock_df = pd.DataFrame({
            "period_enddate": pd.date_range("2020-01-01", periods=5, freq="ME"),
            "target": [1, 2, 3, 4, 5],
            "feat1": [10, 20, 30, 40, 50],
            "silver_id": range(5),
        })
        mock_read_sql.return_value = mock_df

        extractor = DataExtractor(Path("fake.db"), "gold_table")
        extracted_df = extractor.extract(target_column="target", feature_groups=None)

        self.assertIn("feat1", extracted_df.columns)
        self.assertNotIn("silver_id", extracted_df.columns)

    @patch("src.ml_engineering.ml_1_data_extraction.create_engine")
    @patch("src.ml_engineering.ml_1_data_extraction.pd.read_sql_table")
    @patch("src.ml_engineering.ml_1_data_extraction.FEATURE_CATALOG", _MOCK_CATALOG)
    def test_extract_unknown_group_logs_warning(self, mock_read_sql, mock_engine):
        """Unknown group name logs a warning and is skipped — no crash."""
        mock_df = pd.DataFrame({
            "period_enddate": pd.date_range("2020-01-01", periods=3, freq="ME"),
            "target": [1, 2, 3],
            "feat1": [10, 20, 30],
        })
        mock_read_sql.return_value = mock_df

        extractor = DataExtractor(Path("fake.db"), "gold_table")
        extracted_df = extractor.extract(target_column="target", feature_groups=["nonexistent_group"])

        self.assertIn("target", extracted_df.columns)

    @patch("src.ml_engineering.ml_1_data_extraction.create_engine")
    @patch("src.ml_engineering.ml_1_data_extraction.pd.read_sql_table")
    @patch("src.ml_engineering.ml_1_data_extraction.FEATURE_CATALOG", _MOCK_CATALOG)
    def test_extract_missing_column_in_group_logs_warning(self, mock_read_sql, mock_engine):
        """Column declared in a group but absent from the table is skipped — no crash."""
        mock_df = pd.DataFrame({
            "period_enddate": pd.date_range("2020-01-01", periods=3, freq="ME"),
            "target": [1, 2, 3],
            # feat1 intentionally absent
        })
        mock_read_sql.return_value = mock_df

        extractor = DataExtractor(Path("fake.db"), "gold_table")
        extracted_df = extractor.extract(target_column="target", feature_groups=["test_group"])

        self.assertNotIn("feat1", extracted_df.columns)
        self.assertIn("target", extracted_df.columns)


if __name__ == "__main__":
    unittest.main()
