"""
Unit tests for Step 1 — ML Data Extraction.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
from src.ml_engineering.ml_1_data_extraction import DataExtractor


class TestDataExtraction(unittest.TestCase):
    @patch("src.ml_engineering.ml_1_data_extraction.create_engine")
    @patch("src.ml_engineering.ml_1_data_extraction.pd.read_sql_table")
    def test_extract_subset(self, mock_read_sql, mock_engine):
        mock_df = pd.DataFrame({
            "period_enddate": pd.date_range("2020-01-01", periods=5, freq="ME"),
            "target": [1, 2, 3, 4, 5],
            "feat1": [10, 20, 30, 40, 50],
            "feat2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "silver_id": range(5)
        })
        mock_read_sql.return_value = mock_df
        
        extractor = DataExtractor(Path("fake.db"), "gold_table")
        extracted_df = extractor.extract(target_column="target", features=["feat1"])
        
        self.assertIn("feat1", extracted_df.columns)
        self.assertIn("target", extracted_df.columns)
        self.assertNotIn("feat2", extracted_df.columns)
        self.assertNotIn("silver_id", extracted_df.columns)

    @patch("src.ml_engineering.ml_1_data_extraction.create_engine")
    @patch("src.ml_engineering.ml_1_data_extraction.pd.read_sql_table")
    def test_extract_all_numeric(self, mock_read_sql, mock_engine):
        mock_df = pd.DataFrame({
            "period_enddate": pd.date_range("2020-01-01", periods=5, freq="ME"),
            "target": [1, 2, 3, 4, 5],
            "feat1": [10, 20, 30, 40, 50],
            "silver_id": range(5)
        })
        mock_read_sql.return_value = mock_df
        
        extractor = DataExtractor(Path("fake.db"), "gold_table")
        extracted_df = extractor.extract(target_column="target", features=None)
        
        self.assertIn("feat1", extracted_df.columns)
        self.assertNotIn("silver_id", extracted_df.columns)


if __name__ == "__main__":
    unittest.main()
