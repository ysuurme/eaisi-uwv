import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_engineering.data_loader_gold import DatabaseGold, apply_gold_baseline, transform_generic_feature_table

class TestDataLoaderGold(unittest.TestCase):
    @patch("src.data_engineering.data_loader_gold.create_engine")
    def setUp(self, mock_engine):
        self.silver_path = Path("fake_silver.db")
        self.gold_path = Path("fake_gold.db")
        self.gold = DatabaseGold(self.silver_path, self.gold_path)

    @patch("src.data_engineering.data_loader_gold.pd.read_sql_query")
    @patch("src.data_engineering.data_loader_gold.pd.DataFrame.to_sql")
    @patch("src.data_engineering.data_loader_gold.text")
    def test_process_silver_table_success(self, mock_text, mock_to_sql, mock_read_sql):
        mock_read_sql.return_value = pd.DataFrame({"feat": [1]})
        
        # DataFrame without NaNs
        mock_transform = MagicMock(return_value=pd.DataFrame({"feat_gold": [1]}))
        
        mock_conn = MagicMock()
        self.gold.engine.connect.return_value.__enter__.return_value = mock_conn

        self.gold.process_silver_table("80072ned", mock_transform)
        
        mock_transform.assert_called_once()
        mock_to_sql.assert_called_once()

    def test_apply_gold_baseline(self):
        # Test temporal parsing and column dropping
        df = pd.DataFrame({
            "Perioden": ["2020KW01", "2020KW02"],
            "bronze_pk": ["1", "2"],
            "Desc_Description": ["a", "b"],
            "Metric": [1, 2]
        })
        
        df_out = apply_gold_baseline(df)
        
        # Assertions
        self.assertTrue("period_enddate" in df_out.columns)
        self.assertEqual(len(df_out), 2)
        self.assertFalse("bronze_pk" in df_out.columns)
        self.assertFalse("Desc_Description" in df_out.columns)
        self.assertTrue("Metric" in df_out.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_out["period_enddate"]))

    def test_apply_gold_baseline_extrapolation(self):
        # Test Yearly extrapolation
        df = pd.DataFrame({
            "Perioden": ["2020JJ00"],
            "Metric": [10]
        })
        
        df_out = apply_gold_baseline(df)
        
        # Assertions: Should expand 1 yearly row into 4 quarters
        self.assertEqual(len(df_out), 4)
        self.assertEqual(df_out["Metric"].sum(), 40)
        self.assertEqual(df_out["period_enddate"].dt.month.tolist(), [3, 6, 9, 12])

    def test_transform_generic_feature_table(self):
        df = pd.DataFrame({
            "Perioden": ["2020KW01", "2020KW01"],
            "Dimension": ["Branch_A", "T001081"],  # 'T00' is total, should be filtered
            "Metric": [10.0, 50.0]
        })
        
        df_out = transform_generic_feature_table(df)
        
        # T001081 is dropped. We are left with Branch_A.
        # Pivot occurs: index=period_enddate, columns=Dimension (Branch_A), values=Metric
        self.assertEqual(len(df_out), 1)
        # Flattened feature name gracefully maps Metric + Dimension together natively
        self.assertTrue("Metric_Branch_A" in df_out.columns)
        self.assertEqual(df_out["Metric_Branch_A"].iloc[0], 10.0)

if __name__ == "__main__":
    unittest.main()
