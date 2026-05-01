"""
Unit tests for Step 3 — ML Data Preparation.
"""

import unittest
import pandas as pd
from src.ml_engineering.ml_3_data_preparation import DataPreparator


class TestDataPreparation(unittest.TestCase):
    def test_prepare_splits_correctly(self):
        df = pd.DataFrame({
            "period_enddate": pd.date_range("2020-01-01", periods=10, freq="ME"),
            "target": [0.1 * i for i in range(10)],
            "feat1": [float(i) for i in range(10)],
        })
        
        x_train, x_test, y_train, y_test, lineage = DataPreparator.prepare(
            df, target_column="target", n_splits=2
        )
        
        self.assertEqual(len(x_train) + len(x_test), 10)
        self.assertEqual(lineage["feature_count"], 1)
        self.assertEqual(lineage["target"], "target")
        self.assertEqual(x_train.shape[1], 1)
        self.assertNotIn("period_enddate", x_train.columns)


if __name__ == "__main__":
    unittest.main()
