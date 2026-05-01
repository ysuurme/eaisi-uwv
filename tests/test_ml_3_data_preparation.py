"""
Unit tests for Step 3 — ML Data Preparation.

Step 1 (DataExtractor) now always delivers exactly one row per unique quarter
date (either all-industry aggregate or a single-sector filtered series).
Step 3 therefore receives clean 1-row-per-quarter data and performs a simple
temporal split.
"""
import unittest
import pandas as pd
from src.ml_engineering.ml_3_data_preparation import DataPreparator


class TestDataPreparation(unittest.TestCase):

    def _make_quarterly_df(self, n_periods: int = 32) -> pd.DataFrame:
        """Creates a minimal 1-row-per-quarter DataFrame (post-Step-1 format)."""
        dates = pd.date_range("2015-01-01", periods=n_periods, freq="QE")
        return pd.DataFrame({
            "period_enddate": dates,
            "target": [4.0 + 0.1 * i for i in range(n_periods)],
            "feat1": [float(i) for i in range(n_periods)],
            "quarter": [d.quarter for d in dates],
            "year": [d.year for d in dates],
        })

    def test_prepare_splits_correctly(self):
        df = self._make_quarterly_df(n_periods=32)
        x_train, x_test, y_train, y_test, lineage = DataPreparator.prepare(
            df, target_column="target", n_test=8
        )
        # Total rows = 32 unique quarters; split by date
        self.assertEqual(len(y_train) + len(y_test), 32)
        # Test set: last 8 quarters
        self.assertEqual(len(y_test), 8)
        # Train: remaining 24 quarters
        self.assertEqual(len(y_train), 24)
        # Index is DatetimeIndex
        self.assertIsInstance(y_train.index, pd.DatetimeIndex)
        self.assertIsInstance(x_train.index, pd.DatetimeIndex)

    def test_prepare_feature_count(self):
        df = self._make_quarterly_df(n_periods=32)
        x_train, _, _, _, lineage = DataPreparator.prepare(
            df, target_column="target", n_test=8
        )
        # Numeric features: feat1, quarter, year → 3
        self.assertEqual(lineage["feature_count"], 3)
        self.assertNotIn("period_enddate", x_train.columns)
        self.assertNotIn("target", x_train.columns)
        # feat1 is present
        self.assertIn("feat1", x_train.columns)

    def test_prepare_lineage_keys(self):
        df = self._make_quarterly_df(n_periods=32)
        _, _, _, _, lineage = DataPreparator.prepare(
            df, target_column="target", n_test=8
        )
        for key in ("target", "feature_count", "train_size", "test_size", "cutoff_date"):
            self.assertIn(key, lineage)

    def test_prepare_raises_when_too_few_dates(self):
        """Should raise ValueError when dataset has fewer unique dates than n_test."""
        df = self._make_quarterly_df(n_periods=5)  # only 5 unique dates
        with self.assertRaises(ValueError):
            DataPreparator.prepare(df, target_column="target", n_test=8)

    def test_temporal_ordering_preserved(self):
        """Train dates must all precede test dates."""
        df = self._make_quarterly_df(n_periods=32)
        x_train, x_test, y_train, y_test, _ = DataPreparator.prepare(
            df, target_column="target", n_test=8
        )
        self.assertLess(y_train.index.max(), y_test.index.min())


if __name__ == "__main__":
    unittest.main()
