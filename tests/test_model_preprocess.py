"""
Unit tests for src/ml_engineering/model_preprocess.py.

All tests use synthetic DataFrames — no database access required.
"""

import unittest

import numpy as np
import pandas as pd

from src.ml_engineering.model_preprocess import (
    validate_master_dataset,
    impute_missing_values,
    _identify_ohe_columns,
)


def _make_synthetic_df(include_nans: bool = True) -> pd.DataFrame:
    """Build a small synthetic DataFrame mimicking master_data_ml_joined."""
    data = {
        "period_enddate": pd.to_datetime([
            "2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31",
            "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31",
            "2021-03-31", "2021-06-30",
        ]),
        "BedrijfstakkenBranchesSBI2008": [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J"
        ],
        "stress_rate": [1.2, 2.3, np.nan, 4.1, 3.5, np.nan, 2.9, 1.1, 0.8, 3.3],
        "employment_pct": [65.0, 70.0, 68.0, np.nan, 72.0, 74.0, np.nan, 69.0, 71.0, 73.0],
        "SBI_flag_A": [1.0, 0.0, 0.0, 1.0, np.nan, 0.0, 1.0, 0.0, 0.0, 1.0],
        "SBI_flag_B": [0.0, 1.0, np.nan, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    }
    df = pd.DataFrame(data)
    if not include_nans:
        df = df.fillna(0.0)
    return df


def _make_numeric_only_df(include_nans: bool = True) -> pd.DataFrame:
    """Build a fully numeric synthetic DataFrame (no string columns)."""
    df = _make_synthetic_df(include_nans=include_nans)
    df = df.drop(columns=["BedrijfstakkenBranchesSBI2008"])
    return df


class TestImputation(unittest.TestCase):
    """Tests for impute_missing_values()."""

    def test_impute_numeric_median(self) -> None:
        """Numeric NaN values are filled with the column median."""
        df = _make_numeric_only_df(include_nans=True)
        result = impute_missing_values(df, add_missing_indicator=False)

        # stress_rate: non-NaN values [1.2, 2.3, 4.1, 3.5, 2.9, 1.1, 0.8, 3.3] → median = 2.6
        stress_median = pd.Series([1.2, 2.3, 4.1, 3.5, 2.9, 1.1, 0.8, 3.3]).median()
        self.assertAlmostEqual(result["stress_rate"].iloc[2], stress_median, places=5)

    def test_impute_ohe_zeros(self) -> None:
        """Binary OHE flag NaN values are filled with 0."""
        df = _make_numeric_only_df(include_nans=True)
        result = impute_missing_values(df, add_missing_indicator=False)

        # SBI_flag_A had NaN at index 4 → should be 0.0
        self.assertEqual(result["SBI_flag_A"].iloc[4], 0.0)
        # SBI_flag_B had NaN at index 2 → should be 0.0
        self.assertEqual(result["SBI_flag_B"].iloc[2], 0.0)

    def test_missing_indicator_columns(self) -> None:
        """Missing-indicator columns are created for columns that had NaNs."""
        df = _make_numeric_only_df(include_nans=True)
        result = impute_missing_values(df, add_missing_indicator=True)

        expected_indicators = [
            "stress_rate_is_missing",
            "employment_pct_is_missing",
            "SBI_flag_A_is_missing",
            "SBI_flag_B_is_missing",
        ]
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns)

        # stress_rate had NaN at index 2 → indicator should be 1.0
        self.assertEqual(result["stress_rate_is_missing"].iloc[2], 1.0)
        # stress_rate was NOT NaN at index 0 → indicator should be 0.0
        self.assertEqual(result["stress_rate_is_missing"].iloc[0], 0.0)

    def test_no_nan_after_impute(self) -> None:
        """Output DataFrame has zero NaN values."""
        df = _make_numeric_only_df(include_nans=True)
        result = impute_missing_values(df)
        self.assertEqual(result.isna().sum().sum(), 0)

    def test_all_float64_after_impute(self) -> None:
        """All feature output columns are float64."""
        df = _make_numeric_only_df(include_nans=True)
        result = impute_missing_values(df)
        from src.ml_engineering.model_preprocess import DATE_COL, SBI_COL
        for col in result.columns:
            if col not in [DATE_COL, SBI_COL]:
                self.assertEqual(result[col].dtype, np.float64, f"Column '{col}' is not float64")


class TestValidation(unittest.TestCase):
    """Tests for validate_master_dataset()."""

    def test_validate_raw_passes(self) -> None:
        """Raw validation passes on a valid synthetic DataFrame with NaNs."""
        df = _make_numeric_only_df(include_nans=True)
        # Should return without raising
        returned = validate_master_dataset(df, stage="raw")
        self.assertIs(returned, df)

    def test_validate_clean_catches_nan(self) -> None:
        """Clean validation raises ValueError if NaNs remain."""
        df = _make_numeric_only_df(include_nans=True)
        with self.assertRaises(ValueError) as ctx:
            validate_master_dataset(df, stage="clean")
        self.assertIn("NaN", str(ctx.exception))

    def test_validate_raw_allows_duplicates(self) -> None:
        """Validation soft-warns on date-only duplicates (SBI absent)."""
        df = _make_numeric_only_df(include_nans=False)
        df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        # Should NOT raise — no SBI column means date-only duplicates are expected
        returned = validate_master_dataset(df_dup, stage="raw")
        self.assertIsNotNone(returned)

    def test_validate_catches_composite_key_duplicates(self) -> None:
        """Validation raises ValueError when full composite key has duplicates."""
        df = _make_synthetic_df(include_nans=False)
        df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        # SBI column IS present → composite key duplicates are a real error
        with self.assertRaises(ValueError) as ctx:
            validate_master_dataset(df_dup, stage="clean")
        self.assertIn("duplicate", str(ctx.exception).lower())


class TestOHEDetection(unittest.TestCase):
    """Tests for the internal _identify_ohe_columns() helper."""

    def test_identifies_binary_columns(self) -> None:
        """Correctly identifies columns with values in {0, 1, NaN}."""
        df = _make_numeric_only_df(include_nans=True)
        ohe_cols = _identify_ohe_columns(df)
        self.assertIn("SBI_flag_A", ohe_cols)
        self.assertIn("SBI_flag_B", ohe_cols)
        self.assertNotIn("stress_rate", ohe_cols)
        self.assertNotIn("employment_pct", ohe_cols)


if __name__ == "__main__":
    unittest.main()
