"""
Unit tests for Step 2 — ML Data Validation.
"""

import unittest
import numpy as np
import pandas as pd
from src.ml_engineering.ml_2_data_validation import DataValidator


def _make_synthetic_df(include_nans: bool = True) -> pd.DataFrame:
    """Build a small synthetic DataFrame."""
    data = {
        "period_enddate": pd.to_datetime([
            "2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31",
        ]),
        "target": [0.1, 0.2, 0.3, 0.4],
        "feature1": [1.0, 2.0, np.nan if include_nans else 3.0, 4.0],
    }
    return pd.DataFrame(data)


class TestDataValidation(unittest.TestCase):
    """Tests for DataValidator.validate()."""

    def test_validate_pre_prep_passes(self):
        """Pre-prep validation passes even with NaNs."""
        df = _make_synthetic_df(include_nans=True)
        returned = DataValidator.validate(df, target_column="target", stage="pre_prep")
        self.assertIs(returned, df)

    def test_validate_post_prep_fails_on_nan(self):
        """Post-prep validation fails if NaNs remain."""
        df = _make_synthetic_df(include_nans=True)
        with self.assertRaises(ValueError) as ctx:
            DataValidator.validate(df, target_column="target", stage="post_prep")
        self.assertIn("NaN", str(ctx.exception))

    def test_validate_post_prep_passes_clean(self):
        """Post-prep validation passes with clean data."""
        df = _make_synthetic_df(include_nans=False)
        returned = DataValidator.validate(df, target_column="target", stage="post_prep")
        self.assertIs(returned, df)

    def test_missing_target_fails(self):
        """Fails if target column is missing."""
        df = _make_synthetic_df()
        with self.assertRaises(ValueError) as ctx:
            DataValidator.validate(df, target_column="wrong_target")
        self.assertIn("wrong_target", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
