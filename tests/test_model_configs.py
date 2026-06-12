"""
Unit tests for model_configs — custom estimators + the experiment catalog.

Covers ``QuarterlyPeriodForecaster`` (the sktime adapter that runs
decomposition-based forecasters on the pipeline's freq=None quarterly index by
converting to a PeriodIndex internally) and the statistical catalog entries
(``autoets`` / ``stl_ets``).  The adapter must remain a fully sklearn/sktime-
compatible estimator (clone, pickle, nested param routing for grid search).
"""
import pickle
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.trend import STLForecaster

from src.ml_engineering.model_configs import ModelConfiguration, QuarterlyPeriodForecaster


def _quarterly_series(n: int = 40) -> tuple:
    """(y, X) on a freq=None quarterly DatetimeIndex, like Step 3 produces."""
    idx = pd.DatetimeIndex(pd.date_range("2012-03-31", periods=n, freq="QE"), freq=None)
    rng = np.random.default_rng(11)
    y = pd.Series(
        4.0 + 0.02 * np.arange(n) + 0.5 * np.sin(np.arange(n) * np.pi / 2)
        + rng.normal(0, 0.05, n),
        index=idx,
    )
    X = pd.DataFrame(
        {"quarter": idx.quarter.astype(float), "trend_index": np.arange(1.0, n + 1)},
        index=idx,
    )
    return y, X


class TestQuarterlyPeriodForecaster(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings("ignore")
        self.y, self.X = _quarterly_series()
        self.fh = ForecastingHorizon([1, 2, 3, 4], is_relative=True)

    def test_fit_predict_returns_quarter_end_timestamps(self):
        """STLForecaster fails on freq=None directly; through the adapter it
        must fit and return 4 predictions on quarter-end timestamps."""
        forecaster = QuarterlyPeriodForecaster(STLForecaster(sp=4))
        forecaster.fit(y=self.y, X=self.X)
        y_pred = forecaster.predict(fh=self.fh, X=self.X.iloc[:4])

        self.assertEqual(len(y_pred), 4)
        expected = pd.DatetimeIndex(
            ["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31"]
        )
        self.assertTrue((pd.DatetimeIndex(y_pred.index) == expected).all())
        self.assertTrue(np.isfinite(y_pred.to_numpy()).all())

    def test_clone_roundtrip(self):
        forecaster = QuarterlyPeriodForecaster(STLForecaster(sp=4))
        cloned = clone(forecaster)
        cloned.fit(y=self.y)
        y_pred = cloned.predict(fh=self.fh)
        self.assertEqual(len(y_pred), 4)

    def test_pickle_roundtrip(self):
        forecaster = QuarterlyPeriodForecaster(STLForecaster(sp=4))
        forecaster.fit(y=self.y)
        restored = pickle.loads(pickle.dumps(forecaster))
        y_pred = restored.predict(fh=self.fh)
        self.assertEqual(len(y_pred), 4)

    def test_nested_param_routing(self):
        """Grid-search keys (forecaster__<param>) must route to the wrapped model."""
        forecaster = QuarterlyPeriodForecaster(STLForecaster(sp=4, seasonal=7))
        forecaster.set_params(forecaster__seasonal=13)
        self.assertEqual(forecaster.get_params()["forecaster__seasonal"], 13)


class TestStatCatalogEntries(unittest.TestCase):

    def test_autoets_entry(self):
        config = ModelConfiguration.get("autoets")
        self.assertEqual(config.name, "AutoETS_Stat")
        self.assertEqual(config.feature_groups, ["structural_only"])
        # List-of-dicts grid: only valid ETS combos
        self.assertIsInstance(config.param_grid, list)
        n_combos = sum(
            int(np.prod([len(v) for v in block.values()]))
            for block in config.param_grid
        )
        self.assertEqual(n_combos, 11)

    def test_stl_ets_entry(self):
        config = ModelConfiguration.get("stl_ets")
        self.assertEqual(config.name, "STLETS_Stat")
        self.assertIsInstance(config.estimator, QuarterlyPeriodForecaster)
        self.assertEqual(config.feature_groups, ["structural_only"])

    def test_param_grid_isolation_for_list_grids(self):
        """get() must deepcopy list-of-dicts grids — mutations cannot leak back."""
        first = ModelConfiguration.get("autoets")
        first.param_grid[0]["error"].append("mutated")
        fresh = ModelConfiguration.get("autoets")
        self.assertNotIn("mutated", fresh.param_grid[0]["error"])


if __name__ == "__main__":
    unittest.main()
