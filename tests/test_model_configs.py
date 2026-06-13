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
from sktime.forecasting.model_selection import ExpandingWindowSplitter, ForecastingGridSearchCV
from sktime.forecasting.trend import STLForecaster
from sktime.performance_metrics.forecasting import MeanSquaredError

from src.ml_engineering.model_configs import (
    Base,
    ModelConfiguration,
    ModelForecastRecord,
    QuarterlyPeriodForecaster,
)


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


class TestPhase4Challengers(unittest.TestCase):
    """Deseasonalize/detrend + direct-strategy feature-ML challengers.

    Each new catalog key must retrieve with a bounded grid, and the wrapped
    estimators must fit/predict/clone/pickle on the pipeline's freq=None
    quarterly index and tune through ForecastingGridSearchCV (with fh at fit —
    mandatory for the direct-strategy reducer).
    """

    NEW_KEYS = {
        "ridge_deseason":            ("RidgeDeseason_Reduced", ["all_survivors"]),
        "ridge_deseason_detrend":    ("RidgeDeseasonDetrend_Reduced", ["all_survivors"]),
        "ridge_direct":              ("RidgeDirect_Reduced", ["all_survivors"]),
        "ridge_deseason_structural": ("RidgeDeseasonStructural_Reduced", ["structural_only"]),
        "ridge_deseason_labor":      ("RidgeDeseasonLabor_Reduced", ["labor_structure"]),
    }

    def setUp(self):
        warnings.filterwarnings("ignore")
        self.y, self.X = _quarterly_series(48)
        self.fh = ForecastingHorizon([1, 2, 3, 4], is_relative=True)

    @staticmethod
    def _combos(grid):
        if isinstance(grid, list):
            return sum(int(np.prod([len(v) for v in b.values()])) for b in grid)
        return int(np.prod([len(v) for v in grid.values()]))

    def test_catalog_retrieval_for_each_new_key(self):
        for key, (name, groups) in self.NEW_KEYS.items():
            config = ModelConfiguration.get(key)
            self.assertEqual(config.name, name, key)
            self.assertEqual(config.feature_groups, groups, key)
            self.assertLessEqual(self._combos(config.param_grid), 16, key)

    def test_deseason_wrapper_fit_predict_clone_pickle(self):
        config = ModelConfiguration.get("ridge_deseason")
        est = config.estimator
        self.assertIsInstance(est, QuarterlyPeriodForecaster)

        est.fit(y=self.y, X=self.X, fh=self.fh)
        pred = est.predict(fh=self.fh, X=self.X.iloc[:4])
        self.assertEqual(len(pred), 4)
        self.assertTrue(np.isfinite(pred.to_numpy()).all())
        # freq=None quarterly index in → normalized quarter-end timestamps out
        self.assertTrue(all(ts == ts.normalize() for ts in pd.DatetimeIndex(pred.index)))

        cloned = clone(est)
        cloned.fit(y=self.y, X=self.X, fh=self.fh)
        self.assertEqual(len(cloned.predict(fh=self.fh, X=self.X.iloc[:4])), 4)

        restored = pickle.loads(pickle.dumps(est))
        self.assertEqual(len(restored.predict(fh=self.fh, X=self.X.iloc[:4])), 4)

    def test_detrend_wrapper_fit_predict(self):
        est = ModelConfiguration.get("ridge_deseason_detrend").estimator
        est.fit(y=self.y, X=self.X, fh=self.fh)
        self.assertEqual(len(est.predict(fh=self.fh, X=self.X.iloc[:4])), 4)

    def test_direct_strategy_requires_and_accepts_fh(self):
        """ridge_direct uses strategy='direct' → fh is mandatory at fit time."""
        est = ModelConfiguration.get("ridge_direct").estimator
        self.assertTrue(est.get_tag("requires-fh-in-fit"))
        with self.assertRaises(ValueError):
            est.fit(y=self.y, X=self.X)  # no fh → sktime rejects direct fit
        est = ModelConfiguration.get("ridge_direct").estimator
        est.fit(y=self.y, X=self.X, fh=self.fh)
        self.assertEqual(len(est.predict(fh=self.fh, X=self.X.iloc[:4])), 4)

    def test_param_grid_survives_grid_search_with_fh(self):
        """The recursive-deseason and direct grids both tune through
        ForecastingGridSearchCV when fh is supplied at fit (mirrors ml_4)."""
        cv = ExpandingWindowSplitter(initial_window=40, step_length=4, fh=[1, 2, 3, 4])
        for key in ("ridge_deseason", "ridge_direct"):
            config = ModelConfiguration.get(key)
            grid = ForecastingGridSearchCV(
                forecaster=config.estimator, param_grid=config.param_grid, cv=cv,
                scoring=MeanSquaredError(square_root=False),
            )
            grid.fit(y=self.y, X=self.X, fh=self.fh)
            self.assertEqual(len(grid.predict(fh=self.fh, X=self.X.iloc[:4])), 4, key)


class TestModelForecastRecord(unittest.TestCase):
    """The forecast read-model is additive and does not rename existing tables."""

    def test_table_name_and_columns(self):
        self.assertEqual(ModelForecastRecord.__tablename__, "model_forecasts")
        cols = {c.name for c in ModelForecastRecord.__table__.columns}
        self.assertEqual(
            cols,
            {
                "id", "sector_code", "model_family", "model_type", "experiment_key",
                "champion_version", "forecast_made_on", "target_date", "horizon",
                "y_pred", "feature_catalog_hash", "generated_at",
            },
        )

    def test_existing_eval_tables_still_registered(self):
        """Adding model_forecasts must not drop/rename the established tables."""
        tables = set(Base.metadata.tables)
        self.assertTrue(
            {
                "model_forecasts", "model_predictions", "model_evaluations",
                "model_tuning_results", "sector_performance",
            }
            <= tables
        )


if __name__ == "__main__":
    unittest.main()
