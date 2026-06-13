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
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import ExpandingWindowSplitter, ForecastingGridSearchCV
from sktime.forecasting.trend import STLForecaster
from sktime.performance_metrics.forecasting import MeanSquaredError

from src.ml_engineering.model_configs import (
    Base,
    ChronosForecaster,
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


class TestDeseasonChallenger(unittest.TestCase):
    """The deseasonalized multivariate feature-ML model (``ridge_deseason``).

    Must retrieve with a bounded grid and fit/predict/clone/pickle on the
    pipeline's freq=None quarterly index, and tune through ForecastingGridSearchCV.
    """

    def setUp(self):
        warnings.filterwarnings("ignore")
        self.y, self.X = _quarterly_series(48)
        self.fh = ForecastingHorizon([1, 2, 3, 4], is_relative=True)

    def test_catalog_entry(self):
        config = ModelConfiguration.get("ridge_deseason")
        self.assertEqual(config.name, "RidgeDeseason_Reduced")
        self.assertEqual(config.feature_groups, ["all_survivors"])
        combos = int(np.prod([len(v) for v in config.param_grid.values()]))
        self.assertLessEqual(combos, 16)

    def test_wrapper_fit_predict_clone_pickle(self):
        est = ModelConfiguration.get("ridge_deseason").estimator
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

    def test_param_grid_survives_grid_search(self):
        cv = ExpandingWindowSplitter(initial_window=40, step_length=4, fh=[1, 2, 3, 4])
        config = ModelConfiguration.get("ridge_deseason")
        grid = ForecastingGridSearchCV(
            forecaster=config.estimator, param_grid=config.param_grid, cv=cv,
            scoring=MeanSquaredError(square_root=False),
        )
        grid.fit(y=self.y, X=self.X, fh=self.fh)
        self.assertEqual(len(grid.predict(fh=self.fh, X=self.X.iloc[:4])), 4)


class _FakeChronosPipeline:
    """Stand-in for BaseChronosPipeline — 'forecasts' the last value repeated,
    so the wrapper can be tested without downloading model weights."""

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        import torch
        last = float(inputs[-1])
        q = torch.full((1, prediction_length, len(quantile_levels)), last)
        return q, q[:, :, 0]


class TestChronosForecaster(unittest.TestCase):
    """Zero-shot Chronos-Bolt wrapper — catalog entry + fit/predict (mocked
    pipeline, no weight download) + clone/pickle without the heavy model."""

    def setUp(self):
        warnings.filterwarnings("ignore")
        self.y, self.X = _quarterly_series(40)
        self.fh = ForecastingHorizon([1, 2, 3, 4], is_relative=True)

    def test_catalog_entry(self):
        c = ModelConfiguration.get("chronos_bolt")
        self.assertEqual(c.name, "ChronosBolt_Stat")
        self.assertEqual(c.feature_groups, ["structural_only"])
        self.assertEqual(c.param_grid, {})  # zero-shot — nothing to tune
        self.assertIsInstance(c.estimator, ChronosForecaster)
        self.assertEqual(c.estimator.model_id, "amazon/chronos-bolt-base")

    def test_params_clone_pickle_without_weights(self):
        est = ChronosForecaster(model_id="amazon/chronos-bolt-base", device="cpu")
        self.assertEqual(est.get_params()["model_id"], "amazon/chronos-bolt-base")
        clone(est)                       # sklearn-cloneable (params only)
        pickle.loads(pickle.dumps(est))  # picklable without the multi-hundred-MB model

    @patch("src.ml_engineering.model_configs._load_chronos")
    def test_fit_predict_median_and_ignores_X(self, mock_load):
        mock_load.return_value = _FakeChronosPipeline()
        est = ChronosForecaster()
        est.fit(y=self.y, X=self.X, fh=self.fh)      # zero-shot: just stores context
        pred = est.predict(fh=self.fh, X=self.X.iloc[:4])

        self.assertEqual(len(pred), 4)
        # fake returns the last observed value as the median for every step
        self.assertTrue(np.allclose(pred.to_numpy(), float(self.y.iloc[-1]), atol=1e-3))
        expected = pd.DatetimeIndex(["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31"])
        self.assertTrue((pd.DatetimeIndex(pred.index) == expected).all())
        self.assertTrue(est.get_tag("ignores-exogeneous-X"))
        mock_load.assert_called()  # pipeline loaded lazily at predict (not import)


class TestCuratedCatalog(unittest.TestCase):
    """The catalog is the curated comparison set (no clutter)."""

    def test_exactly_the_curated_keys(self):
        self.assertEqual(
            set(ModelConfiguration._CATALOG),
            {"baseline", "autoets", "stl_ets", "chronos_bolt",
             "ridge", "random_forest", "ridge_deseason"},
        )

    def test_multivariate_models_use_selected_features(self):
        for key in ("ridge", "random_forest", "ridge_deseason"):
            self.assertEqual(ModelConfiguration.get(key).feature_groups, ["all_survivors"])


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
