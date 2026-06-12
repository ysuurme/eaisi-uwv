"""
Unit tests for Step 7 — ML Model Inference (forward forecast production).

ml_7_model_inference loads each sector's ``@prod`` champion, rebuilds it from
its MLflow lineage, refits on the FULL observed history (so the forecast is
anchored at the latest quarter, not the train-split end), and forecasts the
next 4 quarters via the shared ``build_future_x`` helper.

These tests exercise the pure, MLflow-free seams:
* sector ↔ SBI-filter mapping and the registry-name prefix,
* JSON-param coercion + estimator rebuild from the catalog,
* the core history → forecast frame (both estimator dispatch branches),
* champion-lineage parsing from a mocked MLflow run.
The end-to-end registry/gold-DB path is covered by the week-plan smoke runs.
"""
import json
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from sktime.forecasting.naive import NaiveForecaster

from src.ml_engineering import ml_7_model_inference as ml7
from src.ml_engineering.model_configs import SectorQuarterRollingMean


def _quarterly_history(start: str = "2019-03-31", periods: int = 24) -> tuple:
    """Synthetic single-sector quarterly history (X with structural cols, y)."""
    idx = pd.DatetimeIndex(pd.date_range(start, periods=periods, freq="QE"), freq=None)
    x = pd.DataFrame(
        {
            "year": idx.year.astype(float),
            "quarter": idx.quarter.astype(float),
            "trend_index": np.arange(1.0, periods + 1.0),
            "post_covid": (idx > pd.Timestamp("2022-12-31")).astype(float),
            "exog": np.linspace(10.0, 20.0, periods),
        },
        index=idx,
    )
    # Seasonal-ish positive target so MAPE / rolling-mean lookups are well defined.
    y = pd.Series(5.0 + 0.5 * idx.quarter.to_numpy(dtype=float), index=idx, name="target")
    return x, y


class TestSectorMapping(unittest.TestCase):

    def test_national_total_maps_to_none(self):
        self.assertIsNone(ml7._sector_to_sbi_filter("T001081"))

    def test_sector_maps_to_ohe_column(self):
        self.assertEqual(
            ml7._sector_to_sbi_filter("301000"),
            "BedrijfskenmerkenSBI2008_301000",
        )

    def test_experiment_prefix_matches_orchestrator_naming(self):
        self.assertEqual(
            ml7._experiment_prefix("master_data_ml_preprocessed"),
            "master_SickLeave_4Q_",
        )


class TestParamCoercion(unittest.TestCase):

    def test_numeric_strings_coerced(self):
        self.assertEqual(ml7._coerce_param("8"), 8)
        self.assertIsInstance(ml7._coerce_param("8"), int)
        self.assertEqual(ml7._coerce_param("1.0"), 1.0)
        self.assertIsInstance(ml7._coerce_param("1.0"), float)

    def test_bool_and_none_strings_coerced(self):
        self.assertIs(ml7._coerce_param("True"), True)
        self.assertIs(ml7._coerce_param("False"), False)
        self.assertIsNone(ml7._coerce_param("None"))

    def test_non_numeric_string_preserved(self):
        self.assertEqual(ml7._coerce_param("add"), "add")

    def test_native_values_passthrough(self):
        self.assertEqual(ml7._coerce_param(8), 8)
        self.assertIsNone(ml7._coerce_param(None))


class TestRebuildEstimator(unittest.TestCase):

    def test_baseline_rebuilds_to_rolling_mean(self):
        estimator, config = ml7._rebuild_estimator("baseline", {})
        self.assertIsInstance(estimator, SectorQuarterRollingMean)
        self.assertEqual(config.name, "SectorQuarterRollingMean")

    def test_best_params_applied_with_coercion(self):
        # window_length arrives JSON-stringified (numpy int -> "8") and must be
        # coerced + applied so the rebuilt estimator matches the tuned champion.
        estimator, _ = ml7._rebuild_estimator("ridge", {"window_length": "8"})
        self.assertEqual(estimator.get_params()["window_length"], 8)


class TestForecastFromHistory(unittest.TestCase):

    def test_sklearn_baseline_branch(self):
        x, y = _quarterly_history()
        frame = ml7._forecast_from_history(
            SectorQuarterRollingMean(n_years=3),
            x, y,
            sector_code="T001081", model_family="SectorQuarterRollingMean",
            model_type="SectorQuarterRollingMean", experiment_key="baseline",
            champion_version="1", n_steps=4,
        )
        self.assertEqual(len(frame), 4)
        self.assertEqual(list(frame["horizon"]), [1, 2, 3, 4])
        expected_dates = pd.DatetimeIndex(
            ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]
        )
        self.assertTrue((pd.DatetimeIndex(frame["target_date"]) == expected_dates).all())
        self.assertTrue(np.isfinite(frame["y_pred"]).all())
        self.assertTrue((frame["sector_code"] == "T001081").all())

    def test_sktime_univariate_branch(self):
        x, y = _quarterly_history()
        # Non-seasonal last: works on the pipeline's freq-less index (seasonal
        # sp=4 would need a freq, which is why stat models are wrapped in
        # QuarterlyPeriodForecaster).  Exercises the sktime branch of _predict_origin.
        frame = ml7._forecast_from_history(
            NaiveForecaster(strategy="last"),
            x, y,
            sector_code="301000", model_family="Naive", model_type="NaiveForecaster",
            experiment_key="baseline", champion_version="2", n_steps=4,
        )
        self.assertEqual(len(frame), 4)
        self.assertTrue(np.isfinite(frame["y_pred"]).all())
        # origin is the last observed quarter; "last" repeats its value forward
        self.assertEqual(frame["origin_date"].iloc[0], pd.Timestamp("2024-12-31"))
        self.assertTrue((frame["y_pred"] == y.iloc[-1]).all())


class TestReadChampionLineage(unittest.TestCase):

    def _mock_client(self, params, tags):
        client = MagicMock()
        mv = MagicMock()
        mv.version = "7"
        mv.run_id = "abc123"
        mv.tags = tags
        client.get_model_version_by_alias.return_value = mv
        run = MagicMock()
        run.data.params = params
        client.get_run.return_value = run
        return client

    def test_lineage_parsed_from_run_and_tags(self):
        client = self._mock_client(
            params={
                "experiment_key": "ridge",
                "best_params": json.dumps({"window_length": "8"}),
            },
            tags={"model_family": "Ridge_Reduced", "model_type": "Ridge"},
        )
        lineage = ml7._read_champion_lineage(
            client, "master_SickLeave_4Q_T001081", "T001081",
        )
        self.assertEqual(lineage.sector_code, "T001081")
        self.assertEqual(lineage.experiment_key, "ridge")
        self.assertEqual(lineage.model_family, "Ridge_Reduced")
        self.assertEqual(lineage.model_type, "Ridge")
        self.assertEqual(lineage.version, "7")
        self.assertEqual(lineage.best_params, {"window_length": "8"})

    def test_missing_best_params_yields_empty_dict(self):
        client = self._mock_client(
            params={"experiment_key": "baseline"}, tags={"model_family": "X"},
        )
        lineage = ml7._read_champion_lineage(
            client, "master_SickLeave_4Q_301000", "301000",
        )
        self.assertEqual(lineage.best_params, {})
        self.assertEqual(lineage.experiment_key, "baseline")


if __name__ == "__main__":
    unittest.main()
