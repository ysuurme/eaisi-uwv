"""
Unit tests for Step 5 — ML Model Evaluation.

Step 5 uses walk-forward (rolling-origin) evaluation: for each of
n_test_points // 4 origins it clones the estimator, refits on the expanding
training window, and forecasts 4 quarters ahead.  Metrics are aggregated
across all origins.
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.ml_engineering.model_configs import SectorQuarterRollingMean
from src.ml_engineering.ml_5_model_evaluation import (
    QUARTER_DAYS,
    ModelEvaluator,
    _build_eval_tables,
    build_future_x,
)


def _make_series(n: int, start: str = "2015-01-01") -> tuple:
    """Returns (X, y) with n quarterly rows and a quarter column."""
    idx = pd.date_range(start, periods=n, freq="QE")
    y = pd.Series([4.0 + 0.05 * i for i in range(n)], index=idx, name="target")
    X = pd.DataFrame(
        {
            "quarter": [d.quarter for d in idx],
            "year": [d.year for d in idx],
            "feat1": [float(i) for i in range(n)],
        },
        index=idx,
    )
    return X, y


class TestModelEvaluation(unittest.TestCase):

    def setUp(self):
        # 32 training quarters + 8 test quarters (2 origins × 4Q = 8 eval points)
        self.x_train, self.y_train = _make_series(32)
        self.x_test,  self.y_test  = _make_series(8, start="2023-01-01")

    @patch("src.ml_engineering.ml_5_model_evaluation.mlflow")
    def test_evaluate_baseline_walk_forward(self, mock_mlflow):
        """Baseline (sklearn path): walk-forward produces metrics dict and stores record."""
        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

        baseline = SectorQuarterRollingMean()
        baseline.fit(self.x_train, self.y_train)

        mock_session = MagicMock()
        evaluator = ModelEvaluator(session=mock_session)
        metrics = evaluator.evaluate(
            run_id="run_baseline",
            fitted_model=baseline,
            x_train=self.x_train,
            y_train=self.y_train,
            x_test=self.x_test,
            y_test=self.y_test,
            model_name="test_baseline",
            n_test_points=8,  # 2 origins × 4Q
        )

        self.assertIn("r2", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIsInstance(metrics["r2"], float)
        mock_session.merge.assert_called_once()
        mock_mlflow.log_metrics.assert_called_once()

    @patch("src.ml_engineering.ml_5_model_evaluation.clone")
    @patch("src.ml_engineering.ml_5_model_evaluation.mlflow")
    def test_evaluate_sktime_walk_forward(self, mock_mlflow, mock_clone):
        """sktime forecaster path: clone + fit + predict called per origin."""
        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

        # mock_clone returns a fresh mock each call that predicts 4 constant values
        def make_mock_estimator():
            m = MagicMock()
            m.predict.return_value = pd.Series(
                [4.0, 4.1, 4.0, 4.1],
                index=pd.date_range("2023-01-01", periods=4, freq="QE"),
            )
            return m

        mock_clone.side_effect = lambda _: make_mock_estimator()

        mock_model = MagicMock()  # the "fitted" model from Step 4
        mock_session = MagicMock()
        evaluator = ModelEvaluator(session=mock_session)

        metrics = evaluator.evaluate(
            run_id="run_sktime",
            fitted_model=mock_model,
            x_train=self.x_train,
            y_train=self.y_train,
            x_test=self.x_test,
            y_test=self.y_test,
            model_name="sktime_model",
            n_test_points=8,  # 2 origins × 4Q
        )

        self.assertIn("r2", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        # clone and predict called once per origin (2 origins)
        self.assertEqual(mock_clone.call_count, 2)
        mock_session.merge.assert_called_once()

    @patch("src.ml_engineering.ml_5_model_evaluation.mlflow")
    def test_evaluate_returns_logs_and_persists_mape(self, mock_mlflow):
        """Tracer bullet: MAPE is computed on outer folds, returned by the
        public evaluate(), logged as an MLflow metric, and persisted on the
        evaluation record."""
        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

        baseline = SectorQuarterRollingMean()
        baseline.fit(self.x_train, self.y_train)

        mock_session = MagicMock()
        evaluator = ModelEvaluator(session=mock_session)
        metrics = evaluator.evaluate(
            run_id="run_mape",
            fitted_model=baseline,
            x_train=self.x_train,
            y_train=self.y_train,
            x_test=self.x_test,
            y_test=self.y_test,
            model_name="test_mape",
            n_test_points=8,
        )

        # (1) returned in the public metrics dict, finite, on outer folds
        self.assertIn("mape", metrics)
        self.assertIsInstance(metrics["mape"], float)
        self.assertTrue(np.isfinite(metrics["mape"]))

        # (2) logged to MLflow as a run metric
        logged = mock_mlflow.log_metrics.call_args[0][0]
        self.assertIn("mean_absolute_percentage_error", logged)
        self.assertEqual(logged["mean_absolute_percentage_error"], metrics["mape"])

        # (3) persisted on the ModelEvaluationRecord handed to session.merge
        persisted = mock_session.merge.call_args[0][0]
        self.assertTrue(hasattr(persisted, "mape"))
        self.assertIsInstance(persisted.mape, float)

    @patch("src.ml_engineering.ml_5_model_evaluation.mlflow")
    def test_eval_origin_counts_logged(self, mock_mlflow):
        """Walk-forward origin/prediction provenance counts are logged to MLflow.

        ml_5 logs the inner/outer breakdown (not a single n_eval_points key).
        With n_test_points=8 → 2 origins (1 inner + 1 outer) → 8 predictions.
        """
        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

        baseline = SectorQuarterRollingMean()
        baseline.fit(self.x_train, self.y_train)

        evaluator = ModelEvaluator(session=MagicMock())
        evaluator.evaluate(
            run_id="run_log_check",
            fitted_model=baseline,
            x_train=self.x_train,
            y_train=self.y_train,
            x_test=self.x_test,
            y_test=self.y_test,
            model_name="test",
            n_test_points=8,
        )

        logged = mock_mlflow.log_metrics.call_args[0][0]
        self.assertIn("n_inner_origins", logged)
        self.assertIn("n_outer_origins", logged)
        self.assertEqual(logged["n_inner_origins"] + logged["n_outer_origins"], 2)
        self.assertEqual(
            logged["n_inner_predictions"] + logged["n_outer_predictions"], 8
        )


class TestProductionHonestFutureX(unittest.TestCase):
    """x_future_mode wiring: forecast-window X must come from build_future_x
    by default (no future-covariate leakage) and from the actual rows only in
    the explicit diagnostic mode."""

    def setUp(self):
        self.x_train, self.y_train = _make_series(32)
        self.x_test,  self.y_test  = _make_series(8, start="2023-01-01")

    def _evaluate(self, mock_bfx, **kwargs):
        baseline = SectorQuarterRollingMean()
        baseline.fit(self.x_train, self.y_train)
        evaluator = ModelEvaluator(session=MagicMock())
        return evaluator.evaluate(
            run_id="run_future_x",
            fitted_model=baseline,
            x_train=self.x_train,
            y_train=self.y_train,
            x_test=self.x_test,
            y_test=self.y_test,
            model_name="test_future_x",
            n_test_points=8,  # 2 origins × 4Q
            **kwargs,
        )

    @patch("src.ml_engineering.ml_5_model_evaluation.build_future_x")
    @patch("src.ml_engineering.ml_5_model_evaluation.mlflow")
    def test_production_mode_constructs_future_x(self, mock_mlflow, mock_bfx):
        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
        # Shape-compatible stand-in: last 4 observed rows
        mock_bfx.side_effect = lambda x_hist, n_steps: x_hist.iloc[-n_steps:]

        self._evaluate(mock_bfx)  # default mode = "production"

        self.assertEqual(mock_bfx.call_count, 2)  # one call per origin
        mock_mlflow.log_param.assert_any_call("x_future_mode", "production")

    @patch("src.ml_engineering.ml_5_model_evaluation.build_future_x")
    @patch("src.ml_engineering.ml_5_model_evaluation.mlflow")
    def test_actual_mode_uses_observed_rows(self, mock_mlflow, mock_bfx):
        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

        self._evaluate(mock_bfx, x_future_mode="actual")

        mock_bfx.assert_not_called()
        mock_mlflow.log_param.assert_any_call("x_future_mode", "actual")


# ---------------------------------------------------------------------------
# build_future_x — production-honest future feature construction (Step 5/7 shared)
# ---------------------------------------------------------------------------

def _hist_frame(start: str, periods: int, trend_start: float = 1.0) -> pd.DataFrame:
    """History frame with the full structural column set + one exogenous col."""
    idx = pd.DatetimeIndex(pd.date_range(start, periods=periods, freq="QE"), freq=None)
    post_covid = (idx > pd.Timestamp("2022-12-31")).astype(float)
    trend = np.arange(trend_start, trend_start + periods, dtype=float)
    return pd.DataFrame(
        {
            "year": idx.year.astype(float),
            "quarter": idx.quarter.astype(float),
            "trend_index": trend,
            "covid_period": ((idx >= pd.Timestamp("2020-03-31"))
                             & (idx <= pd.Timestamp("2022-12-31"))).astype(float),
            "post_covid": post_covid,
            "covid_depth": np.clip((idx - pd.Timestamp("2019-12-31")).days / QUARTER_DAYS, 0, 12),
            "recovery_quarters": np.clip((idx - pd.Timestamp("2022-12-31")).days / QUARTER_DAYS, 0, None),
            "trend_x_post_covid": trend * post_covid,
            "quarter_x_post_covid": idx.quarter.astype(float) * post_covid,
            "exog": np.linspace(10.0, 20.0, periods),
        },
        index=idx,
    )


class TestBuildFutureX(unittest.TestCase):
    """Deterministic structural columns are recomputed from the future dates,
    stochastic exogenous columns carried forward, and regime boundaries honoured
    even when the forecast window crosses them."""

    def test_deterministic_extension_and_carry_forward(self):
        """From 2025Q3: dates, year/quarter, trend continue; exog carried forward."""
        hist = _hist_frame("2023-03-31", periods=11)  # 2023Q1 … 2025Q3
        fut = build_future_x(hist, n_steps=4)

        expected_dates = pd.DatetimeIndex(
            ["2025-12-31", "2026-03-31", "2026-06-30", "2026-09-30"]
        )
        self.assertTrue((fut.index == expected_dates).all())
        self.assertEqual(list(fut.columns), list(hist.columns))

        self.assertEqual(list(fut["quarter"]), [4.0, 1.0, 2.0, 3.0])
        self.assertEqual(list(fut["year"]), [2025.0, 2026.0, 2026.0, 2026.0])
        last_trend = hist["trend_index"].iloc[-1]
        self.assertEqual(list(fut["trend_index"]), [last_trend + i for i in (1, 2, 3, 4)])

        # Stochastic exogenous column: strict carry-forward of the last value
        self.assertTrue((fut["exog"] == hist["exog"].iloc[-1]).all())

        # Post-COVID regime: flags constant, ramps continue
        self.assertTrue((fut["covid_period"] == 0.0).all())
        self.assertTrue((fut["post_covid"] == 1.0).all())
        self.assertTrue((fut["covid_depth"] == 12.0).all())
        self.assertTrue(fut["recovery_quarters"].is_monotonic_increasing)

        # Interactions recomputed from the extended parents
        self.assertTrue((fut["trend_x_post_covid"] == fut["trend_index"]).all())
        self.assertTrue((fut["quarter_x_post_covid"] == fut["quarter"]).all())

    def test_regime_boundary_crossing(self):
        """Origin at 2022Q4 (COVID end): the future window must flip the regime
        flags — carry-forward of regime columns would be wrong here."""
        hist = _hist_frame("2020-03-31", periods=12)  # 2020Q1 … 2022Q4 (all COVID)
        self.assertTrue((hist["covid_period"] == 1.0).all())

        fut = build_future_x(hist, n_steps=4)  # 2023Q1 … 2023Q4

        self.assertTrue((fut["covid_period"] == 0.0).all())
        self.assertTrue((fut["post_covid"] == 1.0).all())
        self.assertTrue((fut["covid_depth"] == 12.0).all())
        # recovery ramp starts: 2023-03-31 is 90 days past COVID end
        self.assertAlmostEqual(fut["recovery_quarters"].iloc[0], 90 / QUARTER_DAYS, places=6)
        self.assertTrue(fut["recovery_quarters"].is_monotonic_increasing)
        # Interactions become active even though they were 0 throughout history
        self.assertTrue((fut["trend_x_post_covid"] == fut["trend_index"]).all())

    def test_subset_of_structural_columns(self):
        """Frames missing some structural columns are extended without error;
        unknown columns are carried forward."""
        idx = pd.DatetimeIndex(pd.date_range("2024-03-31", periods=6, freq="QE"), freq=None)
        hist = pd.DataFrame({"quarter": idx.quarter.astype(float), "some_feature": 7.5}, index=idx)

        fut = build_future_x(hist, n_steps=4)

        self.assertEqual(list(fut.columns), ["quarter", "some_feature"])
        self.assertTrue((fut["some_feature"] == 7.5).all())
        self.assertEqual(len(fut), 4)

    def test_empty_history_raises(self):
        with self.assertRaises(ValueError):
            build_future_x(pd.DataFrame(), n_steps=4)


# ---------------------------------------------------------------------------
# MLflow evaluation table artifacts (mlflow.log_table) — consolidated eval data
# ---------------------------------------------------------------------------

def _pred_records():
    return [
        {"origin_date": pd.Timestamp("2021-09-30"), "target_date": pd.Timestamp("2021-12-31"),
         "horizon": 1, "y_true": 5.0, "y_pred": 4.5, "fold_set": "outer"},
        {"origin_date": pd.Timestamp("2021-09-30"), "target_date": pd.Timestamp("2022-03-31"),
         "horizon": 2, "y_true": 5.0, "y_pred": 6.0, "fold_set": "outer"},
        {"origin_date": pd.Timestamp("2020-09-30"), "target_date": pd.Timestamp("2020-12-31"),
         "horizon": 1, "y_true": 4.0, "y_pred": 4.0, "fold_set": "inner"},
    ]


class TestBuildEvalTables(unittest.TestCase):

    def test_predictions_table_has_errors_and_serializable_dates(self):
        # args: (records, mase, mape, r2, mae, rmse, mae_inner)
        preds, _ = _build_eval_tables(_pred_records(), 0.9, 0.1, 0.5, 0.4, 0.5, 0.6)
        self.assertEqual(len(preds), 3)
        self.assertIn("abs_error", preds.columns)
        self.assertIn("abs_pct_error", preds.columns)
        # dates coerced to str so the JSON table artifact serializes cleanly
        self.assertTrue(all(isinstance(d, str) for d in preds["origin_date"]))
        self.assertEqual(sorted(preds["abs_error"].round(3).tolist()), [0.0, 0.5, 1.0])

    def test_metrics_table_headline_inner_and_per_horizon(self):
        # args: (records, mase, mape, r2, mae, rmse, mae_inner)
        _, metrics = _build_eval_tables(_pred_records(), 0.9, 0.1, 0.5, 0.4, 0.5, 0.6)
        m = {(r.scope, r.metric): r.value for r in metrics.itertuples()}
        self.assertEqual(m[("outer", "MASE")], 0.9)  # THE metric leads the table
        self.assertEqual(m[("outer", "MAPE")], 0.1)
        self.assertEqual(m[("outer", "R2")], 0.5)
        self.assertEqual(m[("inner", "MAE")], 0.6)
        # per-horizon outer MAPE: h1 = |5-4.5|/5 = 0.10 ; h2 = |5-6|/5 = 0.20
        self.assertAlmostEqual(m[("outer_h1", "MAPE")], 0.10)
        self.assertAlmostEqual(m[("outer_h2", "MAPE")], 0.20)

    def test_empty_records_still_returns_headline_metrics(self):
        nan = float("nan")
        preds, metrics = _build_eval_tables([], nan, nan, nan, nan, nan, nan)
        self.assertTrue(preds.empty)
        self.assertEqual(len(metrics), 6)  # 5 outer headline (incl. MASE) + 1 inner


class TestEvalTablesLogged(unittest.TestCase):

    @patch("src.ml_engineering.ml_5_model_evaluation.mlflow")
    def test_evaluate_logs_eval_table_artifacts(self, mock_mlflow):
        """evaluate() must log the predictions + metrics tables via mlflow.log_table
        so the run's Evaluation tab is populated."""
        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
        x_train, y_train = _make_series(32)
        x_test, y_test = _make_series(8, start="2023-01-01")
        baseline = SectorQuarterRollingMean()
        baseline.fit(x_train, y_train)

        ModelEvaluator(session=MagicMock()).evaluate(
            run_id="run_x", fitted_model=baseline,
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            model_name="test_baseline", n_test_points=8,
        )

        paths = [
            (c.kwargs.get("artifact_file") or (c.args[1] if len(c.args) > 1 else None))
            for c in mock_mlflow.log_table.call_args_list
        ]
        self.assertIn("eval/walk_forward_predictions.json", paths)
        self.assertIn("eval/metrics_summary.json", paths)


if __name__ == "__main__":
    unittest.main()
