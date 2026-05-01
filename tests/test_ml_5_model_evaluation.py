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
from src.ml_engineering.ml_5_model_evaluation import ModelEvaluator


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
    def test_n_eval_points_logged(self, mock_mlflow):
        """n_eval_points and n_eval_origins must be logged to MLflow."""
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
        self.assertIn("n_eval_points", logged)
        self.assertIn("n_eval_origins", logged)
        self.assertEqual(logged["n_eval_points"], 8)
        self.assertEqual(logged["n_eval_origins"], 2)


if __name__ == "__main__":
    unittest.main()
