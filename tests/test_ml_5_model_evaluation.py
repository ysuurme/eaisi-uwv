"""
Unit tests for Step 5 — ML Model Evaluation.
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.ml_engineering.model_configs import SectorQuarterRollingMean
from src.ml_engineering.ml_5_model_evaluation import ModelEvaluator


class TestModelEvaluation(unittest.TestCase):

    def setUp(self):
        idx = pd.date_range("2022-01-01", periods=4, freq="QE")
        self.y_test = pd.Series([4.0, 4.2, 3.9, 4.1], index=idx, name="target")
        self.x_test = pd.DataFrame(
            {
                "BedrijfstakkenBranchesSBI2008": ["A"] * 4,
                "quarter": [d.quarter for d in idx],
                "year": [d.year for d in idx],
                "feat1": [1.0, 2.0, 3.0, 4.0],
            },
            index=idx,
        )

    @patch("src.ml_engineering.ml_5_model_evaluation.mlflow")
    def test_evaluate_baseline_returns_metrics(self, mock_mlflow):
        """Baseline (sklearn path) should call mlflow.models.evaluate and return metrics dict."""
        mock_result = MagicMock()
        mock_result.metrics = {
            "r2_score": 0.75,
            "mean_absolute_error": 0.1,
            "root_mean_squared_error": 0.15,
        }
        mock_mlflow.models.evaluate.return_value = mock_result
        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

        # Fit baseline on minimal data so predict() works
        idx_train = pd.date_range("2018-01-01", periods=12, freq="QE")
        y_train = pd.Series([4.0] * 12, index=idx_train, name="target")
        x_train = pd.DataFrame(
            {
                "BedrijfstakkenBranchesSBI2008": ["A"] * 12,
                "quarter": [d.quarter for d in idx_train],
            },
            index=idx_train,
        )
        baseline = SectorQuarterRollingMean()
        baseline.fit(x_train, y_train)

        mock_session = MagicMock()
        evaluator = ModelEvaluator(session=mock_session)
        metrics = evaluator.evaluate(
            run_id="run_123",
            fitted_model=baseline,
            x_test=self.x_test,
            y_test=self.y_test,
            model_name="test_model",
        )

        self.assertAlmostEqual(metrics["r2"], 0.75)
        mock_mlflow.models.evaluate.assert_called_once()
        mock_session.merge.assert_called_once()

    @patch("src.ml_engineering.ml_5_model_evaluation.mlflow")
    def test_evaluate_sktime_returns_metrics(self, mock_mlflow):
        """sktime forecaster path should call model.predict(fh, X) and log metrics."""
        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

        mock_model = MagicMock()
        # Return predictions aligned to y_test index
        mock_model.predict.return_value = pd.Series(
            [4.0, 4.1, 4.0, 4.1], index=self.y_test.index
        )
        # Ensure isinstance(mock_model, SectorQuarterRollingMean) returns False
        # (MagicMock is not a SectorQuarterRollingMean instance by default)

        mock_session = MagicMock()
        evaluator = ModelEvaluator(session=mock_session)
        metrics = evaluator.evaluate(
            run_id="run_456",
            fitted_model=mock_model,
            x_test=self.x_test,
            y_test=self.y_test,
            model_name="sktime_model",
        )

        self.assertIn("r2", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        mock_model.predict.assert_called_once()
        mock_mlflow.log_metrics.assert_called_once()
        mock_session.merge.assert_called_once()


if __name__ == "__main__":
    unittest.main()
