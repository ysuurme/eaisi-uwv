import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.ml_engineering.ml_5_model_evaluation import ModelEvaluator


class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        self.x_test = pd.DataFrame({"f1": [1.0, 2.0]})
        self.y_test = pd.Series([0.5, 0.6], name="target")

    @patch("src.ml_engineering.ml_5_model_evaluation.mlflow")
    def test_evaluate_returns_metrics(self, mock_mlflow):
        mock_result = MagicMock()
        mock_result.metrics = {"r2_score": 0.8, "mean_absolute_error": 0.1, "root_mean_squared_error": 0.15}
        mock_mlflow.models.evaluate.return_value = mock_result

        model = LinearRegression()
        model.fit(self.x_test, self.y_test)

        mock_session = MagicMock()
        evaluator = ModelEvaluator(session=mock_session)
        metrics = evaluator.evaluate(
            run_id="run_123", fitted_model=model,
            x_test=self.x_test, y_test=self.y_test, model_name="test_model",
        )
        self.assertAlmostEqual(metrics["r2"], 0.8)
        mock_session.merge.assert_called()


if __name__ == "__main__":
    unittest.main()
