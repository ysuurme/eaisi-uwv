import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from pathlib import Path
from src.ml_engineering.model_evaluation import ModelEvaluator
from sklearn.linear_model import LinearRegression

class TestModelEvaluation(unittest.TestCase):
    @patch("src.ml_engineering.model_evaluation.create_engine")
    def setUp(self, mock_engine):
        self.evaluator = ModelEvaluator(db_eval_path=Path("fake_eval.db"))
        self.x_test = pd.DataFrame({"f1": [1, 2]})
        self.y_test = pd.Series([0.5, 0.6])

    @patch("src.ml_engineering.model_evaluation.mlflow")
    def test_evaluate_candidate_pass(self, mock_mlflow):
        mock_result = MagicMock()
        mock_result.metrics = {"r2_score": 0.8}
        mock_mlflow.log_metrics.return_value = None
        
        # Use real estimator to allow pickling
        model = LinearRegression()
        model.fit(self.x_test, self.y_test)

        mock_session = MagicMock()
        passed = self.evaluator.evaluate_candidate(
            session=mock_session,
            run_id="run_123",
            best_model=model,
            x_test=self.x_test,
            y_test=self.y_test,
            model_name="test_model",
            threshold_r2=0.5
        )
        self.assertTrue(passed)
        mock_session.merge.assert_called()

if __name__ == "__main__":
    unittest.main()
