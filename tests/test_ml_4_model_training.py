import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.ml_engineering.model_configs import ModelExperiment


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.mock_df = pd.DataFrame({
            "period_enddate": pd.date_range("2020-01-01", periods=10, freq="ME"),
            "feat1": [float(i) for i in range(10)],
            "target_col": [0.1 * i for i in range(10)],
        })

    @patch("src.ml_engineering.ml_4_model_training.mlflow")
    def test_model_trainer(self, mock_mlflow):
        from src.ml_engineering.ml_4_model_training import ModelTrainer

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run"
        mock_run.info.experiment_id = "exp_1"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.active_run.return_value = mock_run

        from sklearn.linear_model import LinearRegression
        exp = ModelExperiment(name="test", estimator=LinearRegression(), param_grid={"copy_X": [True]})

        mock_session = MagicMock()
        trainer = ModelTrainer(session=mock_session)
        best_model, run_id = trainer.train(
            experiment=exp, x_train=self.mock_df[["feat1"]],
            y_train=self.mock_df["target_col"], run_name="test",
            lineage={"dataset": "test", "target": "target_col"},
        )
        self.assertEqual(run_id, "test_run")
        mock_session.merge.assert_called()


if __name__ == "__main__":
    unittest.main()
