import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from pathlib import Path
from src.ml_engineering.model_training import DatasetLoader, ModelTrainer
from src.ml_engineering.model_configs import ModelExperiment

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.db_path = Path("fake_db.db")
        self.table_name = "fake_table"
        self.target = "target_col"
        self.mock_df = pd.DataFrame({
            "period_enddate": pd.date_range("2020-01-01", periods=10, freq="ME"),
            "feat1": [float(i) for i in range(10)],
            "target_col": [0.1 * i for i in range(10)],
            "silver_id": range(10)
        })

    @patch("src.ml_engineering.model_training.create_engine")
    @patch("src.ml_engineering.model_training.pd.read_sql_table")
    def test_dataset_loader(self, mock_read_sql, mock_engine):
        mock_read_sql.return_value = self.mock_df
        loader = DatasetLoader(self.db_path, self.table_name)
        x_tr, x_te, y_tr, y_te, lineage = loader.load_and_split(self.target, n_splits=2)
        self.assertIn("dataset", lineage)
        self.assertEqual(lineage["target"], self.target)

    @patch("src.ml_engineering.model_training.mlflow")
    def test_model_trainer(self, mock_mlflow):
        mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = "test_run"
        trainer = ModelTrainer(experiment_name="test_exp", engine=MagicMock())
        
        from sklearn.linear_model import LinearRegression
        exp = ModelExperiment(name="test", estimator=LinearRegression(), param_grid={"copy_X": [True]})
        
        mock_session = MagicMock()
        best_model, run_id = trainer.train_experiment(
            session=mock_session,
            experiment=exp,
            x_train=self.mock_df[["feat1"]],
            y_train=self.mock_df["target_col"],
            run_name="test",
            lineage={"dataset": "test", "target": "target_col"}
        )
        self.assertEqual(run_id, "test_run")
        mock_session.merge.assert_called()

if __name__ == "__main__":
    unittest.main()
