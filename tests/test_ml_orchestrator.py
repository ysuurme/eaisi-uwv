import unittest
from unittest.mock import MagicMock, patch

from src.ml_engineering.ml_orchestrator import run_pipeline


class TestPipeline(unittest.TestCase):
    @patch("src.ml_engineering.ml_orchestrator.ModelValidator")
    @patch("src.ml_engineering.ml_orchestrator.ModelEvaluator")
    @patch("src.ml_engineering.ml_orchestrator.ModelTrainer")
    @patch("src.ml_engineering.ml_orchestrator.DataPreparator")
    @patch("src.ml_engineering.ml_orchestrator.DataValidator")
    @patch("src.ml_engineering.ml_orchestrator.DataExtractor")
    @patch("src.ml_engineering.ml_orchestrator._ensure_eval_db")
    @patch("src.ml_engineering.ml_orchestrator._configure_mlflow")
    @patch("src.ml_engineering.ml_orchestrator.mlflow")
    @patch("src.ml_engineering.ml_orchestrator.create_engine")
    def test_pipeline_passes_gate(
        self, mock_engine, mock_mlflow, mock_configure, mock_ensure,
        mock_extractor_cls, mock_validator_cls, mock_preparator_cls,
        mock_trainer_cls, mock_evaluator_cls, mock_model_validator_cls,
    ):
        mock_extractor_cls.return_value.extract.return_value = MagicMock()
        mock_preparator_cls.prepare.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock(),
            {"dataset": "test", "target": "t", "feature_count": 1, "train_size": 10, "test_size": 5},
        )
        mock_trainer_cls.return_value.train.return_value = (MagicMock(), "run_123")
        mock_evaluator_cls.return_value.evaluate.return_value = {"r2": 0.9, "mae": 0.1, "rmse": 0.1}
        mock_model_validator_cls.return_value.validate_and_register.return_value = True
        mock_mlflow.get_experiment_by_name.return_value = MagicMock()

        run_pipeline(experiment_key="linear", gold_table="test_table")

        mock_trainer_cls.return_value.train.assert_called_once()
        mock_evaluator_cls.return_value.evaluate.assert_called_once()
        mock_model_validator_cls.return_value.validate_and_register.assert_called_once()

    @patch("src.ml_engineering.ml_orchestrator.ModelValidator")
    @patch("src.ml_engineering.ml_orchestrator.ModelEvaluator")
    @patch("src.ml_engineering.ml_orchestrator.ModelTrainer")
    @patch("src.ml_engineering.ml_orchestrator.DataPreparator")
    @patch("src.ml_engineering.ml_orchestrator.DataValidator")
    @patch("src.ml_engineering.ml_orchestrator.DataExtractor")
    @patch("src.ml_engineering.ml_orchestrator._ensure_eval_db")
    @patch("src.ml_engineering.ml_orchestrator._configure_mlflow")
    @patch("src.ml_engineering.ml_orchestrator.mlflow")
    @patch("src.ml_engineering.ml_orchestrator.create_engine")
    def test_registry_name_is_sector_only_with_family_separate(
        self, mock_engine, mock_mlflow, mock_configure, mock_ensure,
        mock_extractor_cls, mock_validator_cls, mock_preparator_cls,
        mock_trainer_cls, mock_evaluator_cls, mock_model_validator_cls,
    ):
        """ADR-002: validator receives a sector-only registered_model_name
        (no family token) plus the model_family passed separately."""
        mock_extractor_cls.return_value.extract.return_value = MagicMock()
        mock_preparator_cls.prepare.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock(),
            {"dataset": "test", "target": "t", "feature_count": 1, "train_size": 10, "test_size": 5},
        )
        mock_trainer_cls.return_value.train.return_value = (MagicMock(), "run_789")
        mock_evaluator_cls.return_value.evaluate.return_value = {
            "mape": 0.04, "r2": 0.9, "mae": 0.1, "rmse": 0.1,
        }
        mock_model_validator_cls.return_value.validate_and_register.return_value = True
        mock_mlflow.get_experiment_by_name.return_value = MagicMock()

        run_pipeline(
            experiment_key="random_forest",
            gold_table="master_data_ml_preprocessed",
            sbi_filter_col="BedrijfskenmerkenSBI2008_301000",
        )

        _, kwargs = mock_model_validator_cls.return_value.validate_and_register.call_args
        self.assertIn("301000", kwargs["registered_model_name"])
        self.assertNotIn("RandomForest", kwargs["registered_model_name"])
        self.assertEqual(kwargs["model_family"], "RandomForest_Reduced")

    @patch("src.ml_engineering.ml_orchestrator.ModelValidator")
    @patch("src.ml_engineering.ml_orchestrator.ModelEvaluator")
    @patch("src.ml_engineering.ml_orchestrator.ModelTrainer")
    @patch("src.ml_engineering.ml_orchestrator.DataPreparator")
    @patch("src.ml_engineering.ml_orchestrator.DataValidator")
    @patch("src.ml_engineering.ml_orchestrator.DataExtractor")
    @patch("src.ml_engineering.ml_orchestrator._ensure_eval_db")
    @patch("src.ml_engineering.ml_orchestrator._configure_mlflow")
    @patch("src.ml_engineering.ml_orchestrator.mlflow")
    @patch("src.ml_engineering.ml_orchestrator.create_engine")
    def test_pipeline_fails_gate(
        self, mock_engine, mock_mlflow, mock_configure, mock_ensure,
        mock_extractor_cls, mock_validator_cls, mock_preparator_cls,
        mock_trainer_cls, mock_evaluator_cls, mock_model_validator_cls,
    ):
        mock_extractor_cls.return_value.extract.return_value = MagicMock()
        mock_preparator_cls.prepare.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock(),
            {"dataset": "test", "target": "t", "feature_count": 1, "train_size": 10, "test_size": 5},
        )
        mock_trainer_cls.return_value.train.return_value = (MagicMock(), "run_456")
        mock_evaluator_cls.return_value.evaluate.return_value = {"r2": 0.1, "mae": 0.9, "rmse": 1.0}
        mock_model_validator_cls.return_value.validate_and_register.return_value = False
        mock_mlflow.get_experiment_by_name.return_value = MagicMock()

        run_pipeline(experiment_key="linear", gold_table="test_table")

        mock_model_validator_cls.return_value.validate_and_register.assert_called_once()


if __name__ == "__main__":
    unittest.main()
