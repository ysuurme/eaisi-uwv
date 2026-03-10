import unittest
from unittest.mock import MagicMock, patch
from src.ml_engineering.model_orchestrator import ModelOrchestrator

class TestModelOrchestrator(unittest.TestCase):
    @patch("src.ml_engineering.model_orchestrator.ModelTrainer")
    @patch("src.ml_engineering.model_orchestrator.ModelEvaluator")
    @patch("src.ml_engineering.model_orchestrator.ModelRegistry")
    @patch("src.ml_engineering.model_orchestrator.DatasetLoader")
    def test_run_experiment_success(self, mock_loader, mock_registry, mock_evaluator, mock_trainer):
        mock_loader.return_value.load_and_split.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), {"dataset": "test"}
        )
        mock_trainer.return_value.train_experiment.return_value = (MagicMock(), "run_123")
        mock_evaluator.return_value.evaluate_candidate.return_value = True
        mock_registry.return_value.register_candidate.return_value = "1"
        
        orchestrator = ModelOrchestrator("exp", "model")
        exp_config = MagicMock()
        exp_config.name = "test_model"
        orchestrator.run_experiment("table", exp_config)
        
        mock_evaluator.return_value.evaluate_candidate.assert_called()
        mock_registry.return_value.register_candidate.assert_called()

if __name__ == "__main__":
    unittest.main()
