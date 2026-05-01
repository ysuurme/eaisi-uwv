import unittest
from unittest.mock import MagicMock, patch
from src.ml_engineering.ml_6_model_validation import ModelValidator


class TestModelValidation(unittest.TestCase):
    @patch("src.ml_engineering.ml_6_model_validation.MlflowClient")
    def setUp(self, mock_client):
        self.validator = ModelValidator()
        self.mock_client = self.validator.client

    @patch("src.ml_engineering.ml_6_model_validation.mlflow.register_model")
    def test_validate_pass_registers(self, mock_register):
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_register.return_value = mock_version

        passed = self.validator.validate_and_register(
            run_id="run_123", model_name="test_model",
            metrics={"r2": 0.8, "mae": 0.1, "rmse": 0.15},
            threshold_r2=0.5, tags={"prio": "high"}, description="desc",
        )
        self.assertTrue(passed)
        self.mock_client.set_registered_model_alias.assert_called()

    def test_validate_fail_no_register(self):
        passed = self.validator.validate_and_register(
            run_id="run_456", model_name="test_model",
            metrics={"r2": 0.1, "mae": 0.9, "rmse": 1.0},
            threshold_r2=0.5,
        )
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
