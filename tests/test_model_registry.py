import unittest
from unittest.mock import MagicMock, patch
from src.ml_engineering.model_registry import ModelRegistry

class TestModelRegistry(unittest.TestCase):
    @patch("src.ml_engineering.model_registry.MlflowClient")
    def setUp(self, mock_client):
        self.registry = ModelRegistry()
        self.mock_client = self.registry.client

    @patch("src.ml_engineering.model_registry.mlflow.register_model")
    def test_register_candidate(self, mock_register):
        # Mock registration result
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_register.return_value = mock_version
        
        version = self.registry.register_candidate(
            run_id="run_123",
            model_name="test_model",
            tags={"prio": "high"},
            description="desc"
        )
        
        self.assertEqual(version, "1")
        self.mock_client.set_model_version_tag.assert_called_with(
            name="test_model", version="1", key="prio", value="high"
        )
        self.mock_client.update_model_version.assert_called()

    def test_promote_to_alias(self):
        self.registry.promote_to_alias("test_model", "1", "prod")
        self.mock_client.set_registered_model_alias.assert_called_with(
            name="test_model", alias="prod", version="1"
        )

if __name__ == "__main__":
    unittest.main()
