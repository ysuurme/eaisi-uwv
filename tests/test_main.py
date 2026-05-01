import unittest
from unittest.mock import patch, MagicMock


class TestMain(unittest.TestCase):
    @patch("main.run_pipeline")
    @patch("main.ensure_mlflow_ui")
    def test_main_calls_pipeline(self, mock_ui, mock_run):
        """Verify main.py wires CLI args into run_pipeline correctly."""
        import sys
        with patch.object(sys, "argv", ["main.py", "test_gold", "linear"]):
            from main import main
            main()

        mock_run.assert_called_once_with(
            experiment_key="linear",
            gold_table="test_gold",
            features=None,
        )

    @patch("main.run_pipeline")
    @patch("main.ensure_mlflow_ui")
    def test_main_defaults(self, mock_ui, mock_run):
        """Verify main.py uses sensible defaults when no CLI args provided."""
        import sys
        with patch.object(sys, "argv", ["main.py"]):
            from main import main
            main()

        mock_run.assert_called_once_with(
            experiment_key="random_forest",
            gold_table="master_data_ml_preprocessed",
            features=None,
        )


if __name__ == "__main__":
    unittest.main()
