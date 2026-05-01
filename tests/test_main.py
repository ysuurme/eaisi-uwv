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
            feature_groups=None,
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
            feature_groups=None,
        )

    @patch("main.run_pipeline")
    @patch("main.ensure_mlflow_ui")
    def test_main_feature_groups_parsed(self, mock_ui, mock_run):
        """Verify CLI feature group names are split and passed correctly."""
        import sys
        with patch.object(sys, "argv", ["main.py", "my_table", "linear", "temporal,labor_market"]):
            from main import main
            main()

        mock_run.assert_called_once_with(
            experiment_key="linear",
            gold_table="my_table",
            feature_groups=["temporal", "labor_market"],
        )


if __name__ == "__main__":
    unittest.main()
