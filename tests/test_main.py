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
            sbi_filter_col=None,
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
            experiment_key="linear",
            gold_table="master_data_ml_preprocessed",
            sbi_filter_col=None,
            feature_groups=None,
        )

    @patch("main.run_pipeline")
    @patch("main.ensure_mlflow_ui")
    def test_main_sbi_filter_col_parsed(self, mock_ui, mock_run):
        """Verify a sector-specific OHE column is passed through correctly."""
        import sys
        with patch.object(
            sys, "argv",
            ["main.py", "my_table", "baseline", "BedrijfskenmerkenSBI2008_301000"],
        ):
            from main import main
            main()

        mock_run.assert_called_once_with(
            experiment_key="baseline",
            gold_table="my_table",
            sbi_filter_col="BedrijfskenmerkenSBI2008_301000",
            feature_groups=None,
        )

    @patch("main.run_pipeline")
    @patch("main.ensure_mlflow_ui")
    def test_main_sbi_placeholder_skipped(self, mock_ui, mock_run):
        """A '-' placeholder for sbi_filter_col should resolve to None."""
        import sys
        with patch.object(
            sys, "argv",
            ["main.py", "my_table", "linear", "-", "temporal,labor_market"],
        ):
            from main import main
            main()

        mock_run.assert_called_once_with(
            experiment_key="linear",
            gold_table="my_table",
            sbi_filter_col=None,
            feature_groups=["temporal", "labor_market"],
        )

    @patch("main.run_pipeline")
    @patch("main.run_feature_selection")
    @patch("main.ensure_mlflow_ui")
    def test_main_select_features_flag(self, mock_ui, mock_select, mock_run):
        """--select-features runs the selection flow and exits (no training)."""
        import sys
        with patch.object(sys, "argv", ["main.py", "--select-features"]):
            from main import main
            main()

        mock_select.assert_called_once_with(gold_table="master_data_ml_preprocessed")
        mock_run.assert_not_called()

    @patch("main.run_pipeline")
    @patch("main.run_forecast")
    @patch("main.ensure_mlflow_ui")
    def test_main_forecast_flag(self, mock_ui, mock_forecast, mock_run):
        """--forecast runs the forward-forecast flow and exits (no training)."""
        import sys
        with patch.object(sys, "argv", ["main.py", "--forecast"]):
            from main import main
            main()

        mock_forecast.assert_called_once_with(gold_table="master_data_ml_preprocessed")
        mock_run.assert_not_called()

    @patch("main.run_pipeline")
    @patch("main.run_forecast")
    @patch("main.ensure_mlflow_ui")
    def test_main_forecast_flag_with_gold_table(self, mock_ui, mock_forecast, mock_run):
        """--forecast honours an explicit gold-table positional argument."""
        import sys
        with patch.object(sys, "argv", ["main.py", "my_gold", "--forecast"]):
            from main import main
            main()

        mock_forecast.assert_called_once_with(gold_table="my_gold")
        mock_run.assert_not_called()

    @patch("main.run_pipeline")
    @patch("main.run_report")
    @patch("main.ensure_mlflow_ui")
    def test_main_report_flag(self, mock_ui, mock_report, mock_run):
        """--report runs the reporting flow and exits (no training)."""
        import sys
        with patch.object(sys, "argv", ["main.py", "--report"]):
            from main import main
            main()

        mock_report.assert_called_once_with(gold_table="master_data_ml_preprocessed")
        mock_run.assert_not_called()

    @patch("main.run_pipeline")
    @patch("main.ensure_mlflow_ui")
    def test_main_feature_groups_parsed(self, mock_ui, mock_run):
        """Verify CLI feature group names are split and passed correctly (with sbi col)."""
        import sys
        with patch.object(
            sys, "argv",
            ["main.py", "my_table", "linear", "BedrijfskenmerkenSBI2008_301000", "temporal,labor_market"],
        ):
            from main import main
            main()

        mock_run.assert_called_once_with(
            experiment_key="linear",
            gold_table="my_table",
            sbi_filter_col="BedrijfskenmerkenSBI2008_301000",
            feature_groups=["temporal", "labor_market"],
        )


if __name__ == "__main__":
    unittest.main()
