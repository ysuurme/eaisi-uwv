import os
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from main import run_ml_pipeline

class TestMainE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up a temporary environment for E2E testing."""
        cls.test_dir = Path("tests/tmp_e2e")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        cls.gold_db = cls.test_dir / "gold_test.db"
        cls.eval_db = cls.test_dir / "eval_test.db"
        
        # Create a mock gold table
        conn = sqlite3.connect(cls.gold_db)
        df = pd.DataFrame({
            "Perioden_dt": pd.date_range("2020-01-01", periods=20, freq="ME"),
            "feature1": range(20),
            "Ziekteverzuimpercentage_1": [0.1 * i for i in range(20)],
            "silver_id": range(20)
        })
        df.to_sql("80072ned_gold", conn, index=False)
        conn.close()

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test files."""
        import shutil
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    @patch("src.ml_engineering.model_training.DIR_DB_GOLD")
    @patch("src.ml_engineering.model_training.DIR_DB_EVAL")
    @patch("src.ml_engineering.model_evaluation.DIR_DB_EVAL")
    @patch("src.ml_engineering.model_orchestrator.DIR_DB_GOLD")
    @patch("main.ModelOrchestrator")
    @patch("main.ensure_mlflow_ui")
    def test_run_ml_pipeline_e2e(self, mock_ui, mock_orch, mock_gold_orch, mock_eval_eval, mock_eval_train, mock_gold_train):
        """
        Verify that main.py can trigger the orchestrator correctly.
        (Note: We mock the orchestrator itself here to avoid complex MLflow/SQLite locking 
        issues in a single-threaded unit test, focusing on the main.py logic.)
        """
        # Configure mocks
        mock_gold_train.return_value = self.gold_db
        mock_eval_train.return_value = self.eval_db
        
        # Trigger the pipeline
        run_ml_pipeline(gold_table="80072ned_gold", model_key="baseline")
        
        # Verify Orchestrator was called with correct derived names
        mock_orch.assert_called_once()
        args, kwargs = mock_orch.call_args
        self.assertEqual(kwargs["experiment_name"], "80072ned_SickLeave")
        self.assertEqual(kwargs["model_name"], "Baseline_Mean_80072ned")
        
        # Verify run_experiment was called
        mock_orch.return_value.run_experiment.assert_called_once()

if __name__ == "__main__":
    unittest.main()
