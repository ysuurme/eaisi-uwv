"""
Unit tests for Step 4 — ML Model Training.
"""
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.ml_engineering.model_configs import ModelExperiment, SectorQuarterRollingMean


class TestModelTraining(unittest.TestCase):

    def setUp(self):
        """Build minimal 1-row-per-quarter training data (post-Step-1 format)."""
        idx = pd.date_range("2015-01-01", periods=20, freq="QE")
        self.y_train = pd.Series(
            [4.0 + 0.1 * i for i in range(20)],
            index=idx,
            name="target",
        )
        self.x_train = pd.DataFrame(
            {
                "quarter": [d.quarter for d in idx],
                "year": [d.year for d in idx],
                "feat1": [float(i) for i in range(20)],
            },
            index=idx,
        )

    @patch("src.ml_engineering.ml_4_model_training.mlflow")
    def test_model_trainer_baseline(self, mock_mlflow):
        """SectorQuarterRollingMean (sklearn path) should fit and return a valid model."""
        from src.ml_engineering.ml_4_model_training import ModelTrainer

        # Wire up mock MLflow run context
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run"
        mock_run.info.experiment_id = "exp_1"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.active_run.return_value = mock_run

        exp = ModelExperiment(
            name="test_baseline",
            estimator=SectorQuarterRollingMean(n_years=3),
            param_grid={},
            feature_groups=None,
        )

        mock_session = MagicMock()
        trainer = ModelTrainer(session=mock_session)
        best_model, run_id = trainer.train(
            experiment=exp,
            x_train=self.x_train,
            y_train=self.y_train,
            run_name="test",
            lineage={"dataset": "test", "target": "target"},
        )

        self.assertEqual(run_id, "test_run")
        self.assertIsInstance(best_model, SectorQuarterRollingMean)
        # Baseline builds a non-empty lookup after fitting
        self.assertGreater(len(best_model._lookup), 0)

    @patch("src.ml_engineering.ml_4_model_training.mlflow")
    def test_model_trainer_logs_params(self, mock_mlflow):
        """Trainer should call mlflow.log_param for model_class and train_rows."""
        from src.ml_engineering.ml_4_model_training import ModelTrainer

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_2"
        mock_run.info.experiment_id = "exp_2"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.active_run.return_value = mock_run

        exp = ModelExperiment(
            name="test_baseline",
            estimator=SectorQuarterRollingMean(n_years=3),
            param_grid={},
            feature_groups=None,
        )

        trainer = ModelTrainer(session=MagicMock())
        trainer.train(
            experiment=exp,
            x_train=self.x_train,
            y_train=self.y_train,
            run_name="test",
            lineage={"dataset": "test", "target": "target"},
        )

        logged_keys = [call.args[0] for call in mock_mlflow.log_param.call_args_list]
        self.assertIn("model_class", logged_keys)
        self.assertIn("train_rows", logged_keys)

    @patch("src.ml_engineering.ml_4_model_training.mlflow")
    def test_trainer_logs_reproducibility_lineage(self, mock_mlflow):
        """A run must expose feature_groups, active preset, experiment_key, and a
        feature_set_hash so two runs of the same estimator are distinguishable
        and reproducible (TODO: reproducibility/lineage)."""
        from src.ml_engineering.ml_4_model_training import ModelTrainer

        mock_run = MagicMock()
        mock_run.info.run_id = "run_lineage"
        mock_run.info.experiment_id = "exp_3"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.active_run.return_value = mock_run

        exp = ModelExperiment(
            name="RandomForest_Reduced",
            estimator=SectorQuarterRollingMean(n_years=3),
            param_grid={},
            feature_groups=["labor_volume", "workforce"],
        )

        trainer = ModelTrainer(session=MagicMock())
        trainer.train(
            experiment=exp,
            x_train=self.x_train,
            y_train=self.y_train,
            run_name="t",
            lineage={"dataset": "d", "target": "target", "experiment_key": "random_forest"},
        )

        # Collect params from both log_param (singular) and log_params (plural).
        logged = {}
        for c in mock_mlflow.log_param.call_args_list:
            logged[c.args[0]] = c.args[1] if len(c.args) > 1 else c.kwargs.get("value")
        for c in mock_mlflow.log_params.call_args_list:
            logged.update(c.args[0] if c.args else c.kwargs.get("params", {}))

        self.assertIn("feature_groups", logged)
        self.assertIn("labor_volume", logged["feature_groups"])
        self.assertIn("active_preset", logged)
        self.assertEqual(logged["experiment_key"], "random_forest")
        # Clear, human-readable model identity (not the opaque sktime wrapper).
        self.assertEqual(logged["model_name"], "RandomForest_Reduced")
        self.assertIn("base_estimator_class", logged)

        # feature_set_hash + model identity exposed as filterable run tags.
        tagged = {}
        for c in mock_mlflow.set_tags.call_args_list:
            tagged.update(c.args[0] if c.args else c.kwargs.get("tags", {}))
        for c in mock_mlflow.set_tag.call_args_list:
            tagged[c.args[0]] = c.args[1] if len(c.args) > 1 else c.kwargs.get("value")
        self.assertIn("feature_set_hash", tagged)
        self.assertEqual(tagged.get("model_family"), "RandomForest_Reduced")
        self.assertIn("model_type", tagged)


    def test_base_estimator_name_unwraps_reduction_pipeline(self):
        """The clear model type must unwrap the sktime reducer + sklearn Pipeline
        down to the actual algorithm (e.g. LinearRegression), not the wrapper."""
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sktime.forecasting.compose import make_reduction
        from src.ml_engineering.ml_4_model_training import _base_estimator_name

        reduced = make_reduction(
            Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())]),
            window_length=4, strategy="recursive",
        )
        self.assertEqual(_base_estimator_name(reduced), "LinearRegression")
        self.assertEqual(
            _base_estimator_name(SectorQuarterRollingMean()), "SectorQuarterRollingMean"
        )


if __name__ == "__main__":
    unittest.main()
