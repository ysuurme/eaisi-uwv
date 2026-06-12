import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.config import ML_TARGET_COLUMN
from src.ml_engineering.ml_orchestrator import (
    _group_by_registry_category,
    _partition_candidates,
    run_feature_selection,
    run_pipeline,
)


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


# ---------------------------------------------------------------------------
# Feature selection flow (run_feature_selection)
# ---------------------------------------------------------------------------

#: Permissive funnel — every filter passes everything through, so the e2e test
#: exercises plumbing (registry filter, chain, grouping, catalog write), not
#: statistical thresholds.
_PERMISSIVE_FUNNEL = {
    "near_constant":      {"max_fraction": 1.0},
    "correlation":        {"threshold": 0.0},
    "lagged_correlation": {"threshold": 0.0, "horizons": [1]},
    "granger":            {"max_lag": 1, "p_threshold": 1.0, "min_sector_fraction": 0.0},
    "lasso_stability":    {"n_bootstraps": 4, "threshold": 0.0, "horizons": [1]},
    "redundancy":         {"threshold": 0.999},
}

_ORIGIN = {
    "feat_q1": "85920NED",            # quarterly (labor_volume)
    "feat_q2": "85920NED",            # quarterly (labor_volume)
    "feat_wage": "85917NED",          # quarterly (wages)
    "feat_yearly_origin": "81628NED", # yearly (health) → excluded by frequency
    "y_already": "85542NED",          # yearly prefix → excluded by prefix
}


def _synthetic_panel(n_quarters: int = 28, sectors=("S1", "S2", "S3")) -> pd.DataFrame:
    """Full-panel frame as load_full_panel would return it (incl. sector col)."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2016-03-31", periods=n_quarters, freq="QE")
    frames = []
    for s in sectors:
        x1 = np.cumsum(rng.normal(size=n_quarters))
        x2 = np.cumsum(rng.normal(size=n_quarters))
        xw = np.cumsum(rng.normal(size=n_quarters))
        target = (
            0.6 * np.roll(x1, 1) + 0.4 * np.roll(xw, 1)
            + rng.normal(scale=0.1, size=n_quarters)
        )
        frames.append(pd.DataFrame({
            "period_enddate": dates,
            "year": dates.year,
            "quarter": dates.quarter,
            "sector": s,
            ML_TARGET_COLUMN: target,
            "feat_q1": x1,
            "feat_q2": x2,
            "feat_wage": xw,
            "y_already": rng.normal(size=n_quarters),
            "feat_yearly_origin": rng.normal(size=n_quarters),
            "feat_unknown": rng.normal(size=n_quarters),
        }))
    return pd.concat(frames, ignore_index=True)


class TestPartitionCandidates(unittest.TestCase):
    def test_registry_rules(self):
        """y_* prefix, unknown origin, and non-allowed-frequency origins are
        excluded; allowed-table origins become candidates."""
        cols = ["feat_q1", "feat_wage", "feat_yearly_origin", "y_already", "feat_unknown"]
        allowed = {"85920NED", "85917NED"}

        candidates, exclusions = _partition_candidates(cols, _ORIGIN, allowed)

        self.assertEqual(candidates, ["feat_q1", "feat_wage"])
        self.assertEqual(exclusions["yearly_prefix"], 1)
        self.assertEqual(exclusions["frequency"], 1)
        self.assertEqual(exclusions["unknown_origin"], 1)


class TestGroupByRegistryCategory(unittest.TestCase):
    def test_groups_follow_registry_categories(self):
        """Survivors map onto registry categories via column_origin; columns
        with no category stay ungrouped."""
        groups, ungrouped = _group_by_registry_category(
            ["feat_q1", "feat_q2", "feat_wage", "feat_unknown"], _ORIGIN
        )

        self.assertEqual(set(groups), {"labor_volume", "wages"})
        self.assertEqual(groups["labor_volume"]["columns"], ["feat_q1", "feat_q2"])
        self.assertEqual(groups["labor_volume"]["source_table"], "85920NED")
        self.assertEqual(groups["wages"]["columns"], ["feat_wage"])
        self.assertEqual(ungrouped, ["feat_unknown"])


class TestRunFeatureSelection(unittest.TestCase):
    def test_end_to_end_writes_catalog(self):
        """Full flow: panel → registry filter → funnel → grouped catalog JSON."""
        panel = _synthetic_panel()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with open(tmp_path / "column_origin.json", "w") as fh:
                json.dump(_ORIGIN, fh)

            with patch(
                "src.ml_engineering.ml_orchestrator.DIR_FEATURE_SELECTION", tmp_path
            ), patch(
                "src.ml_engineering.ml_orchestrator.DataExtractor.load_full_panel",
                return_value=panel,
            ), patch(
                "src.ml_engineering.ml_orchestrator.reload_feature_catalog"
            ) as mock_reload:
                path = run_feature_selection(
                    gold_table="test_gold", funnel_params=_PERMISSIVE_FUNNEL
                )

            self.assertEqual(path.name, "feature_catalog.json")
            with open(path) as fh:
                artifact = json.load(fh)

            survivors = set(artifact["surviving_features"])
            self.assertTrue(survivors)  # funnel must not eliminate everything
            self.assertTrue(survivors <= {"feat_q1", "feat_q2", "feat_wage"})
            # Excluded at the registry layer — must never reach the catalog
            self.assertNotIn("y_already", survivors)
            self.assertNotIn("feat_yearly_origin", survivors)
            self.assertNotIn("feat_unknown", survivors)

            self.assertTrue(set(artifact["feature_groups"]) <= {"labor_volume", "wages"})
            self.assertEqual(artifact["gold_table"], "test_gold")
            self.assertIn("frequencies_included", artifact)
            pool = artifact["candidate_pool"]
            self.assertEqual(pool["excluded_yearly_prefix"], 1)
            self.assertEqual(pool["excluded_frequency"], 1)
            self.assertEqual(pool["excluded_unknown_origin"], 1)
            # Full filter-chain lineage persisted
            self.assertEqual(
                [entry["filter"] for entry in artifact["filter_chain"]][:2],
                ["near_constant", "correlation"],
            )
            mock_reload.assert_called_once()

    def test_no_candidates_raises(self):
        """If the registry filter excludes everything, fail loud (no empty catalog)."""
        panel = _synthetic_panel()[
            ["period_enddate", "year", "quarter", "sector",
             ML_TARGET_COLUMN, "y_already", "feat_unknown"]
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with open(tmp_path / "column_origin.json", "w") as fh:
                json.dump(_ORIGIN, fh)

            with patch(
                "src.ml_engineering.ml_orchestrator.DIR_FEATURE_SELECTION", tmp_path
            ), patch(
                "src.ml_engineering.ml_orchestrator.DataExtractor.load_full_panel",
                return_value=panel,
            ):
                with self.assertRaises(RuntimeError):
                    run_feature_selection(gold_table="test_gold")


if __name__ == "__main__":
    unittest.main()
