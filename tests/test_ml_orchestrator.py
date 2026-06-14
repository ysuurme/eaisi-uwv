import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import ML_TARGET_COLUMN
from src.ml_engineering.ml_orchestrator import (
    _apply_feature_selection_holdout,
    _ensure_eval_db,
    _group_by_registry_category,
    _log_forecast_tables,
    _partition_candidates,
    _persist_forecasts,
    run_comparison,
    run_feature_selection,
    run_forecast,
    run_full_sweep,
    run_pipeline,
    run_report,
)
from src.ml_engineering.model_configs import ModelForecastRecord


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

        run_pipeline(experiment_key="ridge", gold_table="test_table")

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

        run_pipeline(experiment_key="ridge", gold_table="test_table")

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
                # holdout=0 here so the funnel plumbing is tested on all 28
                # synthetic quarters (the holdout itself is tested separately).
                path = run_feature_selection(
                    gold_table="test_gold", funnel_params=_PERMISSIVE_FUNNEL,
                    holdout_last_n_quarters=0,
                )

            self.assertEqual(path.name, "feature_catalog.json")
            with open(path) as fh:
                artifact = json.load(fh)

            # Holdout lineage is recorded in the catalog metadata.
            self.assertEqual(artifact["feature_selection_holdout"]["held_out_quarters"], 0)

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


class TestFeatureSelectionHoldout(unittest.TestCase):
    """Leakage guard: the funnel must not see the last N quarters."""

    @staticmethod
    def _panel(n_quarters=28, sectors=("S1", "S2")):
        dates = pd.date_range("2016-03-31", periods=n_quarters, freq="QE")
        return pd.concat(
            [pd.DataFrame({"period_enddate": dates, "sector": s,
                           "v": range(n_quarters)}) for s in sectors],
            ignore_index=True,
        )

    def test_drops_last_n_unique_quarters(self):
        panel = self._panel(n_quarters=28, sectors=("S1", "S2"))
        train, info = _apply_feature_selection_holdout(panel, 20)

        self.assertEqual(info["held_out_quarters"], 20)
        self.assertEqual(train["period_enddate"].nunique(), 8)   # 28 - 20
        self.assertEqual(info["rows_after"], 16)                 # 8 quarters × 2 sectors
        # every surviving row is strictly before the (first held-out) cutoff
        cutoff = pd.Timestamp(info["cutoff_date"])
        self.assertTrue((pd.to_datetime(train["period_enddate"]) < cutoff).all())

    def test_zero_disables_guard(self):
        panel = self._panel(n_quarters=12)
        train, info = _apply_feature_selection_holdout(panel, 0)
        self.assertEqual(info["held_out_quarters"], 0)
        self.assertEqual(len(train), len(panel))

    def test_raises_when_holdout_not_smaller_than_history(self):
        panel = self._panel(n_quarters=10)
        with self.assertRaises(RuntimeError):
            _apply_feature_selection_holdout(panel, 10)


# ---------------------------------------------------------------------------
# Forecast production (run_forecast + _persist_forecasts) — Step 7 wiring
# ---------------------------------------------------------------------------

def _forecast_frame(sector: str, y_preds, hash_: str = "abc12345") -> pd.DataFrame:
    """A tidy ml_7-style forecast frame (origin 2025Q3 → future 2025Q4…)."""
    n = len(y_preds)
    origin = pd.Timestamp("2025-09-30")
    dates = pd.date_range("2025-12-31", periods=n, freq="QE")
    return pd.DataFrame({
        "sector_code": [sector] * n,
        "model_family": ["Ridge_Reduced"] * n,
        "model_type": ["Ridge"] * n,
        "experiment_key": ["ridge"] * n,
        "champion_version": ["3"] * n,
        "origin_date": [origin] * n,
        "target_date": dates,
        "horizon": list(range(1, n + 1)),
        "y_pred": list(y_preds),
        "feature_catalog_hash": [hash_] * n,
        "champion_run_id": [f"run_{sector}"] * n,
    })


class TestPersistForecasts(unittest.TestCase):
    """Delete-then-insert into model_forecasts against a real temp SQLite DB."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        db = Path(self._tmp.name) / "eval.db"
        self.engine = create_engine(f"sqlite:///{db.as_posix()}")
        _ensure_eval_db(self.engine)  # creates model_forecasts (additive table)

    def tearDown(self):
        self.engine.dispose()
        self._tmp.cleanup()

    def _rows(self):
        with sessionmaker(bind=self.engine)() as session:
            return session.query(ModelForecastRecord).all()

    def test_write_maps_origin_to_forecast_made_on_and_hash(self):
        n = _persist_forecasts(self.engine, _forecast_frame("T001081", [4.1, 3.7, 3.5, 3.9]))
        self.assertEqual(n, 4)
        rows = sorted(self._rows(), key=lambda r: r.horizon)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0].sector_code, "T001081")
        # the frame's origin_date maps onto the table's forecast_made_on column
        self.assertEqual(pd.Timestamp(rows[0].forecast_made_on), pd.Timestamp("2025-09-30"))
        self.assertEqual(pd.Timestamp(rows[0].target_date), pd.Timestamp("2025-12-31"))
        self.assertEqual(rows[0].feature_catalog_hash, "abc12345")
        self.assertEqual([r.horizon for r in rows], [1, 2, 3, 4])

    def test_delete_then_insert_replaces_only_same_sector(self):
        _persist_forecasts(self.engine, _forecast_frame("T001081", [4.1, 3.7, 3.5, 3.9]))
        _persist_forecasts(self.engine, _forecast_frame("301000", [5.1, 4.7, 4.5, 4.9]))
        # Re-run T001081 with new values — must REPLACE its rows, not duplicate.
        _persist_forecasts(
            self.engine, _forecast_frame("T001081", [9.0, 9.0, 9.0, 9.0], hash_="zzz")
        )
        rows = self._rows()
        self.assertEqual(len(rows), 8)  # 4 (301000) + 4 (fresh T001081), no dupes
        t = [r for r in rows if r.sector_code == "T001081"]
        self.assertEqual(len(t), 4)
        self.assertTrue(all(r.y_pred == 9.0 for r in t))
        self.assertTrue(all(r.feature_catalog_hash == "zzz" for r in t))
        # the other sector is left untouched by a single-sector re-run
        self.assertEqual(len([r for r in rows if r.sector_code == "301000"]), 4)


class TestRunForecastWiring(unittest.TestCase):
    """Orchestrator wiring with a mocked registry/eval-DB boundary."""

    @patch("src.ml_engineering.ml_orchestrator._log_forecast_tables", return_value=2)
    @patch("src.ml_engineering.ml_orchestrator._render_forecast_figures")
    @patch("src.ml_engineering.ml_orchestrator._persist_forecasts", return_value=8)
    @patch("src.ml_engineering.ml_7_model_inference.forecast_all_champions")
    @patch("mlflow.tracking.MlflowClient")
    @patch("src.ml_engineering.ml_orchestrator._ensure_eval_db")
    @patch("src.ml_engineering.ml_orchestrator.create_engine")
    @patch("src.ml_engineering.ml_orchestrator._configure_mlflow")
    def test_persists_logs_tables_and_renders_when_champions_exist(
        self, mock_cfg, mock_engine, mock_ensure, mock_client_cls,
        mock_forecast_all, mock_persist, mock_render, mock_log_tables,
    ):
        frame = pd.concat(
            [_forecast_frame("T001081", [1, 2, 3, 4]),
             _forecast_frame("301000", [1, 2, 3, 4])],
            ignore_index=True,
        )
        mock_forecast_all.return_value = frame

        n = run_forecast(gold_table="master_data_ml_preprocessed")

        self.assertEqual(n, 8)
        mock_forecast_all.assert_called_once()
        mock_persist.assert_called_once()
        mock_log_tables.assert_called_once()   # forecast tables logged to MLflow
        mock_render.assert_called_once()

    @patch("src.ml_engineering.ml_orchestrator._log_forecast_tables")
    @patch("src.ml_engineering.ml_orchestrator._render_forecast_figures")
    @patch("src.ml_engineering.ml_orchestrator._persist_forecasts")
    @patch("src.ml_engineering.ml_7_model_inference.forecast_all_champions")
    @patch("mlflow.tracking.MlflowClient")
    @patch("src.ml_engineering.ml_orchestrator._ensure_eval_db")
    @patch("src.ml_engineering.ml_orchestrator.create_engine")
    @patch("src.ml_engineering.ml_orchestrator._configure_mlflow")
    def test_empty_registry_writes_nothing(
        self, mock_cfg, mock_engine, mock_ensure, mock_client_cls,
        mock_forecast_all, mock_persist, mock_render, mock_log_tables,
    ):
        mock_forecast_all.return_value = pd.DataFrame()

        n = run_forecast()

        self.assertEqual(n, 0)
        mock_persist.assert_not_called()
        mock_log_tables.assert_not_called()
        mock_render.assert_not_called()


class TestLogForecastTables(unittest.TestCase):
    """Per-champion-run forecast table logging (mlflow.log_table) + idempotency."""

    @staticmethod
    def _client(existing_paths):
        client = MagicMock()
        client.list_artifacts.return_value = [MagicMock(path=p) for p in existing_paths]
        return client

    @patch("src.ml_engineering.ml_orchestrator.mlflow")
    def test_logs_table_per_champion_run_when_absent(self, mock_mlflow):
        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
        client = self._client([])  # no prior forecast artifact
        frame = pd.concat(
            [_forecast_frame("T001081", [1, 2, 3, 4]),
             _forecast_frame("301000", [1, 2, 3, 4])],
            ignore_index=True,
        )
        n = _log_forecast_tables(client, frame)

        self.assertEqual(n, 2)  # two distinct champion runs
        self.assertEqual(mock_mlflow.log_table.call_count, 2)
        for call in mock_mlflow.log_table.call_args_list:
            self.assertEqual(call.kwargs.get("artifact_file"), "eval/forward_forecast.json")
            # the run-id helper column is stripped from the logged table
            self.assertNotIn("champion_run_id", call.args[0].columns)

    @patch("src.ml_engineering.ml_orchestrator.mlflow")
    def test_skips_runs_that_already_have_the_table(self, mock_mlflow):
        client = self._client(["eval/forward_forecast.json"])  # already present
        n = _log_forecast_tables(client, _forecast_frame("T001081", [1, 2, 3, 4]))
        self.assertEqual(n, 0)
        mock_mlflow.log_table.assert_not_called()

    @patch("src.ml_engineering.ml_orchestrator.mlflow")
    def test_no_champion_run_id_column_is_noop(self, mock_mlflow):
        frame = _forecast_frame("T001081", [1, 2, 3, 4]).drop(columns=["champion_run_id"])
        n = _log_forecast_tables(MagicMock(), frame)
        self.assertEqual(n, 0)
        mock_mlflow.log_table.assert_not_called()


class TestRunReportWiring(unittest.TestCase):
    """run_report orchestrates the read-model refresh + figure bundle + summary.

    After consolidation, the full figure bundle (standard + experiment matrix)
    comes from m_model_viz.generate_all, and the narrative from
    m_sector_quality.build_narrative_markdown — m_experiment_matrix is gone.
    """

    @patch("src.ml_engineering.ml_orchestrator._configure_mlflow")
    @patch("src.utils.m_sector_quality.build_narrative_markdown", return_value="# report\n")
    @patch("src.utils.m_model_viz.generate_all", return_value=["f1.png", "f2.png", "f3.png"])
    @patch("src.utils.m_sector_quality.write_report")
    @patch("src.utils.m_sector_quality.load_sector_performance")
    @patch("src.utils.m_sector_quality.refresh_sector_performance", return_value=3)
    def test_report_runs_all_stages_and_writes_summary(
        self, mock_refresh, mock_load, mock_write, mock_viz_all, mock_narrative, mock_cfg,
    ):
        mock_load.return_value = pd.DataFrame()
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / ".claude" / "week_2026-06-19_model_report.md"
            with patch("src.ml_engineering.ml_orchestrator.PROJECT_ROOT", Path(tmp)):
                result = run_report(gold_table="master_data_ml_preprocessed")

            self.assertTrue(summary.exists())
            self.assertEqual(summary.read_text(encoding="utf-8"), "# report\n")

        mock_refresh.assert_called_once()
        mock_viz_all.assert_called_once()
        mock_narrative.assert_called_once()
        self.assertEqual(result["sectors"], 3)
        self.assertEqual(result["figures"], 3)

    @patch("src.ml_engineering.ml_orchestrator._configure_mlflow")
    @patch("src.utils.m_sector_quality.build_narrative_markdown", return_value="# r\n")
    @patch("src.utils.m_model_viz.generate_all", return_value=[])
    @patch("src.utils.m_sector_quality.write_report")
    @patch("src.utils.m_sector_quality.load_sector_performance")
    @patch("src.utils.m_sector_quality.refresh_sector_performance")
    def test_report_survives_refresh_failure(
        self, mock_refresh, mock_load, mock_write, mock_viz_all, mock_narrative, mock_cfg,
    ):
        """A failing hierarchy refresh (e.g. missing dimension file) must not
        abort the report — the rest of the stages still run."""
        mock_refresh.side_effect = FileNotFoundError("dimension json missing")
        mock_load.return_value = pd.DataFrame()
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.ml_engineering.ml_orchestrator.PROJECT_ROOT", Path(tmp)):
                result = run_report()

        mock_viz_all.assert_called_once()
        self.assertEqual(result["sectors"], 0)


class TestRunComparisonWiring(unittest.TestCase):
    """run_comparison loads families from the eval DB and feeds compare_all_models."""

    @patch("src.utils.m_evaluation.compare_all_models")
    @patch("src.utils.m_pipeline_loader.load_families_from_eval_db")
    def test_loads_families_and_compares(self, mock_load, mock_compare):
        df_a = pd.DataFrame({"model_name": ["AutoETS_Stat"]})
        df_b = pd.DataFrame({"model_name": ["Pipeline"]})
        baseline = pd.DataFrame({"model_name": ["baseline"]})
        mock_load.return_value = ([df_a, df_b], baseline, {})

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.ml_engineering.ml_orchestrator.PROJECT_ROOT", Path(tmp)):
                result = run_comparison(gold_table="master_data_ml_preprocessed")

        mock_load.assert_called_once()
        mock_compare.assert_called_once()
        # the scorecard is written under reports/comparison and baseline is passed
        _, kwargs = mock_compare.call_args
        self.assertIn("comparison", str(kwargs["output_dir"]))
        self.assertIsNotNone(kwargs["baseline_df"])
        self.assertEqual(result["families"], 2)
        self.assertEqual(result["models"], ["AutoETS_Stat", "Pipeline"])

    @patch("src.utils.m_evaluation.compare_all_models")
    @patch("src.utils.m_pipeline_loader.load_families_from_eval_db")
    def test_skips_when_fewer_than_two_families(self, mock_load, mock_compare):
        mock_load.return_value = ([pd.DataFrame({"model_name": ["AutoETS_Stat"]})],
                                  pd.DataFrame(), {})
        result = run_comparison()
        mock_compare.assert_not_called()
        self.assertEqual(result["families"], 1)


class TestRunFullSweep(unittest.TestCase):
    """run_full_sweep loops run_sector_sweep over every model family."""

    @patch("src.ml_engineering.ml_orchestrator.run_sector_sweep")
    def test_runs_every_model_family_once(self, mock_sweep):
        from src.ml_engineering.model_configs import ModelConfiguration
        run_full_sweep(gold_table="t")
        keys = ModelConfiguration.get_all_keys()
        self.assertEqual(mock_sweep.call_count, len(keys))
        called = [c.kwargs["experiment_key"] for c in mock_sweep.call_args_list]
        self.assertEqual(called, keys)

    @patch("src.ml_engineering.ml_orchestrator.run_sector_sweep")
    def test_subset_and_resilient_to_one_failure(self, mock_sweep):
        mock_sweep.side_effect = [None, RuntimeError("boom")]  # 2nd family fails
        run_full_sweep(gold_table="t", model_keys=["baseline", "ridge"])
        self.assertEqual(mock_sweep.call_count, 2)  # did not abort the sweep


if __name__ == "__main__":
    unittest.main()
