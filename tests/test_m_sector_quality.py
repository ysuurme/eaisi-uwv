"""
Unit tests for sector forecast-quality bucketing (Good / Medium / Poor).

Tiers are benchmarked against the BASELINE model (SectorQuarterRollingMean, the
rolling 3-year same-quarter mean from 01_baselinemodel.py — "the model we need
to outperform"). A sector is Good only when the champion beats the baseline by a
margin; Poor when the champion fails to beat the baseline at all.

The registry (MlflowClient) is mocked at the boundary; the per-sector baseline
MAPE is injected as a dict; the tier logic is exercised as a pure function.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from src.utils.m_sector_quality import (
    assign_tier,
    build_sector_quality_table,
    enrich_with_hierarchy,
    load_sector_performance,
    sound_result_sectors,
    to_tree,
    write_sector_performance,
)


def _enriched_df():
    return pd.DataFrame({
        "sector_code":   ["T001081", "301000", "305700"],
        "sbi_title":     ["All economic activities", "Agriculture", "Mining"],
        "sbi_level":     ["totaal", "section", "section"],
        "model_family":  ["Ridge_Reduced", "HistGBR_Reduced", "SectorQuarterRollingMean"],
        "model_type":    ["Ridge", "HistGradientBoostingRegressor", "SectorQuarterRollingMean"],
        "feature_groups": ['["all_survivors"]', '["labor_volume"]', "discovery"],
        "champion_mape": [0.05, 0.12, 0.20],
        "baseline_mape": [0.07, 0.18, 0.20],
        "improvement":   [0.286, 0.33, 0.0],
        "r2":            [0.6, 0.4, -0.2],
        "tier":          ["Good", "Good", "Poor"],
    })


def _registered(name):
    rm = MagicMock()
    rm.name = name
    return rm


def _version(mape, r2, family="Ridge_Reduced", model_type="Ridge", feature_groups='["all_survivors"]'):
    mv = MagicMock()
    mv.tags = {
        "mape": str(mape), "r2": str(r2), "model_family": family,
        "model_type": model_type, "feature_groups": feature_groups,
    }
    return mv


class TestAssignTier(unittest.TestCase):
    def test_good_when_champion_beats_baseline_by_margin(self):
        # 0.045 vs 0.060 → 25% better → Good
        self.assertEqual(assign_tier(champion_mape=0.045, baseline_mape=0.060), "Good")

    def test_good_boundary_exactly_ten_percent(self):
        # 0.054 vs 0.060 → exactly 10% better → Good (inclusive)
        self.assertEqual(assign_tier(champion_mape=0.054, baseline_mape=0.060), "Good")

    def test_medium_for_small_positive_improvement(self):
        # 0.057 vs 0.060 → 5% better → Medium
        self.assertEqual(assign_tier(champion_mape=0.057, baseline_mape=0.060), "Medium")

    def test_poor_when_no_improvement_over_baseline(self):
        self.assertEqual(assign_tier(champion_mape=0.060, baseline_mape=0.060), "Poor")

    def test_poor_when_worse_than_baseline(self):
        self.assertEqual(assign_tier(champion_mape=0.070, baseline_mape=0.060), "Poor")

    def test_non_finite_or_missing_baseline_is_poor(self):
        self.assertEqual(assign_tier(champion_mape=float("nan"), baseline_mape=0.06), "Poor")
        self.assertEqual(assign_tier(champion_mape=0.05, baseline_mape=None), "Poor")
        self.assertEqual(assign_tier(champion_mape=0.05, baseline_mape=0.0), "Poor")

    def test_margin_is_configurable(self):
        # 5% improvement clears a 3% bar → Good
        self.assertEqual(
            assign_tier(champion_mape=0.057, baseline_mape=0.060, good_improvement=0.03),
            "Good",
        )


class TestBuildSectorQualityTable(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.prefix = "master_SickLeave_4Q_"
        self.client.search_registered_models.return_value = [
            _registered("master_SickLeave_4Q_T001081"),
            _registered("master_SickLeave_4Q_301000"),
            _registered("some_other_model"),  # ignored (wrong prefix)
        ]

        def by_alias(name, alias):
            return {
                "master_SickLeave_4Q_T001081": _version(0.05, 0.6, "HistGBR_Reduced"),
                "master_SickLeave_4Q_301000": _version(0.30, -1.0, "SectorQuarterRollingMean"),
            }[name]

        self.client.get_model_version_by_alias.side_effect = by_alias
        # Baseline MAPE per sector (from SectorQuarterRollingMean runs).
        self.baseline = {"T001081": 0.07, "301000": 0.30}

    def _build(self):
        return build_sector_quality_table(
            self.client, self.baseline, registered_model_prefix=self.prefix
        )

    def test_one_row_per_sector_champion_with_benchmark_columns(self):
        df = self._build()
        self.assertEqual(set(df["sector_code"]), {"T001081", "301000"})
        self.assertEqual(
            set(df.columns),
            {"sector_code", "model_family", "model_type", "feature_groups",
             "champion_mape", "baseline_mape", "improvement", "r2", "tier"},
        )

    def test_champion_self_description_columns_populated(self):
        df = self._build()
        row = df[df["sector_code"] == "T001081"].iloc[0]
        self.assertEqual(row["model_type"], "Ridge")
        self.assertEqual(row["feature_groups"], '["all_survivors"]')

    def test_tiers_benchmarked_against_baseline(self):
        df = self._build()
        tier = dict(zip(df["sector_code"], df["tier"]))
        # 0.05 vs 0.07 → 28.6% better → Good
        self.assertEqual(tier["T001081"], "Good")
        # champion == baseline (no lift) → Poor
        self.assertEqual(tier["301000"], "Poor")

    def test_sound_result_sectors_lists_good_tier(self):
        self.assertEqual(sound_result_sectors(self._build()), ["T001081"])


class TestSectorHierarchyStructure(unittest.TestCase):
    def test_enrich_with_hierarchy_adds_title_and_level(self):
        q = pd.DataFrame({"sector_code": ["T001081", "301000"], "champion_mape": [0.05, 0.12]})
        hierarchy = {
            "T001081": {"sbi_title": "All", "sbi_level": "totaal"},
            "301000": {"sbi_title": "Agri", "sbi_level": "section"},
        }
        out = enrich_with_hierarchy(q, hierarchy)
        self.assertEqual(out.loc[out.sector_code == "301000", "sbi_level"].iloc[0], "section")
        self.assertEqual(out.loc[out.sector_code == "T001081", "sbi_title"].iloc[0], "All")

    def test_enrich_unknown_sector_is_labelled_not_dropped(self):
        q = pd.DataFrame({"sector_code": ["999999"], "champion_mape": [0.1]})
        out = enrich_with_hierarchy(q, {})
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["sbi_level"], "unknown")

    def test_to_tree_roots_at_national_total_and_is_json_serializable(self):
        tree = to_tree(_enriched_df())
        self.assertEqual(tree["sector_code"], "T001081")
        self.assertEqual(len(tree["children"]), 2)
        # leading attributes are present on every node
        child = tree["children"][0]
        for key in ["sector_code", "sbi_level", "model_family", "model_type",
                    "champion_mape", "baseline_mape", "improvement", "tier"]:
            self.assertIn(key, child)
        json.dumps(tree)  # must not raise — JSON-serializable (NaN → None)

    def test_write_then_load_sector_performance_roundtrip(self):
        enriched = _enriched_df()
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "eval.db"
            n = write_sector_performance(enriched, db)
            self.assertEqual(n, len(enriched))
            loaded = load_sector_performance(db)
            self.assertEqual(set(loaded["sector_code"]), set(enriched["sector_code"]))
            for col in ["model_type", "feature_groups", "tier", "improvement", "sbi_level"]:
                self.assertIn(col, loaded.columns)

    def test_write_is_idempotent_replace(self):
        enriched = _enriched_df()
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "eval.db"
            write_sector_performance(enriched, db)
            write_sector_performance(enriched.head(1), db)  # replace, not append
            loaded = load_sector_performance(db)
            self.assertEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()
