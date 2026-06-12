"""
Unit tests for sector forecast-quality bucketing (Good / Medium / Poor).

Tiers are benchmarked against the BASELINE model (SectorQuarterRollingMean, the
rolling 3-year same-quarter mean from 01_baselinemodel.py — "the model we need
to outperform"). A sector is Good only when the champion beats the baseline by a
margin; Poor when the champion fails to beat the baseline at all.

The registry (MlflowClient) is mocked at the boundary; the per-sector baseline
MAPE is injected as a dict; the tier logic is exercised as a pure function.
"""
import unittest
from unittest.mock import MagicMock

from src.utils.m_sector_quality import (
    assign_tier,
    build_sector_quality_table,
    sound_result_sectors,
)


def _registered(name):
    rm = MagicMock()
    rm.name = name
    return rm


def _version(mape, r2, family="Ridge_Reduced"):
    mv = MagicMock()
    mv.tags = {"mape": str(mape), "r2": str(r2), "model_family": family}
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
            {"sector_code", "model_family", "champion_mape", "baseline_mape",
             "improvement", "r2", "tier"},
        )

    def test_tiers_benchmarked_against_baseline(self):
        df = self._build()
        tier = dict(zip(df["sector_code"], df["tier"]))
        # 0.05 vs 0.07 → 28.6% better → Good
        self.assertEqual(tier["T001081"], "Good")
        # champion == baseline (no lift) → Poor
        self.assertEqual(tier["301000"], "Poor")

    def test_sound_result_sectors_lists_good_tier(self):
        self.assertEqual(sound_result_sectors(self._build()), ["T001081"])


if __name__ == "__main__":
    unittest.main()
