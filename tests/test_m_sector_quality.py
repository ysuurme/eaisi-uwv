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
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from src.utils.m_sector_quality import (
    _division_section_letter,
    _feature_group_label,
    _find_importance_vector,
    _section_letter,
    _sector_range_letters,
    assign_tier,
    build_experiment_matrix,
    build_narrative_markdown,
    build_sector_quality_table,
    enrich_with_hierarchy,
    load_sector_performance,
    per_horizon_mape,
    sound_result_sectors,
    to_tree,
    write_sector_performance,
)


def _cbs_hierarchy_df():
    """Realistic multi-level fixture mirroring actual CBS dimension titles:
    totaal → sector (letter range) → section (letter) → subdivision (division)."""
    return pd.DataFrame({
        "sector_code":  ["T001081", "300003", "307500", "307610", "317105",
                         "354200", "354300", "305700", "WP19078", "999999"],
        "sbi_title":    ["A-U Alle economische activiteiten", "B-F Nijverheid en energie",
                         "C Industrie", "10-12 Voedings-, genotmiddelenindustrie",
                         "17-18 Papier- en grafische industrie", "G Handel",
                         "45 Autohandel en -reparatie", "B Delfstoffenwinning",
                         "WP 0-10 werknemers", "Mystery sector"],
        "sbi_level":    ["totaal", "sector", "section", "subdivision", "subdivision",
                         "section", "subdivision", "section", "size", "section"],
        "model_family": ["RidgeDeseason_Reduced"] * 10,
        "model_type":   ["Recursive"] * 10,
        "champion_mape": [0.05] * 10,
        "baseline_mape": [0.07] * 10,
        "improvement":  [0.28] * 10,
        "tier":         ["Good"] * 10,
    })


def _child_codes(node):
    return [c["sector_code"] for c in node["children"]]


def _find(node, code):
    if node.get("sector_code") == code:
        return node
    for child in node.get("children", []):
        hit = _find(child, code)
        if hit is not None:
            return hit
    return None


def _count_nodes(node):
    return 1 + sum(_count_nodes(c) for c in node.get("children", []))


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


class TestSbiTitleParsers(unittest.TestCase):
    def test_sector_range_letters(self):
        self.assertEqual(_sector_range_letters("B-F Nijverheid en energie"),
                         ["B", "C", "D", "E", "F"])
        self.assertEqual(_sector_range_letters("O-U Niet-commercieel"), ["O", "P", "Q", "R", "S", "T", "U"])
        self.assertEqual(_sector_range_letters("G Handel"), [])  # single letter is not a range

    def test_section_letter(self):
        self.assertEqual(_section_letter("G Handel"), "G")
        self.assertEqual(_section_letter("A Landbouw, bosbouw en visserij"), "A")
        self.assertIsNone(_section_letter("B-F Nijverheid"))   # a range is a sector, not a section
        self.assertIsNone(_section_letter("Agriculture"))      # 'A' not followed by a boundary

    def test_division_section_letter(self):
        self.assertEqual(_division_section_letter("10-12 Voedingsindustrie"), "C")  # div 10 → C
        self.assertEqual(_division_section_letter("45 Autohandel"), "G")            # div 45 → G
        self.assertEqual(_division_section_letter("24-30, 33 Metaal-elektro"), "C")  # div 24 → C
        self.assertIsNone(_division_section_letter("no number here"))


class TestToTreeNesting(unittest.TestCase):
    def test_multilevel_parent_child_edges(self):
        """national → sector → section → subdivision are correctly nested."""
        tree = to_tree(_cbs_hierarchy_df())
        self.assertEqual(tree["sector_code"], "T001081")

        sector_bf = _find(tree, "300003")          # 'B-F' sector, child of root
        self.assertIn("300003", _child_codes(tree))
        section_c = _find(sector_bf, "307500")      # 'C Industrie' under the B-F sector
        self.assertIsNotNone(section_c)
        self.assertIn("307500", _child_codes(sector_bf))
        # subdivisions 10-12 and 17-18 (divisions → section C) nest under section C
        self.assertEqual(set(_child_codes(section_c)), {"307610", "317105"})
        # section 'B Delfstoffenwinning' also under the B-F sector
        self.assertIn("305700", _child_codes(sector_bf))

    def test_section_without_present_sector_falls_back_to_root(self):
        """'G Handel' has no 'G-N' sector in the data → attaches to the root,
        but its subdivision (45 → G) still nests under it (nearest ancestor)."""
        tree = to_tree(_cbs_hierarchy_df())
        self.assertIn("354200", _child_codes(tree))      # section G under root
        section_g = _find(tree, "354200")
        self.assertEqual(_child_codes(section_g), ["354300"])  # subdivision 45 under section G

    def test_size_and_unmapped_attach_to_root_never_dropped(self):
        tree = to_tree(_cbs_hierarchy_df())
        self.assertIn("WP19078", _child_codes(tree))     # business-size node → root
        self.assertIn("999999", _child_codes(tree))      # unparseable title → root
        # every input row appears exactly once, nothing dropped
        self.assertEqual(_count_nodes(tree), len(_cbs_hierarchy_df()))

    def test_each_node_keeps_leading_attributes(self):
        tree = to_tree(_cbs_hierarchy_df())
        node = _find(tree, "307610")
        for key in ["sector_code", "sbi_level", "model_family", "model_type",
                    "champion_mape", "baseline_mape", "improvement", "tier"]:
            self.assertIn(key, node)

    def test_synthetic_root_when_no_total(self):
        """With no totaal row, a synthetic 'ALL' root holds everything; no drops.
        The synthetic root is an extra node, so the count is len(df) + 1."""
        df = _cbs_hierarchy_df()
        df = df[df["sbi_level"] != "totaal"].reset_index(drop=True)
        tree = to_tree(df)
        self.assertEqual(tree["sector_code"], "ALL")
        self.assertEqual(_count_nodes(tree), len(df) + 1)
        # every original sector is still reachable
        for code in df["sector_code"]:
            self.assertIsNotNone(_find(tree, code))

    def test_empty_frame_returns_empty_dict(self):
        self.assertEqual(to_tree(pd.DataFrame()), {})


# ---------------------------------------------------------------------------
# Experiment-matrix aggregations + narrative (consolidated from the retired
# m_experiment_matrix module — read-only over MLflow + the eval DB)
# ---------------------------------------------------------------------------

class TestFeatureGroupLabel(unittest.TestCase):
    def test_json_list_joined(self):
        self.assertEqual(_feature_group_label('["all_survivors"]'), "all_survivors")
        self.assertEqual(_feature_group_label('["a", "b"]'), "a+b")

    def test_discovery_and_missing(self):
        self.assertEqual(_feature_group_label("discovery"), "discovery")
        self.assertEqual(_feature_group_label(None), "discovery")
        self.assertEqual(_feature_group_label(float("nan")), "discovery")
        self.assertEqual(_feature_group_label(""), "discovery")


class TestBuildExperimentMatrix(unittest.TestCase):
    def _runs(self):
        return pd.DataFrame({
            "model_family":  ["Ridge", "Ridge", "Baseline", "Baseline", "Linear"],
            "feature_group": ["all_survivors", "all_survivors", "discovery", "discovery", "all_survivors"],
            "sector":        ["A", "B", "A", "B", "A"],
            "mape":          [0.05, 0.07, 0.06, 0.06, 0.20],
        })

    def test_median_mape_and_sector_wins(self):
        mape, wins = build_experiment_matrix(self._runs())
        self.assertAlmostEqual(mape.loc["Ridge", "all_survivors"], 0.06)   # median(.05,.07)
        self.assertEqual(wins.loc["Ridge", "all_survivors"], 1)            # sector A best
        self.assertEqual(wins.loc["Baseline", "discovery"], 1)            # sector B best
        self.assertEqual(wins.to_numpy().sum(), 2)                         # one win per sector

    def test_empty_runs_return_empty(self):
        mape, wins = build_experiment_matrix(pd.DataFrame())
        self.assertTrue(mape.empty and wins.empty)


class TestPerHorizonMape(unittest.TestCase):
    def test_outer_folds_only_and_per_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "eval.db"
            engine = create_engine(f"sqlite:///{db.as_posix()}")
            pd.DataFrame({
                "horizon":  [1, 1, 2, 2, 1],
                "y_true":   [10.0, 10.0, 10.0, 10.0, 10.0],
                "y_pred":   [9.0, 11.0, 8.0, 12.0, 5.0],
                "fold_set": ["outer", "outer", "outer", "outer", "inner"],
            }).to_sql("model_predictions", engine, index=False)
            engine.dispose()
            out = per_horizon_mape(db)

        h1 = out[out["horizon"] == 1].iloc[0]
        self.assertAlmostEqual(h1["mape"], 0.10)  # inner row excluded
        self.assertEqual(h1["n"], 2)
        self.assertAlmostEqual(out[out["horizon"] == 2].iloc[0]["mape"], 0.20)

    def test_missing_table_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "empty.db"
            create_engine(f"sqlite:///{db.as_posix()}").dispose()
            self.assertTrue(per_horizon_mape(db).empty)


class TestImportanceVector(unittest.TestCase):
    def test_finds_coef_in_nested_pipeline(self):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([("s", StandardScaler()), ("r", Ridge())])
        pipe.fit(np.arange(20).reshape(10, 2).astype(float), np.arange(10.0))
        vec = _find_importance_vector(pipe)
        self.assertIsNotNone(vec)
        self.assertEqual(vec.shape[0], 2)

    def test_returns_none_when_no_flat_vector(self):
        class StatLike:  # no coef_/feature_importances_ anywhere (e.g. AutoETS)
            pass
        self.assertIsNone(_find_importance_vector(StatLike()))
        self.assertIsNone(_find_importance_vector(None))


class TestNarrativeMarkdown(unittest.TestCase):
    @patch("src.utils.m_sector_quality.per_horizon_mape", return_value=pd.DataFrame())
    @patch("src.utils.m_sector_quality.load_forecasts", return_value=pd.DataFrame())
    @patch("src.utils.m_sector_quality.load_runs", return_value=pd.DataFrame())
    @patch("src.utils.m_sector_quality.load_sector_performance", return_value=pd.DataFrame())
    def test_degrades_gracefully_on_empty_sources(self, *_mocks):
        md = build_narrative_markdown(eval_db_path="unused.db")
        self.assertIn("# Sick-leave forecasting", md)
        self.assertIn("## Headline", md)
        self.assertIn("No champions registered yet", md)
        self.assertIn("## Caveats & notes", md)

    @patch("src.utils.m_sector_quality.per_horizon_mape", return_value=pd.DataFrame())
    @patch("src.utils.m_sector_quality.load_runs", return_value=pd.DataFrame())
    @patch("src.utils.m_sector_quality.load_sector_performance")
    def test_renders_champion_and_forecast_tables(self, mock_perf, mock_runs, mock_horizon):
        mock_perf.return_value = pd.DataFrame([{
            "sector_code": "T001081", "sbi_title": "All industries",
            "model_family": "RidgeDeseason_Reduced", "model_type": "Recursive",
            "champion_mape": 0.0566, "baseline_mape": 0.0729,
            "improvement": 0.2229, "r2": -0.38, "tier": "Good",
        }])
        forecasts = pd.DataFrame({
            "sector_code": ["T001081"], "model_family": ["RidgeDeseason_Reduced"],
            "target_date": pd.to_datetime(["2025-12-31"]), "horizon": [1], "y_pred": [5.74],
        })
        with patch("src.utils.m_sector_quality.load_forecasts", return_value=forecasts):
            md = build_narrative_markdown(eval_db_path="unused.db")
        self.assertIn("1 Good", md)
        self.assertIn("RidgeDeseason_Reduced", md)
        self.assertIn("22.29%", md)
        self.assertIn("2025-12-31", md)
        self.assertIn("5.74%", md)


if __name__ == "__main__":
    unittest.main()
