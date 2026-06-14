"""Tests for m_pipeline_loader.load_families_from_eval_db.

Builds a tiny in-memory-style SQLite ``model_predictions`` table (two stat/ML
families + baseline, two sectors, honest inner/outer folds) and checks the
per-family loader's contract: one canonical-schema frame per family with the
reducer collapsed to "Pipeline", the baseline split out, outer-fold rows only.
"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from src.utils.m_evaluation import CANONICAL_COLS
from src.utils.m_pipeline_loader import load_families_from_eval_db

# Explicit quarter-ends indexed by origin/target step (avoids offset ambiguity)
_QENDS = pd.to_datetime([
    "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
    "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31", "2023-03-31",
])
_FIRST_OUTER = pd.Timestamp("2021-09-30")  # _QENDS[2] — inner origins are 0,1


def _make_db(path: Path) -> None:
    rows = []
    families = {
        "AutoETS_Stat": 0.05,            # stays its own family
        "RidgeDeseason_Reduced": 0.10,   # collapses into "Pipeline"
        "SectorQuarterRollingMean": 0.20,  # split out as "baseline"
    }
    for fam, off in families.items():
        for sec in ["301000", "305700"]:
            run_id = f"run_{fam}_{sec}"
            for fold, origins in [("inner", [0, 1]), ("outer", [2, 3, 4])]:
                for o in origins:
                    for h in [1, 2, 3, 4]:
                        rows.append(dict(
                            run_id=run_id,
                            model_name=f"{fam}_{sec}",
                            sector_code=sec,
                            origin_date=_QENDS[o],
                            target_date=_QENDS[o + h],
                            horizon=h,
                            fold_set=fold,
                            y_true=5.0,
                            y_pred=5.0 + off,
                        ))
    engine = create_engine(f"sqlite:///{path.as_posix()}")
    pd.DataFrame(rows).to_sql("model_predictions", engine, index=False)
    engine.dispose()


class TestLoadFamilies(unittest.TestCase):
    def test_family_split_collapse_and_outer_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "eval.db"
            _make_db(db)
            family_dfs, baseline_df, winners = load_families_from_eval_db(db)

        # AutoETS stays; the reducer collapses into "Pipeline"; baseline is separate
        names = sorted(d["model_name"].iloc[0] for d in family_dfs)
        self.assertEqual(names, ["AutoETS_Stat", "Pipeline"])
        self.assertEqual(set(winners.keys()), {"AutoETS_Stat", "Pipeline"})

        # baseline split out under its own name
        self.assertFalse(baseline_df.empty)
        self.assertEqual(baseline_df["model_name"].unique().tolist(), ["baseline"])

        for d in family_dfs + [baseline_df]:
            self.assertEqual(list(d.columns), CANONICAL_COLS)
            self.assertGreater(len(d), 0)
            self.assertEqual(d["sector_code"].nunique(), 2)
            # OUTER folds only — no inner-origin rows leaked in
            self.assertGreaterEqual(d["origin_date"].min(), _FIRST_OUTER)
            # PI columns are NaN (eval DB stores only the median)
            self.assertTrue(d["y_lower_80"].isna().all())

    def test_pipeline_families_override(self):
        """Restricting pipeline_families changes what collapses to 'Pipeline'."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "eval.db"
            _make_db(db)
            # Treat no family as a reducer → RidgeDeseason_Reduced stays its own family
            family_dfs, _, _ = load_families_from_eval_db(db, pipeline_families=())
        names = sorted(d["model_name"].iloc[0] for d in family_dfs)
        self.assertEqual(names, ["AutoETS_Stat", "RidgeDeseason_Reduced"])


if __name__ == "__main__":
    unittest.main()
