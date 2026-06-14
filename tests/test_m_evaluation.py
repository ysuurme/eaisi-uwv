"""Tests for src/utils/m_evaluation.py — the canonical evaluation module.

Two responsibilities are exercised: (1) the metric primitives match the exact
sklearn calls ml_5 uses (so adopting them anywhere is value-preserving), and
(2) the cross-method comparison engine still runs end-to-end after the move.
"""
import math
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from src.utils import m_evaluation as me


class TestPrimitivesMatchSklearn(unittest.TestCase):
    def setUp(self):
        self.yt = np.array([5.0, 5.2, 4.8, 5.1, 4.9, 5.3])
        self.yp = np.array([5.1, 5.0, 4.9, 5.0, 5.0, 5.2])

    def test_mae(self):
        self.assertEqual(me.mae(self.yt, self.yp), mean_absolute_error(self.yt, self.yp))

    def test_rmse(self):
        self.assertEqual(me.rmse(self.yt, self.yp),
                         float(np.sqrt(mean_squared_error(self.yt, self.yp))))

    def test_mape_matches_sklearn_and_is_fraction(self):
        self.assertEqual(me.mape(self.yt, self.yp),
                         mean_absolute_percentage_error(self.yt, self.yp))
        self.assertLess(me.mape(self.yt, self.yp), 1.0)  # fraction (0.0x), not percent

    def test_r2(self):
        self.assertEqual(me.r2(self.yt, self.yp), r2_score(self.yt, self.yp))

    def test_r2_undefined_for_single_point(self):
        self.assertTrue(math.isnan(me.r2([5.0], [5.1])))

    def test_seasonal_naive_mae_matches_ml5(self):
        from src.ml_engineering.ml_5_model_evaluation import _seasonal_naive_mae
        y = np.array([4.5, 5.0, 5.5, 6.0, 4.6, 5.1, 5.6, 6.1])
        self.assertEqual(me.seasonal_naive_mae(y, 4), _seasonal_naive_mae(y, 4))

    def test_mase_guards_short_train_window(self):
        # train window <= sp → scaler NaN → MASE NaN (no crash)
        self.assertTrue(math.isnan(me.mase([5.0, 5.1], [5.0, 5.0], [5.0], sp=4)))

    def test_directional_accuracy_is_fraction(self):
        da = me.directional_accuracy([1.0, 2.0, 1.0, 2.0], [1.0, 3.0, 2.0, 3.0])
        self.assertEqual(da, 1.0)  # every direction matches → 1.0 (fraction, not 100)


class TestPointMetrics(unittest.TestCase):
    def test_columns_and_grouping(self):
        df = pd.DataFrame({
            "model_name": ["A"] * 4 + ["B"] * 4,
            "y_true": [5.0, 5.1, 4.9, 5.0, 5.0, 5.1, 4.9, 5.0],
            "y_pred": [5.0, 5.0, 5.0, 5.0, 4.0, 6.0, 4.0, 6.0],
        })
        out = me.point_metrics(df)
        self.assertEqual(set(out["model_name"]), {"A", "B"})
        for col in ["n", "MAE", "RMSE", "MAPE", "R2", "bias", "dir_acc"]:
            self.assertIn(col, out.columns)
        a = out[out["model_name"] == "A"]["MAE"].iloc[0]
        b = out[out["model_name"] == "B"]["MAE"].iloc[0]
        self.assertLess(a, b)  # A is near-perfect

    def test_compute_point_metrics_alias(self):
        self.assertIs(me.compute_point_metrics, me.point_metrics)


class TestCompareSmoke(unittest.TestCase):
    def test_compare_all_models_runs_end_to_end(self):
        warnings.simplefilter("ignore")
        dates = pd.date_range("2021-03-31", periods=8, freq="QE")

        def mk(name, off):
            rows = []
            for sec in ["301000", "305700", "410200"]:
                for i, d in enumerate(dates):
                    yt = 5.0 + 0.1 * np.sin(i)
                    rows.append(dict(model_name=name, sector_code=sec, origin_date=d,
                                     target_date=d, horizon=(i % 4) + 1, y_true=yt,
                                     y_pred=yt + off, y_lower_80=np.nan, y_upper_80=np.nan,
                                     y_lower_95=np.nan, y_upper_95=np.nan))
            return pd.DataFrame(rows)

        sc = me.compare_all_models(
            [mk("AutoETS_Stat", 0.05), mk("STL_ETS", 0.08), mk("Pipeline", 0.12)],
            verbose=False,
        )
        self.assertEqual(len(sc.aligned_metrics), 3)
        # MAPE column is a fraction in the scorecard (consistent with ml_5)
        self.assertTrue((sc.aligned_metrics["MAPE"] < 1.0).all())
        # decision matrix builds without error
        self.assertFalse(me.make_decision_matrix(sc).empty)


if __name__ == "__main__":
    unittest.main()
