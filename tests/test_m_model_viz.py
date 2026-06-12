"""
Unit tests for model performance visualizations.

Plot functions are smoke-tested (they must return a Matplotlib Figure and save
a PNG) using the non-interactive Agg backend; the data-prep core (leaderboard
ranking) is verified as a pure function.
"""
import unittest
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.m_model_viz import (  # noqa: E402
    leaderboard,
    plot_sector_leaderboard,
    plot_predicted_vs_actual,
    save_figure,
)


def _quality_df():
    return pd.DataFrame({
        "sector_code": ["301000", "T001081", "305700"],
        "model_family": ["HistGBR_Reduced", "Ridge_Reduced", "SectorQuarterRollingMean"],
        "champion_mape": [0.12, 0.05, 0.20],
        "baseline_mape": [0.18, 0.07, 0.20],
        "improvement": [0.33, 0.286, 0.0],
        "r2": [0.4, 0.6, -0.2],
        "tier": ["Good", "Good", "Poor"],
    })


def _predictions_df():
    dates = pd.date_range("2024-03-31", periods=4, freq="QE")
    return pd.DataFrame({
        "sector_code": ["T001081"] * 4,
        "target_date": dates,
        "y_true": [4.4, 3.7, 3.5, 3.9],
        "y_pred": [4.3, 3.8, 3.5, 4.0],
        "fold_set": ["outer"] * 4,
    })


class TestLeaderboard(unittest.TestCase):
    def test_ranks_by_champion_mape_ascending(self):
        lb = leaderboard(_quality_df())
        self.assertEqual(list(lb["sector_code"]), ["T001081", "301000", "305700"])
        self.assertEqual(lb.iloc[0]["rank"], 1)

    def test_empty_input_returns_empty(self):
        self.assertTrue(leaderboard(pd.DataFrame()).empty)


class TestPlots(unittest.TestCase):
    def test_plot_sector_leaderboard_returns_figure(self):
        fig = plot_sector_leaderboard(_quality_df())
        self.assertIsInstance(fig, Figure)

    def test_plot_predicted_vs_actual_returns_figure(self):
        fig = plot_predicted_vs_actual(_predictions_df(), sector_code="T001081")
        self.assertIsInstance(fig, Figure)

    def test_save_figure_writes_png(self):
        fig = plot_sector_leaderboard(_quality_df())
        with tempfile.TemporaryDirectory() as d:
            out = save_figure(fig, Path(d) / "figures" / "leaderboard.png")
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
