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
    plot_forecast,
    plot_matrix_heatmap,
    plot_horizon_curve,
    plot_forecast_overlay,
    plot_importance_bars,
    save_figure,
)


def _quality_df():
    return pd.DataFrame({
        "sector_code": ["301000", "T001081", "305700"],
        "model_family": ["HistGBR_Reduced", "Ridge_Reduced", "SectorQuarterRollingMean"],
        "mase": [0.85, 0.70, 1.10],
        "baseline_mase": [1.00, 1.00, 1.00],
        "champion_mae": [0.22, 0.13, 0.39],
        "champion_mape": [0.12, 0.05, 0.20],
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


def _history_df():
    dates = pd.date_range("2023-03-31", periods=8, freq="QE")
    return pd.DataFrame({
        "sector_code": ["T001081"] * 8,
        "target_date": dates,
        "y_true": [4.0, 3.6, 3.4, 3.8, 4.2, 3.7, 3.5, 3.9],
    })


def _forecast_df():
    dates = pd.date_range("2025-03-31", periods=4, freq="QE")
    return pd.DataFrame({
        "sector_code": ["T001081"] * 4,
        "target_date": dates,
        "y_pred": [4.1, 3.7, 3.5, 3.9],
        "horizon": [1, 2, 3, 4],
    })


class TestLeaderboard(unittest.TestCase):
    def test_ranks_by_mase_ascending(self):
        lb = leaderboard(_quality_df())
        # ascending MASE: T001081 (0.70), 301000 (0.85), 305700 (1.10)
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

    def test_plot_forecast_returns_figure(self):
        fig = plot_forecast(_history_df(), _forecast_df(), sector_code="T001081")
        self.assertIsInstance(fig, Figure)

    def test_plot_forecast_handles_missing_sector(self):
        """A sector with no history/forecast rows renders an empty-state figure,
        not an exception."""
        fig = plot_forecast(_history_df(), _forecast_df(), sector_code="999999")
        self.assertIsInstance(fig, Figure)

    def test_plot_matrix_heatmap_and_empty(self):
        mase = pd.DataFrame({"all_survivors": [0.70, 0.95]}, index=["Ridge", "Baseline"])
        wins = pd.DataFrame({"all_survivors": [1, 0]}, index=["Ridge", "Baseline"])
        self.assertIsInstance(plot_matrix_heatmap(mase, wins), Figure)
        self.assertIsInstance(plot_matrix_heatmap(pd.DataFrame()), Figure)

    def test_plot_horizon_curve_and_empty(self):
        df = pd.DataFrame({"horizon": [1, 2, 3, 4], "mape": [0.06, 0.1, 0.13, 0.16], "n": [3] * 4})
        self.assertIsInstance(plot_horizon_curve(df), Figure)
        self.assertIsInstance(plot_horizon_curve(pd.DataFrame()), Figure)

    def test_plot_forecast_overlay_and_empty(self):
        df = pd.DataFrame({
            "sector_code": ["T001081", "T001081"],
            "target_date": pd.to_datetime(["2025-12-31", "2026-03-31"]),
            "y_pred": [5.0, 5.5],
        })
        self.assertIsInstance(plot_forecast_overlay(df), Figure)
        self.assertIsInstance(plot_forecast_overlay(pd.DataFrame()), Figure)

    def test_plot_importance_bars(self):
        df = pd.DataFrame({"feature": ["a", "b"], "weight": [0.5, -0.3]})
        self.assertIsInstance(plot_importance_bars("T001081", df), Figure)

    def test_save_figure_writes_png(self):
        fig = plot_sector_leaderboard(_quality_df())
        with tempfile.TemporaryDirectory() as d:
            out = save_figure(fig, Path(d) / "figures" / "leaderboard.png")
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
