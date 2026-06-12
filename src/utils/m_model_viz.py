"""
Model performance visualizations.

A small, reproducible set of charts that communicate forecast quality to
technical reviewers and business stakeholders, sourced from the per-sector
quality table (``m_sector_quality``), the MLflow registry, and the per-row
walk-forward predictions in ``model_predictions``:

* ``plot_sector_leaderboard`` — per-sector champion MAPE bars, coloured by
  Good/Medium/Poor tier, with the baseline MAPE marked for reference.
* ``plot_predicted_vs_actual`` — the champion's 4Q-ahead walk-forward
  predictions overlaid on the realised actuals for one sector.

Figures are returned as Matplotlib ``Figure`` objects and saved as PNGs under
``reports/figures/`` via ``save_figure``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

try:
    from src.config import C_BLUE, C_ORANGE, C_GREY
except Exception:  # pragma: no cover - palette is cosmetic
    C_BLUE, C_ORANGE, C_GREY = "#3B82F6", "#F59E0B", "#6B7280"

_C_GOOD = "#16A34A"     # green
_C_MEDIUM = C_ORANGE    # amber
_C_POOR = C_GREY        # grey
_TIER_COLORS = {"Good": _C_GOOD, "Medium": _C_MEDIUM, "Poor": _C_POOR}


def leaderboard(quality_df: pd.DataFrame) -> pd.DataFrame:
    """Rank sectors by champion MAPE (ascending = best first), adding ``rank``."""
    if quality_df is None or quality_df.empty:
        return pd.DataFrame()
    lb = quality_df.sort_values("champion_mape").reset_index(drop=True)
    lb.insert(0, "rank", range(1, len(lb) + 1))
    return lb


def plot_sector_leaderboard(
    quality_df: pd.DataFrame,
    *,
    title: str = "Sector forecast quality — champion MAPE vs baseline",
) -> Figure:
    """Horizontal bar chart of champion MAPE per sector, coloured by tier.

    Each bar is annotated with R² and tier; a black tick marks the baseline
    MAPE so the champion's improvement (or lack of it) is visible at a glance.
    """
    lb = leaderboard(quality_df)
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.5 * len(lb) + 1.5)))
    if lb.empty:
        ax.text(0.5, 0.5, "No sector champions to display", ha="center", va="center")
        return fig

    y = list(range(len(lb)))
    colors = [_TIER_COLORS.get(t, _C_POOR) for t in lb["tier"]]
    ax.barh(y, lb["champion_mape"] * 100.0, color=colors, zorder=2)
    ax.scatter(
        lb["baseline_mape"] * 100.0, y,
        marker="|", color="black", s=240, zorder=3, label="baseline MAPE",
    )
    for i, row in lb.reset_index(drop=True).iterrows():
        ax.text(
            row["champion_mape"] * 100.0, i,
            f"  R²={row['r2']:.2f} · {row['tier']}",
            va="center", ha="left", fontsize=8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(lb["sector_code"])
    ax.invert_yaxis()
    ax.set_xlabel("MAPE (%) — lower is better")
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _TIER_COLORS.values()]
    ax.legend(
        handles + [plt.Line2D([0], [0], color="black", marker="|", linestyle="none")],
        list(_TIER_COLORS.keys()) + ["baseline MAPE"],
        loc="lower right", fontsize=8,
    )
    fig.tight_layout()
    return fig


def plot_predicted_vs_actual(
    predictions_df: pd.DataFrame,
    sector_code: str,
    *,
    fold_set: Optional[str] = "outer",
) -> Figure:
    """Overlay the champion's 4Q-ahead predictions on actuals for one sector."""
    df = predictions_df[predictions_df["sector_code"].astype(str) == str(sector_code)]
    if fold_set and "fold_set" in df.columns:
        df = df[df["fold_set"] == fold_set]
    df = df.sort_values("target_date")

    fig, ax = plt.subplots(figsize=(11, 5))
    if df.empty:
        ax.text(0.5, 0.5, f"No predictions for sector {sector_code}", ha="center", va="center")
        return fig

    ax.plot(df["target_date"], df["y_true"], marker="o", color=C_ORANGE,
            linewidth=2, label="Actual")
    ax.plot(df["target_date"], df["y_pred"], marker="o", color=C_BLUE,
            linewidth=2, linestyle="--", label="Predicted (4Q-ahead)")
    ax.set_title(f"Sector {sector_code} — predicted vs actual ({fold_set or 'all'} folds)")
    ax.set_xlabel("Quarter end date")
    ax.set_ylabel("Sick leave (%)")
    ax.legend(loc="upper left")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: Path) -> Path:
    """Save a figure to ``path`` (parents created), returning the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all(eval_db_path: Optional[Path] = None, out_dir: Optional[Path] = None) -> List[Path]:
    """Build the quality table + per-row predictions and save the figure set."""
    import mlflow
    from mlflow.tracking import MlflowClient
    from src.utils.m_sector_quality import (
        baseline_mape_by_sector, build_sector_quality_table,
    )

    if eval_db_path is None:
        from src.config import DIR_DB_EVAL
        eval_db_path = DIR_DB_EVAL
    if out_dir is None:
        from src.config import PROJECT_ROOT
        out_dir = PROJECT_ROOT / "reports" / "figures"

    mlflow.set_tracking_uri(f"sqlite:///{Path(eval_db_path).as_posix()}?timeout=30")
    client = MlflowClient()
    quality = build_sector_quality_table(client, baseline_mape_by_sector(eval_db_path))

    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{Path(eval_db_path).as_posix()}")
    preds = pd.read_sql("SELECT * FROM model_predictions", engine)
    preds["target_date"] = pd.to_datetime(preds["target_date"])

    saved: List[Path] = [save_figure(plot_sector_leaderboard(quality), out_dir / "sector_leaderboard.png")]
    for sector in quality["sector_code"] if not quality.empty else []:
        fig = plot_predicted_vs_actual(preds, sector)
        saved.append(save_figure(fig, out_dir / f"predicted_vs_actual_{sector}.png"))
    return saved


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    figs = generate_all()
    print(f"Saved {len(figs)} figures:")
    for p in figs:
        print(f"  {p}")
