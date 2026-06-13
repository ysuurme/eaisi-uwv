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
* ``plot_forecast`` — the champion's forward 4Q forecast (future quarters that
  have not happened) overlaid on the sector's observed history.
* ``plot_matrix_heatmap`` / ``plot_horizon_curve`` / ``plot_forecast_overlay`` /
  ``plot_importance_bars`` — the experiment-comparison views; their data is
  aggregated by ``m_sector_quality`` (read-only over MLflow + the eval DB).

Figures are returned as Matplotlib ``Figure`` objects and saved as PNGs under
``reports/figures/`` via ``save_figure``.  ``generate_all`` regenerates the full
``--report`` figure bundle (leaderboard, predicted-vs-actual, and the experiment
matrix / horizon / overlay / importance views) in one pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
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


def plot_forecast(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    sector_code: str,
) -> Figure:
    """Overlay a sector's forward 4Q champion forecast on its observed history.

    ``history_df`` carries ``target_date`` + ``y_true`` (the realised sick-leave
    series); ``forecast_df`` carries ``target_date`` + ``y_pred`` (the forward
    forecast).  Either frame may include a ``sector_code`` column, in which case
    it is filtered to ``sector_code`` first.  A dashed bridge connects the last
    observed point to the first forecast point for visual continuity.
    """
    def _for_sector(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if "sector_code" in df.columns:
            df = df[df["sector_code"].astype(str) == str(sector_code)]
        return df.sort_values("target_date")

    hist = _for_sector(history_df)
    fc = _for_sector(forecast_df)

    fig, ax = plt.subplots(figsize=(11, 5))
    if hist.empty and fc.empty:
        ax.text(0.5, 0.5, f"No data for sector {sector_code}", ha="center", va="center")
        return fig

    if not hist.empty:
        ax.plot(hist["target_date"], hist["y_true"], marker="o", color=C_ORANGE,
                linewidth=2, label="Observed")
    if not fc.empty:
        if not hist.empty:
            ax.plot(
                [hist["target_date"].iloc[-1], fc["target_date"].iloc[0]],
                [hist["y_true"].iloc[-1], fc["y_pred"].iloc[0]],
                color=C_BLUE, linewidth=2, linestyle="--", zorder=1,
            )
        ax.plot(fc["target_date"], fc["y_pred"], marker="s", color=C_BLUE,
                linewidth=2, linestyle="--", label=f"Forecast ({len(fc)}Q ahead)")

    ax.set_title(f"Sector {sector_code} — forward forecast vs observed history")
    ax.set_xlabel("Quarter end date")
    ax.set_ylabel("Sick leave (%)")
    ax.legend(loc="upper left")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    return fig


def plot_matrix_heatmap(mape_matrix: pd.DataFrame, win_matrix: Optional[pd.DataFrame] = None) -> Figure:
    """Heatmap of median outer MAPE (model family × feature group), wins annotated.

    Data comes from ``m_sector_quality.build_experiment_matrix``; green = low
    MAPE (good), red = high.  ``★N`` marks how many sectors each cell wins.
    """
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * (mape_matrix.shape[1] + 1)),
                                     max(4, 0.6 * mape_matrix.shape[0] + 2)))
    if mape_matrix is None or mape_matrix.empty:
        ax.text(0.5, 0.5, "No runs to compare", ha="center", va="center")
        return fig

    data = mape_matrix.to_numpy(dtype=float) * 100.0
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(mape_matrix.shape[1]))
    ax.set_xticklabels(mape_matrix.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(mape_matrix.shape[0]))
    ax.set_yticklabels(mape_matrix.index, fontsize=8)
    for i in range(mape_matrix.shape[0]):
        for j in range(mape_matrix.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                wins = ""
                if win_matrix is not None and not win_matrix.empty:
                    w = int(win_matrix.iloc[i, j])
                    wins = f"\n★{w}" if w else ""
                ax.text(j, i, f"{val:.1f}%{wins}", ha="center", va="center", fontsize=7)
    ax.set_title("Median outer MAPE — model family × feature group (★ = sector wins)")
    fig.colorbar(im, ax=ax, label="MAPE (%)")
    fig.tight_layout()
    return fig


def plot_horizon_curve(horizon_df: pd.DataFrame) -> Figure:
    """MAPE vs forecast horizon (outer folds) — the forecast-decay curve."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if horizon_df is None or horizon_df.empty:
        ax.text(0.5, 0.5, "No outer-fold predictions", ha="center", va="center")
        return fig
    ax.plot(horizon_df["horizon"], horizon_df["mape"] * 100.0, marker="o",
            linewidth=2, color=C_BLUE)
    ax.set_xticks(horizon_df["horizon"])
    ax.set_xlabel("Forecast horizon (quarters ahead)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Forecast decay — outer-fold MAPE by horizon")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_forecast_overlay(forecasts_df: pd.DataFrame) -> Figure:
    """Forward 4Q trajectories per sector from ``model_forecasts``."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if forecasts_df is None or forecasts_df.empty:
        ax.text(0.5, 0.5, "No forecasts persisted", ha="center", va="center")
        return fig
    for sector, g in forecasts_df.sort_values("target_date").groupby("sector_code"):
        ax.plot(g["target_date"], g["y_pred"], marker="o", linewidth=1.5, label=str(sector))
    ax.set_xlabel("Quarter end date")
    ax.set_ylabel("Forecast sick leave (%)")
    ax.set_title("Forward 4Q forecasts by sector (@prod champions)")
    if forecasts_df["sector_code"].nunique() <= 12:
        ax.legend(loc="best", fontsize=8, ncol=2)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    return fig


def plot_importance_bars(sector: str, importance_df: pd.DataFrame) -> Figure:
    """Horizontal bar chart of a reducer champion's top coefficients/importances."""
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(importance_df) + 1)))
    if importance_df is None or importance_df.empty:
        ax.text(0.5, 0.5, f"No importances for {sector}", ha="center", va="center")
        return fig
    d = importance_df.iloc[::-1]
    ax.barh(range(len(d)), d["weight"], color=C_BLUE)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["feature"], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient / importance")
    ax.set_title(f"Sector {sector} — champion feature weights (top {len(d)})")
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: Path) -> Path:
    """Save a figure to ``path`` (parents created), returning the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all(
    eval_db_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    gold_table: str = "master_data_ml_preprocessed",
) -> List[Path]:
    """Regenerate the full ``--report`` figure bundle (+ the matrix CSV).

    Sources every view from the single sources of truth — the MLflow registry
    and the eval-DB read tables — via ``m_sector_quality`` aggregations, and
    saves: the sector leaderboard, per-sector predicted-vs-actual, the
    model-family × feature-group matrix heatmap (+ ``experiment_matrix.csv``),
    the per-horizon decay curve, the forward-forecast overlay, and best-effort
    champion-importance bars.  Each artifact is independent — an empty source
    yields an empty-state figure rather than aborting the bundle.
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    from src.utils import m_sector_quality as msq

    if eval_db_path is None:
        from src.config import DIR_DB_EVAL
        eval_db_path = DIR_DB_EVAL
    if out_dir is None:
        from src.config import PROJECT_ROOT
        out_dir = PROJECT_ROOT / "reports" / "figures"
    out_dir = Path(out_dir)

    mlflow.set_tracking_uri(f"sqlite:///{Path(eval_db_path).as_posix()}?timeout=30")
    client = MlflowClient()
    quality = msq.build_sector_quality_table(client, msq.baseline_mape_by_sector(eval_db_path))

    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{Path(eval_db_path).as_posix()}")
    try:
        preds = pd.read_sql("SELECT * FROM model_predictions", engine)
    except Exception:
        preds = pd.DataFrame()
    finally:
        engine.dispose()  # release the SQLite handle (Windows-safe)
    if not preds.empty:
        preds["target_date"] = pd.to_datetime(preds["target_date"])

    # --- Standard per-sector figures ---
    saved: List[Path] = [save_figure(plot_sector_leaderboard(quality), out_dir / "sector_leaderboard.png")]
    for sector in (quality["sector_code"] if not quality.empty else []):
        saved.append(save_figure(plot_predicted_vs_actual(preds, sector),
                                 out_dir / f"predicted_vs_actual_{sector}.png"))

    # --- Experiment-comparison views (data aggregated by m_sector_quality) ---
    mape_matrix, win_matrix = msq.build_experiment_matrix(msq.load_runs(eval_db_path))
    if not mape_matrix.empty:
        csv_path = out_dir.parent / "experiment_matrix.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        mape_matrix.to_csv(csv_path)
        saved.append(csv_path)
    saved.append(save_figure(plot_matrix_heatmap(mape_matrix, win_matrix),
                             out_dir / "experiment_matrix.png"))
    saved.append(save_figure(plot_horizon_curve(msq.per_horizon_mape(eval_db_path)),
                             out_dir / "horizon_mape.png"))
    saved.append(save_figure(plot_forecast_overlay(msq.load_forecasts(eval_db_path)),
                             out_dir / "forecast_overlay.png"))
    for sector, frame in msq.champion_importances(gold_table, eval_db_path).items():
        saved.append(save_figure(plot_importance_bars(sector, frame),
                                 out_dir / f"importance_{sector}.png"))
    return saved


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    figs = generate_all()
    print(f"Saved {len(figs)} figures:")
    for p in figs:
        print(f"  {p}")
