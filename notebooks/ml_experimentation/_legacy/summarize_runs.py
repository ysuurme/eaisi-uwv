"""
RETIRED (archived 2026-06) — superseded by ``python main.py --report`` (the
``sector_performance`` read-model + leaderboard) and ``python main.py --compare``.
Kept for reference only; it ranks on ``r2_pre2023`` / ``r2_post2023`` metrics that
ml_5 no longer logs, so it will not run as-is against the current eval DB.

Per-sector overnight summary: best ML model vs baseline and structural floor.

Pulls every MLflow run from the eval database, picks the best-performing
ML model for each sector (excluding `baseline` and `structural_linear`
diagnostic sweeps), and joins on those two floors so the per-sector
"is this any good?" answer is one column away.

Output columns
--------------
sector              SBI sector label (the OHE column suffix)
best_model          Catalog model name (Ridge_Reduced, ElasticNet_Reduced, ...)
best_preset         Preset whose feature set produced the win
r2, mae, rmse       Aggregate metrics across all walk-forward origins
r2_pre2023          R² over origins whose training cutoff ≤ Q4 2022
r2_post2023         R² over origins whose training cutoff  > Q4 2022
baseline_metric     Same metric for SectorQuarterRollingMean on that sector
structural_metric   Same metric for the structural-only Ridge on that sector
uplift_vs_baseline  best_metric − baseline_metric (sign-aware for MAE)
uplift_vs_struct    best_metric − structural_metric (sign-aware for MAE)

Identification logic
--------------------
We rely on two tags set by the pipeline:
- tags.preset == "baseline"        → the SectorQuarterRollingMean sweep
- tags.preset == "structural_only" → the structural_linear sweep
Everything else is a real ML run and competes for "best".

Usage
-----
    uv run python summarize_runs.py
    uv run python summarize_runs.py --metric mean_absolute_error --minimize
    uv run python summarize_runs.py --passed-only --out summary.csv
"""
from __future__ import annotations

import argparse
from typing import Any

import mlflow
import pandas as pd

from src.config import DIR_DB_EVAL


# ---------------------------------------------------------------------------
# Run-name parsing
# ---------------------------------------------------------------------------

def _model_from_run_name(row: pd.Series) -> str:
    """Recover catalog model label from `{config.name}_{sector_label}`."""
    rn = str(row.get("tags.mlflow.runName") or "")
    sector = str(row.get("tags.sector") or "")
    if sector and rn.endswith(f"_{sector}"):
        return rn[: -(len(sector) + 1)]
    return rn


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def summarize_overnight(
    metric: str = "r2_score",
    minimize: bool = False,
    passed_only: bool = False,
) -> pd.DataFrame:
    """Build the per-sector best-model summary.

    Args:
        metric: MLflow metric name used to pick "best" and compute uplift.
            Common choices: "r2_score" (maximize) or "mean_absolute_error"
            (minimize — pair with minimize=True).
        minimize: True when lower values are better (e.g., MAE, RMSE).
        passed_only: When True, restrict ML candidates to runs with
            tags.passed_gate == "true".  Baseline + structural floors are
            never filtered — we still want them for comparison.

    Returns:
        DataFrame with one row per sector.  Empty if no runs were found.
    """
    # Match the tracking URI exactly to what the orchestrator uses.
    mlflow.set_tracking_uri(f"sqlite:///{DIR_DB_EVAL.as_posix()}")
    client = mlflow.tracking.MlflowClient()

    experiments = client.search_experiments()
    if not experiments:
        return pd.DataFrame()

    all_runs: list[pd.DataFrame] = []
    for exp in experiments:
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=10_000,
            output_format="pandas",
        )
        if len(runs):
            runs["experiment_name"] = exp.name
            all_runs.append(runs)

    if not all_runs:
        return pd.DataFrame()

    df = pd.concat(all_runs, ignore_index=True)
    metric_col = f"metrics.{metric}"

    # Sanity: drop runs without sector tag or the selected metric.
    required = ["tags.sector", metric_col, "tags.mlflow.runName"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"MLflow runs are missing required columns: {missing}. "
            f"Did the orchestrator run successfully?"
        )
    df = df.dropna(subset=required).copy()

    # Identify the two diagnostic floors by their preset tag.
    if "tags.preset" not in df.columns:
        df["tags.preset"] = "unknown"
    is_baseline = df["tags.preset"] == "baseline"
    is_structural = df["tags.preset"] == "structural_only"

    # Aggregate floor metric per sector (best across any run for that floor).
    agg_fn = "min" if minimize else "max"
    baseline_metric = (
        df[is_baseline]
        .groupby("tags.sector")[metric_col]
        .agg(agg_fn)
        .rename("baseline_metric")
    )
    structural_metric = (
        df[is_structural]
        .groupby("tags.sector")[metric_col]
        .agg(agg_fn)
        .rename("structural_metric")
    )

    # ML candidates: everything that isn't a floor.
    ml = df[~is_baseline & ~is_structural].copy()
    if passed_only and "tags.passed_gate" in ml.columns:
        ml = ml[ml["tags.passed_gate"] == "true"]
    if ml.empty:
        # Still return floors in case the user wants to see them solo
        return pd.DataFrame()

    ml["_model"] = ml.apply(_model_from_run_name, axis=1)

    # Pick best row per sector (lowest metric if minimize, else highest).
    pick_fn = "idxmin" if minimize else "idxmax"
    best_idx = ml.groupby("tags.sector")[metric_col].agg(pick_fn)
    best = ml.loc[best_idx].copy()

    # Compose the summary in a stable column order.
    col_map = {
        "tags.sector": "sector",
        "_model": "best_model",
        "tags.preset": "best_preset",
        "metrics.r2_score": "r2",
        "metrics.mean_absolute_error": "mae",
        "metrics.root_mean_squared_error": "rmse",
    }
    col_map = {k: v for k, v in col_map.items() if k in best.columns}
    summary = best[list(col_map)].rename(columns=col_map).reset_index(drop=True)

    # Optional regime-split columns (only present after P1.15 landed).
    for src, dst in (
        ("metrics.r2_pre2023", "r2_pre2023"),
        ("metrics.r2_post2023", "r2_post2023"),
    ):
        if src in best.columns:
            summary[dst] = best[src].values

    # The metric we actually used to rank (could be r2 or mae or rmse).
    summary["best_metric"] = best[metric_col].values

    # Join floors and compute sign-aware uplift.
    summary = (
        summary.merge(baseline_metric, left_on="sector", right_index=True, how="left")
               .merge(structural_metric, left_on="sector", right_index=True, how="left")
    )
    sign = -1.0 if minimize else 1.0
    summary["uplift_vs_baseline"]   = sign * (summary["best_metric"] - summary["baseline_metric"])
    summary["uplift_vs_structural"] = sign * (summary["best_metric"] - summary["structural_metric"])

    return summary.sort_values("sector").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print(summary: pd.DataFrame, metric: str) -> None:
    if summary.empty:
        print("No ML runs found in MLflow.")
        return

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 220,
        "display.float_format", lambda v: f"{v: 7.3f}" if pd.notnull(v) else "    nan",
    ):
        print(summary.to_string(index=False))

    print("\n=== Aggregate ===")
    print(f"Sectors with ML results:       {len(summary)}")
    if "r2" in summary.columns:
        print(f"Mean R² (best per sector):     {summary['r2'].mean(): .3f}")
        print(f"Median R² (best per sector):   {summary['r2'].median(): .3f}")
    if "mae" in summary.columns:
        print(f"Mean MAE (best per sector):    {summary['mae'].mean(): .3f}")
    if "uplift_vs_baseline" in summary.columns:
        beats = (summary["uplift_vs_baseline"] > 0).sum()
        print(f"Sectors beating baseline:      {beats}/{len(summary)}")
        print(f"Mean uplift_vs_baseline:       {summary['uplift_vs_baseline'].mean(): .3f}")
    if "uplift_vs_structural" in summary.columns:
        beats = (summary["uplift_vs_structural"] > 0).sum()
        print(f"Sectors beating structural:    {beats}/{len(summary)}")
        print(f"Mean uplift_vs_structural:     {summary['uplift_vs_structural'].mean(): .3f}")
    if "r2_pre2023" in summary.columns and "r2_post2023" in summary.columns:
        pre  = summary["r2_pre2023"].dropna()
        post = summary["r2_post2023"].dropna()
        if len(pre) and len(post):
            print(f"Mean r2_pre2023:               {pre.mean(): .3f}")
            print(f"Mean r2_post2023:              {post.mean(): .3f}")
            print(f"Mean (post − pre):             {(post.mean() - pre.mean()): .3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--metric", default="r2_score",
                   help="MLflow metric used to rank best per sector. "
                        "Default: r2_score.")
    p.add_argument("--minimize", action="store_true",
                   help="Set when the chosen metric is loss-like (e.g. MAE, RMSE).")
    p.add_argument("--passed-only", action="store_true",
                   help="Restrict to runs that passed the quality gate.")
    p.add_argument("--out", default=None,
                   help="Optional CSV path to write the table to.")
    args = p.parse_args()

    summary = summarize_overnight(
        metric=args.metric,
        minimize=args.minimize,
        passed_only=args.passed_only,
    )
    _print(summary, args.metric)

    if args.out and not summary.empty:
        summary.to_csv(args.out, index=False)
        print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
