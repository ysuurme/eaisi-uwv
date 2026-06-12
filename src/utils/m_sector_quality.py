"""
Sector forecast-quality bucketing — Good / Medium / Poor, benchmarked against
the BASELINE model.

The baseline is ``SectorQuarterRollingMean`` — the rolling 3-year same-quarter
mean from ``01_nb_baseline_model.ipynb`` ("the model we need to outperform"),
replicated as an estimator and evaluated through the SAME walk-forward pipeline
as every ML model, so its MAPE is directly comparable.

A sector's tier reflects how much its **champion** (the registry ``@prod`` model)
improves on the baseline's MAPE for that sector:

    Good   — champion beats baseline by ≥ 10% relative MAPE reduction
    Medium — champion beats baseline, but by < 10%
    Poor   — champion does NOT beat the baseline (no demonstrable ML lift),
             or the metrics are missing / non-finite

"Sound results" are the Good-tier sectors: those where the model provably adds
value over the naive seasonal baseline.

Sources of truth:
* champion MAPE  → registered model ``@prod`` version tags (ADR-002)
* baseline MAPE  → ``model_evaluations`` rows for ``SectorQuarterRollingMean_*``
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_DEFAULT_PREFIX = "master_SickLeave_4Q_"
_PROD_ALIAS = "prod"
_BASELINE_FAMILY = "SectorQuarterRollingMean"
_COLUMNS = [
    "sector_code", "model_family", "champion_mape", "baseline_mape",
    "improvement", "r2", "tier",
]

#: Documented default: champion must cut MAPE by ≥ 10% vs baseline to be "Good".
GOOD_IMPROVEMENT = 0.10

#: Float tolerance so the documented inclusive thresholds aren't missed by
#: binary rounding (e.g. 0.054/0.060 → 0.0999…998 instead of exactly 0.10).
_EPS = 1e-9


def _improvement(champion_mape: Optional[float], baseline_mape: Optional[float]) -> float:
    """Relative MAPE reduction of champion vs baseline (NaN if not computable)."""
    if champion_mape is None or baseline_mape is None:
        return float("nan")
    if not math.isfinite(champion_mape) or not math.isfinite(baseline_mape) or baseline_mape <= 0:
        return float("nan")
    return 1.0 - (champion_mape / baseline_mape)


def assign_tier(
    champion_mape: Optional[float],
    baseline_mape: Optional[float],
    *,
    good_improvement: float = GOOD_IMPROVEMENT,
) -> str:
    """Classify a sector by its champion's improvement over the baseline MAPE."""
    improvement = _improvement(champion_mape, baseline_mape)
    if not math.isfinite(improvement):
        return "Poor"
    if improvement >= good_improvement - _EPS:
        return "Good"
    if improvement > _EPS:
        return "Medium"
    return "Poor"


def baseline_mape_by_sector(
    eval_db_path: Optional[Path] = None,
    baseline_family: str = _BASELINE_FAMILY,
) -> Dict[str, float]:
    """Read each sector's baseline MAPE from the ``model_evaluations`` store.

    Baseline runs are persisted with ``model_name == f"{baseline_family}_{sector}"``.
    Returns a ``{sector_code: mape}`` dict (most recent value per sector wins).
    """
    from sqlalchemy import create_engine

    if eval_db_path is None:
        from src.config import DIR_DB_EVAL
        eval_db_path = DIR_DB_EVAL

    engine = create_engine(f"sqlite:///{Path(eval_db_path).as_posix()}")
    df = pd.read_sql(
        "SELECT model_name, mape, timestamp FROM model_evaluations", engine
    )
    prefix = f"{baseline_family}_"
    out: Dict[str, float] = {}
    df = df[df["model_name"].astype(str).str.startswith(prefix)]
    df = df.sort_values("timestamp")  # later rows overwrite earlier per sector
    for _, row in df.iterrows():
        if row["mape"] is None:
            continue
        out[str(row["model_name"])[len(prefix):]] = float(row["mape"])
    return out


def build_sector_quality_table(
    client,
    baseline_mapes: Dict[str, float],
    registered_model_prefix: str = _DEFAULT_PREFIX,
    *,
    good_improvement: float = GOOD_IMPROVEMENT,
) -> pd.DataFrame:
    """Build the per-sector quality table, benchmarking champions vs baseline.

    Parameters
    ----------
    client : mlflow.tracking.MlflowClient
        Enumerates registered models and resolves each sector's ``@prod`` champion.
    baseline_mapes : dict[str, float]
        ``{sector_code: baseline_mape}`` (e.g. from ``baseline_mape_by_sector``).
    registered_model_prefix : str
        Only registered models with this prefix are sector champions; the sector
        code is the remainder of the name.

    Returns
    -------
    pandas.DataFrame
        Columns ``sector_code, model_family, champion_mape, baseline_mape,
        improvement, r2, tier``; one row per sector champion, sorted by tier
        (Good→Medium→Poor) then ascending champion MAPE.
    """
    rows = []
    for rm in client.search_registered_models():
        name = getattr(rm, "name", "")
        if not name.startswith(registered_model_prefix):
            continue
        try:
            mv = client.get_model_version_by_alias(name, _PROD_ALIAS)
        except Exception:
            continue  # no champion promoted for this sector yet
        tags = getattr(mv, "tags", None) or {}
        try:
            champ = float(tags["mape"])
            r2 = float(tags["r2"])
        except (KeyError, TypeError, ValueError):
            continue
        sector = name[len(registered_model_prefix):]
        base = baseline_mapes.get(sector)
        rows.append({
            "sector_code": sector,
            "model_family": tags.get("model_family", ""),
            "champion_mape": champ,
            "baseline_mape": base,
            "improvement": _improvement(champ, base),
            "r2": r2,
            "tier": assign_tier(champ, base, good_improvement=good_improvement),
        })

    df = pd.DataFrame(rows, columns=_COLUMNS)
    if df.empty:
        return df
    tier_order = pd.Categorical(df["tier"], categories=["Good", "Medium", "Poor"], ordered=True)
    return (
        df.assign(_t=tier_order)
        .sort_values(["_t", "champion_mape"])
        .drop(columns="_t")
        .reset_index(drop=True)
    )


def sound_result_sectors(df: pd.DataFrame) -> List[str]:
    """Return the Good-tier sector codes — the project's 'sound results'."""
    if df.empty:
        return []
    return df.loc[df["tier"] == "Good", "sector_code"].tolist()


def write_report(df: pd.DataFrame, path: Path) -> Path:
    """Persist the quality table as a CSV under ``path`` (parents created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def build_from_eval_db(eval_db_path: Optional[Path] = None) -> pd.DataFrame:
    """Convenience: point MLflow at the local eval DB and build the table."""
    import mlflow
    from mlflow.tracking import MlflowClient

    if eval_db_path is None:
        from src.config import DIR_DB_EVAL
        eval_db_path = DIR_DB_EVAL
    mlflow.set_tracking_uri(f"sqlite:///{Path(eval_db_path).as_posix()}?timeout=30")
    baselines = baseline_mape_by_sector(eval_db_path)
    return build_sector_quality_table(MlflowClient(), baselines)


if __name__ == "__main__":
    from src.config import PROJECT_ROOT

    table = build_from_eval_db()
    if table.empty:
        print("No sector champions found in the registry.")
    else:
        print(table.to_string(index=False))
        out = write_report(table, PROJECT_ROOT / "reports" / "sector_quality.csv")
        print(f"\nSound results (Good tier): {sound_result_sectors(table)}")
        print(f"Report written: {out}")
