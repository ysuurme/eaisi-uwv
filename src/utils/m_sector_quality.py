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

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_DEFAULT_PREFIX = "master_SickLeave_4Q_"
_PROD_ALIAS = "prod"
_BASELINE_FAMILY = "SectorQuarterRollingMean"
_COLUMNS = [
    "sector_code", "model_family", "model_type", "feature_groups",
    "champion_mape", "baseline_mape", "improvement", "r2", "tier",
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
            "model_type": tags.get("model_type", ""),
            "feature_groups": tags.get("feature_groups", ""),
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


# ───────────────────────────────────────────────────────────────────────────
# SBI hierarchy enrichment + materialised read-model (viz DB)
# ───────────────────────────────────────────────────────────────────────────

_NATIONAL_TOTAL = "T001081"


def load_sbi_hierarchy(dimension_json_path) -> Dict[str, Dict[str, str]]:
    """Build ``{sector_code: {sbi_title, sbi_level}}`` from the CBS SBI dimension.

    Leverages the generic CBS dimension parser in ``m_sbi_classifier`` (utils),
    keeping the UWV-specific assembly here.
    """
    from src.utils.m_sbi_classifier import _f_load_cbs_dimension_lookup
    df = _f_load_cbs_dimension_lookup(Path(dimension_json_path))
    return {
        str(row["Key"]).strip(): {
            "sbi_title": str(row.get("Title", "")),
            "sbi_level": str(row.get("sbi_level", "unknown")),
        }
        for _, row in df.iterrows()
    }


def enrich_with_hierarchy(quality_df, hierarchy: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Add ``sbi_title`` / ``sbi_level`` from the hierarchy. Unknown sectors are
    labelled (``sbi_level='unknown'``), never dropped, so the tree never breaks."""
    if quality_df is None or quality_df.empty:
        return quality_df
    df = quality_df.copy()
    codes = df["sector_code"].astype(str)
    df["sbi_title"] = codes.map(lambda c: hierarchy.get(c, {}).get("sbi_title", ""))
    df["sbi_level"] = codes.map(lambda c: hierarchy.get(c, {}).get("sbi_level", "unknown"))
    return df


def _json_safe(value):
    """Coerce pandas/numpy NaN to None so the structure is JSON-serializable."""
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


#: Title-prefix parsers for CBS SBI dimension titles.  The Title encodes the
#: hierarchy level (stable across CBS tables — see ``m_sbi_classifier``):
#:   "A-U Alle…"  → totaal (root)         "B-F Nijverheid…" → sector (letter range)
#:   "G Handel"   → section (one letter)  "45 Autohandel…"  → subdivision (division no.)
_RE_SECTOR_RANGE = re.compile(r"^\s*([A-U])-([A-U])\b")   # "B-F …"
_RE_SECTION_LETTER = re.compile(r"^\s*([A-U])\b")          # "G …" (single letter)
_RE_LEADING_DIVISION = re.compile(r"^\s*(\d{1,2})")        # "10-12 …", "45 …"


def _sector_range_letters(title: str) -> List[str]:
    """Letters a 'B-F …' sector title spans → ``['B','C','D','E','F']`` (else [])."""
    match = _RE_SECTOR_RANGE.match(title or "")
    if not match:
        return []
    start, end = ord(match.group(1)), ord(match.group(2))
    return [chr(c) for c in range(start, end + 1)] if start <= end else []


def _section_letter(title: str) -> Optional[str]:
    """Section letter of a 'G Handel' title → ``'G'`` (None if not a single-letter title)."""
    if _RE_SECTOR_RANGE.match(title or ""):
        return None  # a range like 'B-F' is a sector, not a section
    match = _RE_SECTION_LETTER.match(title or "")
    return match.group(1) if match else None


def _division_section_letter(title: str) -> Optional[str]:
    """Section letter for a 'NN …' subdivision title via the CBS division map."""
    match = _RE_LEADING_DIVISION.match(title or "")
    if not match:
        return None
    from src.utils.m_sbi_classifier import _DIVISION_TO_SECTION
    return _DIVISION_TO_SECTION.get(int(match.group(1)))


def to_tree(enriched_df) -> dict:
    """JSON-serializable, genuinely nested SBI tree of the sector champions.

    Builds true CBS parent/child edges — national total → sector (letter range
    like ``B-F``) → section (single letter ``A``-``U``) → subdivision (numeric
    division) — inferred from each row's ``sbi_title`` / ``sbi_level`` (the same
    signals ``m_sbi_classifier`` derives).  Every node keeps its leading
    attributes (champion · model_type · sector · performance vs baseline) and
    gains a ``children`` list.

    A node attaches to the **nearest present ancestor**: a section to the sector
    whose letter range covers it (else the root); a subdivision to its section
    (else the sector covering its division, else the root).  Business-size (WP)
    and unclassifiable nodes hang off the root.  Nothing is ever dropped.
    """
    if enriched_df is None or enriched_df.empty:
        return {}

    def node(record: dict) -> dict:
        clean = {k: _json_safe(v) for k, v in record.items()}
        clean["children"] = []
        return clean

    records = enriched_df.to_dict("records")
    nodes = [node(r) for r in records]

    # Root = the national total (totaal), or a synthetic root if absent.
    root = next(
        (n for n in nodes
         if str(n.get("sector_code")) == _NATIONAL_TOTAL or n.get("sbi_level") == "totaal"),
        None,
    )
    synthetic_root = root is None
    if synthetic_root:
        root = {"sector_code": "ALL", "sbi_level": "root", "children": []}

    # Index the PRESENT sector/section nodes so descendants can find a parent.
    sector_by_letter: Dict[str, dict] = {}
    section_by_letter: Dict[str, dict] = {}
    for n in nodes:
        title = str(n.get("sbi_title", "") or "")
        if n.get("sbi_level") == "sector":
            for letter in _sector_range_letters(title):
                sector_by_letter.setdefault(letter, n)
        elif n.get("sbi_level") == "section":
            letter = _section_letter(title)
            if letter:
                section_by_letter.setdefault(letter, n)

    def parent_for(n: dict) -> dict:
        if n is root:
            return root
        level = n.get("sbi_level")
        title = str(n.get("sbi_title", "") or "")
        if level == "section":
            return sector_by_letter.get(_section_letter(title) or "", root)
        if level == "subdivision":
            letter = _division_section_letter(title) or ""
            return section_by_letter.get(letter) or sector_by_letter.get(letter, root)
        # totaal/sector/size/other/unknown → root (sector ranges live directly under root)
        return root

    for n in nodes:
        if n is root:
            continue
        parent_for(n)["children"].append(n)

    return root


def write_sector_performance(enriched_df, eval_db_path) -> int:
    """Materialise the enriched per-sector table into the ``sector_performance``
    read-model (replace semantics). Returns the number of rows written."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from src.ml_engineering.model_configs import Base, SectorPerformance

    engine = create_engine(
        f"sqlite:///{Path(eval_db_path).as_posix()}", connect_args={"timeout": 30}
    )
    Base.metadata.create_all(engine)
    cols = {c.name for c in SectorPerformance.__table__.columns}
    records = [] if (enriched_df is None or enriched_df.empty) else enriched_df.to_dict("records")
    try:
        with sessionmaker(bind=engine)() as session:
            session.query(SectorPerformance).delete()
            for record in records:
                payload = {k: _json_safe(v) for k, v in record.items() if k in cols}
                session.add(SectorPerformance(**payload))
            session.commit()
    finally:
        engine.dispose()  # release the SQLite file handle (Windows-safe)
    return len(records)


def load_sector_performance(eval_db_path: Optional[Path] = None) -> pd.DataFrame:
    """Read the materialised ``sector_performance`` read-model (for charts/apps)."""
    from sqlalchemy import create_engine
    if eval_db_path is None:
        from src.config import DIR_DB_EVAL
        eval_db_path = DIR_DB_EVAL
    engine = create_engine(f"sqlite:///{Path(eval_db_path).as_posix()}")
    try:
        return pd.read_sql("SELECT * FROM sector_performance", engine)
    finally:
        engine.dispose()


def refresh_sector_performance(
    eval_db_path: Optional[Path] = None,
    dimension_json_path: Optional[Path] = None,
    registered_model_prefix: str = _DEFAULT_PREFIX,
) -> int:
    """Rebuild the read-model FROM the single source of truth — MLflow champions
    (self-describing) + baseline MAPE + the CBS SBI hierarchy — and write it to
    ``sector_performance``. This refresh is the only write path."""
    import mlflow
    from mlflow.tracking import MlflowClient

    if eval_db_path is None:
        from src.config import DIR_DB_EVAL
        eval_db_path = DIR_DB_EVAL
    if dimension_json_path is None:
        from src.config import DIR_DATA_RAW
        dimension_json_path = Path(DIR_DATA_RAW) / "80072ned" / "BedrijfskenmerkenSBI2008.json"

    mlflow.set_tracking_uri(f"sqlite:///{Path(eval_db_path).as_posix()}?timeout=30")
    quality = build_sector_quality_table(
        MlflowClient(), baseline_mape_by_sector(eval_db_path),
        registered_model_prefix=registered_model_prefix,
    )
    enriched = enrich_with_hierarchy(quality, load_sbi_hierarchy(dimension_json_path))
    return write_sector_performance(enriched, eval_db_path)


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


# ───────────────────────────────────────────────────────────────────────────
# Experiment-matrix aggregations + narrative (champion-based reporting, Phase 5)
#
# READ-ONLY over the single sources of truth — the MLflow runs in the
# ``master_SickLeave_4Q`` experiment and the eval-DB read tables
# (``model_predictions``, ``model_forecasts``).  Nothing here stores a metric;
# every output is a derived, ``--report``-regenerable view.  The matching
# figure builders live in ``m_model_viz`` (charts), keeping data aggregation
# here next to the other MLflow-derived read-models.
# ───────────────────────────────────────────────────────────────────────────

_EXPERIMENT_NAME = "master_SickLeave_4Q"
#: ml_5 logs the honest outer-fold MAPE under sklearn's canonical metric name.
_MAPE_METRIC = "metrics.mean_absolute_percentage_error"


def _resolve_eval_db(eval_db_path: Optional[Path]) -> Path:
    if eval_db_path is not None:
        return Path(eval_db_path)
    from src.config import DIR_DB_EVAL
    return DIR_DB_EVAL


def _engine(eval_db_path):
    from sqlalchemy import create_engine
    return create_engine(f"sqlite:///{Path(eval_db_path).as_posix()}")


def _feature_group_label(raw) -> str:
    """Normalise ``params.feature_groups`` to a short, stable column label.

    Logged as a JSON list (``["all_survivors"]``) or the sentinel ``"discovery"``
    (all columns).  Returns a ``+``-joined group label or ``"discovery"``.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "discovery"
    text = str(raw).strip()
    if text in ("", "discovery", "nan", "None"):
        return "discovery"
    try:
        groups = json.loads(text)
        if isinstance(groups, list) and groups:
            return "+".join(str(g) for g in groups)
    except (TypeError, ValueError):
        pass
    return text


def load_runs(eval_db_path: Optional[Path] = None) -> pd.DataFrame:
    """Tidy every experiment run: ``model_family, feature_group, sector, mape``.

    The outer-fold MAPE is the run metric ``mean_absolute_percentage_error``;
    runs without it (none finished) are dropped.
    """
    import mlflow

    eval_db_path = _resolve_eval_db(eval_db_path)
    mlflow.set_tracking_uri(f"sqlite:///{eval_db_path.as_posix()}?timeout=30")
    exp = mlflow.get_experiment_by_name(_EXPERIMENT_NAME)
    cols = ["model_family", "feature_group", "sector", "mape"]
    if exp is None:
        return pd.DataFrame(columns=cols)

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=5000)
    if runs is None or runs.empty or _MAPE_METRIC not in runs.columns:
        return pd.DataFrame(columns=cols)

    family = runs.get("tags.model_family", runs.get("params.model_name"))
    sector = runs.get("tags.sector")
    fgroups = runs.get("params.feature_groups")
    out = pd.DataFrame({
        "model_family": family,
        "feature_group": fgroups.map(_feature_group_label) if fgroups is not None else "discovery",
        "sector": sector,
        "mape": pd.to_numeric(runs[_MAPE_METRIC], errors="coerce"),
    })
    return out.dropna(subset=["mape", "model_family"]).reset_index(drop=True)


def build_experiment_matrix(runs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Median outer MAPE per (family, feature_group) + per-cell sector-win counts.

    Returns ``(mape_matrix, win_matrix)`` — both pivoted with model families on
    the index and feature groups on the columns.  A "win" is awarded to the
    (family, feature_group) cell with the lowest MAPE for each sector.
    """
    if runs_df is None or runs_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    mape_matrix = runs_df.pivot_table(
        index="model_family", columns="feature_group", values="mape", aggfunc="median",
    )

    wins: Dict[Tuple[str, str], int] = {}
    for _, grp in runs_df.groupby("sector"):
        best = grp.loc[grp["mape"].idxmin()]
        key = (best["model_family"], best["feature_group"])
        wins[key] = wins.get(key, 0) + 1

    win_matrix = pd.DataFrame(0, index=mape_matrix.index, columns=mape_matrix.columns)
    for (fam, fg), n in wins.items():
        if fam in win_matrix.index and fg in win_matrix.columns:
            win_matrix.loc[fam, fg] = n
    return mape_matrix, win_matrix


def per_horizon_mape(eval_db_path: Optional[Path] = None) -> pd.DataFrame:
    """MAPE by forecast horizon (h=1..4) over the honest OUTER folds.

    Reads ``model_predictions``, keeps ``fold_set == 'outer'`` when present, and
    computes ``mean(|y_true - y_pred| / |y_true|)`` per horizon — derived on read
    from the stored per-row predictions (the aggregate is not persisted).
    """
    eval_db_path = _resolve_eval_db(eval_db_path)
    engine = _engine(eval_db_path)
    try:
        preds = pd.read_sql("SELECT * FROM model_predictions", engine)
    except Exception:
        return pd.DataFrame(columns=["horizon", "mape", "n"])
    finally:
        engine.dispose()

    if preds.empty:
        return pd.DataFrame(columns=["horizon", "mape", "n"])
    if "fold_set" in preds.columns:
        outer = preds[preds["fold_set"] == "outer"]
        preds = outer if not outer.empty else preds

    preds = preds[preds["y_true"].abs() > 0].copy()
    preds["ape"] = (preds["y_true"] - preds["y_pred"]).abs() / preds["y_true"].abs()
    rows = [
        {"horizon": int(h), "mape": float(g["ape"].mean()), "n": int(len(g))}
        for h, g in preds.groupby("horizon")
    ]
    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)


def load_forecasts(eval_db_path: Optional[Path] = None) -> pd.DataFrame:
    """Read the persisted forward forecasts (``model_forecasts``)."""
    eval_db_path = _resolve_eval_db(eval_db_path)
    engine = _engine(eval_db_path)
    try:
        df = pd.read_sql("SELECT * FROM model_forecasts", engine)
    except Exception:
        return pd.DataFrame()
    finally:
        engine.dispose()
    if not df.empty:
        df["target_date"] = pd.to_datetime(df["target_date"])
    return df


def _find_importance_vector(estimator, _depth: int = 0):
    """Recursively search a fitted estimator for a 1-D coef_/feature_importances_.

    Walks nested sktime/sklearn wrappers (reducer → pipeline → regressor) up to a
    bounded depth.  Returns the first flat vector found, or ``None`` (stat models
    and anything without flat coefficients fall through cleanly — never raises).
    """
    if estimator is None or _depth > 6:
        return None
    import numpy as np

    for attr in ("coef_", "feature_importances_"):
        vec = getattr(estimator, attr, None)
        if vec is not None:
            arr = np.asarray(vec).ravel()
            if arr.ndim == 1 and arr.size:
                return arr

    for attr in ("estimator_", "estimator", "regressor_", "regressor",
                 "forecaster_", "best_forecaster_"):
        found = _find_importance_vector(getattr(estimator, attr, None), _depth + 1)
        if found is not None:
            return found

    steps = getattr(estimator, "steps", None)  # sklearn Pipeline
    if steps:
        for _, step in steps:
            found = _find_importance_vector(step, _depth + 1)
            if found is not None:
                return found

    estimators = getattr(estimator, "estimators_", None)  # direct reducer (per-horizon)
    if estimators is not None and len(estimators):
        return _find_importance_vector(estimators[0], _depth + 1)
    return None


def champion_importances(
    gold_table: str = "master_data_ml_preprocessed",
    eval_db_path: Optional[Path] = None,
    top_n: int = 15,
) -> Dict[str, pd.DataFrame]:
    """Best-effort ``{sector: importance_frame}`` for reducer champions.

    Rebuilds each ``@prod`` champion from its lineage and refits on history
    (reusing the Step-7 machinery), then introspects the fitted estimator for a
    flat coefficient / importance vector.  Stat champions (no flat vector) and
    any rebuild failure are skipped silently — the step never crashes a report.
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    from src.ml_engineering import ml_7_model_inference as ml_7

    eval_db_path = _resolve_eval_db(eval_db_path)
    mlflow.set_tracking_uri(f"sqlite:///{eval_db_path.as_posix()}?timeout=30")
    client = MlflowClient()
    prefix = ml_7._experiment_prefix(gold_table)

    out: Dict[str, pd.DataFrame] = {}
    for rm in client.search_registered_models():
        if not rm.name.startswith(prefix):
            continue
        sector = rm.name[len(prefix):]
        try:
            lineage = ml_7._read_champion_lineage(client, rm.name, sector)
            estimator, config = ml_7._rebuild_estimator(lineage.experiment_key, lineage.best_params)
            sbi = ml_7._sector_to_sbi_filter(sector)
            x_hist, _y_hist = ml_7._load_sector_history(gold_table, sbi, config.feature_groups)
            x_num = x_hist.select_dtypes(include="number")
            x_fit = x_num if (config.feature_groups is not None and not x_num.empty) else None
            estimator.fit(y=_y_hist, X=x_fit, fh=[1, 2, 3, 4])
            vec = _find_importance_vector(estimator)
        except Exception:
            vec = None
        if vec is None:
            continue
        names = list(x_num.columns) if x_fit is not None else [f"lag_{i+1}" for i in range(len(vec))]
        if len(names) != len(vec):
            names = [f"f{i}" for i in range(len(vec))]
        frame = pd.DataFrame({"feature": names, "weight": vec})
        frame["abs"] = frame["weight"].abs()
        out[sector] = frame.sort_values("abs", ascending=False).head(top_n).drop(columns="abs")
    return out


def _fmt_pct(value) -> str:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "—"
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def build_narrative_markdown(
    eval_db_path: Optional[Path] = None,
    gold_table: str = "master_data_ml_preprocessed",
    title: str = "Sick-leave forecasting — model report (week of 2026-06-19)",
) -> str:
    """Render the human-readable model report as a Markdown string.

    Pulls champions + tiers from the ``sector_performance`` read-model, the
    family × feature-group matrix from MLflow runs, and the forward forecast from
    ``model_forecasts`` — all single sources of truth.  Safe on an empty store.
    """
    eval_db_path = _resolve_eval_db(eval_db_path)
    try:
        perf = load_sector_performance(eval_db_path)
    except Exception:
        perf = pd.DataFrame()  # read-model not refreshed yet → headline degrades
    runs = load_runs(eval_db_path)
    mape_matrix, _win_matrix = build_experiment_matrix(runs)
    forecasts = load_forecasts(eval_db_path)
    horizon = per_horizon_mape(eval_db_path)

    lines: List[str] = [f"# {title}", ""]

    if perf is not None and not perf.empty and "tier" in perf.columns:
        counts = perf["tier"].value_counts().to_dict()
        good, medium, poor = counts.get("Good", 0), counts.get("Medium", 0), counts.get("Poor", 0)
        lines += [
            "## Headline",
            f"- **{good} Good** · **{medium} Medium** · **{poor} Poor** "
            f"({len(perf)} sector champions).",
            "- *Good* = champion cuts MAPE by ≥10% vs the `SectorQuarterRollingMean` "
            "baseline; *Poor* = no demonstrable lift over the naive seasonal baseline.",
            "",
            "## Per-sector champions",
            "",
            "| Sector | SBI title | Champion | Model type | MAPE | Baseline | Improvement | Tier |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, r in perf.iterrows():
            lines.append(
                f"| {r.get('sector_code','')} | {str(r.get('sbi_title','') or '')[:40]} "
                f"| {r.get('model_family','')} | {r.get('model_type','') or ''} "
                f"| {_fmt_pct(r.get('champion_mape'))} | {_fmt_pct(r.get('baseline_mape'))} "
                f"| {_fmt_pct(r.get('improvement'))} | {r.get('tier','')} |"
            )
        lines.append("")
    else:
        lines += ["## Headline", "- No champions registered yet — run the training "
                  "sweep before reporting.", ""]

    lines += ["## Model-family × feature-group matrix (median outer MAPE)", ""]
    if mape_matrix.empty:
        lines += ["- No completed runs to compare.", ""]
    else:
        header = "| family ＼ group | " + " | ".join(mape_matrix.columns) + " |"
        sep = "|---|" + "|".join(["---"] * len(mape_matrix.columns)) + "|"
        lines += [header, sep]
        for fam, row in mape_matrix.iterrows():
            cells = " | ".join(_fmt_pct(v) for v in row)
            lines.append(f"| {fam} | {cells} |")
        best_cell = mape_matrix.stack().idxmin()
        lines += [
            "",
            f"- Lowest median MAPE: **{best_cell[0]}** on **{best_cell[1]}** "
            f"({_fmt_pct(mape_matrix.stack().min())}).",
            "- ★ sector-win counts are annotated on the heatmap "
            "(`reports/figures/experiment_matrix.png`).",
            "",
        ]

    lines += ["## Forward 4Q forecast (from @prod champions)", ""]
    if forecasts is None or forecasts.empty:
        lines += ["- No forecasts persisted — run `python main.py --forecast`.", ""]
    else:
        lines += ["| Sector | Champion | Target quarter | Horizon | Forecast |",
                  "|---|---|---|---|---|"]
        for _, r in forecasts.sort_values(["sector_code", "horizon"]).iterrows():
            lines.append(
                f"| {r['sector_code']} | {r.get('model_family','')} "
                f"| {pd.to_datetime(r['target_date']).date()} | {int(r['horizon'])} "
                f"| {float(r['y_pred']):.2f}% |"
            )
        lines.append("")

    if horizon is not None and not horizon.empty:
        decay = ", ".join(f"h{int(r.horizon)}={_fmt_pct(r.mape)}" for r in horizon.itertuples())
        lines += ["## Forecast decay (outer-fold MAPE by horizon)", f"- {decay}", ""]

    lines += [
        "## Caveats & notes",
        "- **Carry-forward X**: forward forecasts use production-honest future X — "
        "deterministic structure extended from the dates, stochastic exogenous "
        "columns carried forward from the last observed quarter (the same no-leak "
        "rule the champion was selected under).",
        "- **Yearly features excluded**: candidate features are restricted to "
        "quarterly/monthly CBS origins; yearly tables (disaggregation artifacts) "
        "are excluded from this first validation.",
        "- MLflow is the single source of truth; every table here is a refresh-only "
        "projection of the registry champions.",
        "",
        "## Next steps",
        "- Run the overnight `--all-sectors` sweeps (baseline → stat → deseason "
        "challengers) so every sector has a champion, then re-run `--report`.",
        "",
    ]
    return "\n".join(lines)


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
