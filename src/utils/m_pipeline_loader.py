"""
m_pipeline_loader.py — Pipeline predictions loader with HONEST nested CV.

Place at: src/utils/m_pipeline_loader.py

READ-ONLY: only reads from the eval DB.  Does NOT modify it.

Why this loader exists
----------------------
The pipeline writes per-row predictions for every (preset × model × sector ×
origin × horizon) combination into ``model_predictions``.  Each prediction is
now labelled with a ``fold_set`` column ("inner" or "outer") by the updated
ml_5_model_evaluation:

* INNER predictions  → used only to pick which variant represents Pipeline
                       for each sector (variant selection).
* OUTER predictions  → reported as Pipeline's canonical out-of-sample
                       performance.  Never inspected during selection.

This is **proper nested cross-validation**.  The variant choice is blind to
the data used to evaluate the chosen variant — eliminating the test-set
selection bias of the earlier (single-CV) loader.

Variant identifier
------------------
A "variant" within a sector is uniquely identified by ``(model_name, run_id)``:

* ``model_name`` = "{config_name}_{sector_label}" (e.g. "ridge_300003").
  Encodes both the model family AND the sector; same across presets.
* ``run_id``     = MLflow run identifier.  DIFFERENT for every (preset, model,
  sector) execution.

Combining (model_name, sector_code, run_id) means: "ridge with preset_A on
sector 300003" and "ridge with preset_B on sector 300003" are TWO DISTINCT
VARIANTS that compete for 300003's slot.  Different presets are NOT silently
averaged together — the loader picks the single best run_id per sector
based on inner-fold MAE.

Modes
-----
* ``per_sector_honest`` (DEFAULT, RECOMMENDED): nested CV.  Per sector, picks
  the (model_name, run_id) with lowest INNER-fold MAE.  Reports only OUTER-fold
  predictions of those winners.  This is what the cross-method comparison
  should use.

* ``global``: picks one MODEL FAMILY globally (extracted from model_name as
  everything before the last underscore — e.g. "ridge").  Within that family,
  per sector picks the best run_id by inner MAE.  Equalises model-family
  discipline with Chronos-Bolt; useful as a conservative alternative.

* ``per_sector_legacy``: old test-set-biased behaviour — picks variant on
  outer-fold MAE.  Kept for diagnostic regression comparison only.  DO NOT
  USE for the formal comparison.

Backward compatibility
----------------------
If the eval DB has no ``fold_set`` column (predictions persisted before the
inner/outer split was added), the loader prints a loud warning and falls
back to legacy behaviour, treating every row as outer.

If the eval DB has no ``run_id`` column (very old format), variants collapse
to ``model_name`` only — presets cannot be distinguished.  Warned.

Usage
-----
    from src.utils.m_pipeline_loader import load_pipeline_honest
    pipeline, pipeline_winners = load_pipeline_honest(eval_db_path)
"""

from pathlib import Path
from typing import Optional, Tuple
import warnings
import numpy as np
import pandas as pd


_VALID_MODES = ("per_sector_honest", "global", "per_sector_legacy")
_DEFAULT_TABLE_PRIORITY = ("model_predictions", "model_evaluation_records")
_BASELINE_PREFIX = "SectorQuarterRollingMean"

# Family display names for the cross-method comparison.  A "family" is the
# model_name with its trailing _<sector> stripped (i.e. the ModelConfiguration
# .name).  The reducer families collapse into a single "Pipeline" competitor;
# the stat families map to the operational-table keys used by m_evaluation.
_PIPELINE_FAMILIES = ("Ridge_Reduced", "RandomForest_Reduced", "RidgeDeseason_Reduced")
_FAMILY_DISPLAY = {
    "AutoETS_Stat":     "AutoETS_Stat",
    "STLETS_Stat":      "STL_ETS",
    "ChronosBolt_Stat": "Chronos_Bolt",
}


def _load_predictions_df(
    eval_db_path: "str | Path",
    table: Optional[str] = None,
    run_filter: Optional[dict] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Read + normalise the predictions table into the working frame shared by
    the per-sector-honest selectors and the per-family comparison loader.

    Returns ``(df, meta)`` where ``df`` carries ``sector_code, model_name,
    target_date, origin_date, horizon, y_true, y_pred, fold_set, [run_id], _ae``
    and ``meta`` is ``{table, has_fold_set, has_run_id, variant_keys}``.  Reads
    only; disposes the SQLAlchemy engine (Windows lock-safety).
    """
    from sqlalchemy import create_engine, inspect as sa_inspect

    eval_db_path = Path(eval_db_path)
    if not eval_db_path.exists():
        raise FileNotFoundError(f"Eval DB not found: {eval_db_path}")

    engine = create_engine(f"sqlite:///{eval_db_path.as_posix()}")
    try:
        insp = sa_inspect(engine)
        existing_tables = set(insp.get_table_names())

        # ── Pick predictions table ────────────────────────────────────────
        if table is None:
            table = next((t for t in _DEFAULT_TABLE_PRIORITY if t in existing_tables), None)
            if table is None:
                raise ValueError(
                    f"No predictions table found in {eval_db_path}. "
                    f"Expected one of {_DEFAULT_TABLE_PRIORITY}; available: {sorted(existing_tables)}"
                )

        # ── Inspect schema to choose the right column names ───────────────
        cols = {c["name"] for c in insp.get_columns(table)}

        if "origin_date" in cols:
            origin_col = "origin_date"
        elif "fold_origin_date" in cols:
            origin_col = "fold_origin_date"
        else:
            raise ValueError(
                f"Table {table} missing an origin-date column (origin_date / fold_origin_date)"
            )

        if "horizon" in cols:
            horizon_col = "horizon"
        elif "horizon_step" in cols:
            horizon_col = "horizon_step"
        else:
            raise ValueError(f"Table {table} missing a horizon column (horizon / horizon_step)")

        has_fold_set = "fold_set" in cols
        has_run_id   = "run_id" in cols
        required     = {"sector_code", "model_name", "y_true", "y_pred", "target_date"}
        missing      = required - cols
        if missing:
            raise ValueError(
                f"Table {table} missing required columns: {missing}.  Available: {sorted(cols)}"
            )

        # ── Build SQL select ──────────────────────────────────────────────
        select_cols = [
            "sector_code", "model_name", "target_date",
            f'{origin_col} AS origin_date',
            f'{horizon_col} AS horizon',
            "y_true", "y_pred",
        ]
        if has_fold_set:
            select_cols.append("fold_set")
        if has_run_id:
            select_cols.append("run_id")

        sql = f'SELECT {", ".join(select_cols)} FROM "{table}"'
        df  = pd.read_sql(sql, engine)
    finally:
        engine.dispose()

    if df.empty:
        raise ValueError(f"Table {table} is empty.")

    # ── Apply run_filter if provided ──────────────────────────────────────
    if run_filter:
        for k, v in run_filter.items():
            if k not in df.columns:
                raise ValueError(f"run_filter key '{k}' not in {table} columns")
            df = df[df[k] == v]
        if df.empty:
            raise ValueError(f"run_filter {run_filter} returned 0 rows")

    df = df.copy()
    df["sector_code"] = df["sector_code"].astype(str)
    df["model_name"]  = df["model_name"].astype(str)
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["origin_date"] = pd.to_datetime(df["origin_date"])

    # ── Handle absent fold_set (legacy data) ──────────────────────────────
    if not has_fold_set:
        warnings.warn(
            "Predictions table has no 'fold_set' column. Falling back to "
            "treating every row as 'outer' — variant selection cannot use "
            "inner folds and will be test-set biased. Re-run the sector sweep "
            "(main.py <table> <model> --all-sectors) with the updated "
            "ml_5_model_evaluation.py to enable honest CV.",
            stacklevel=2,
        )
        df["fold_set"] = "outer"
    else:
        df["fold_set"] = df["fold_set"].astype(str)

    # ── Variant identifier construction ───────────────────────────────────
    if has_run_id:
        df["run_id"] = df["run_id"].astype(str)
        variant_keys = ["model_name", "run_id"]
    else:
        warnings.warn(
            "Predictions table has no 'run_id' column. Variants will be "
            "identified by model_name only — preset-level variation will be "
            "collapsed together. Re-run pipeline to capture run_id.",
            stacklevel=2,
        )
        variant_keys = ["model_name"]

    df["_ae"] = (df["y_pred"] - df["y_true"]).abs()

    return df, {
        "table": table,
        "has_fold_set": has_fold_set,
        "has_run_id": has_run_id,
        "variant_keys": variant_keys,
    }


def _to_canonical(selected: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Project selected outer-fold rows to the comparison canonical schema.

    Stamps ``model_name``, fills the PI columns with NaN (the eval DB stores
    only the median), and deduplicates on (sector, target_date, horizon).
    Returns an empty frame (with the canonical columns) when ``selected`` is empty.
    """
    out = pd.DataFrame({
        "model_name":  model_name,
        "sector_code": selected["sector_code"].astype(str),
        "origin_date": pd.to_datetime(selected["origin_date"]),
        "target_date": pd.to_datetime(selected["target_date"]),
        "horizon":     selected["horizon"].astype(int),
        "y_true":      selected["y_true"].astype(float),
        "y_pred":      selected["y_pred"].astype(float),
        "y_lower_80":  np.nan,
        "y_upper_80":  np.nan,
        "y_lower_95":  np.nan,
        "y_upper_95":  np.nan,
    })
    if out.empty:
        return out
    return (out.sort_values(["sector_code", "target_date", "horizon", "origin_date"])
               .drop_duplicates(subset=["sector_code", "target_date", "horizon"], keep="last")
               .reset_index(drop=True))


def load_pipeline_honest(
    eval_db_path: "str | Path",
    table: Optional[str] = None,
    run_filter: Optional[dict] = None,
    variant_selection: str = "per_sector_honest",
    exclude_baseline: bool = True,
    baseline_prefix: str = _BASELINE_PREFIX,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Pipeline predictions with honest nested-CV variant selection.

    Parameters
    ----------
    eval_db_path : str or Path
        Path to the SQLite eval database (e.g. data/4_eval/eval_data.db).
    table : str or None
        Predictions table.  If None, auto-detected as the first present table
        from {model_predictions, model_evaluation_records}.
    run_filter : dict or None
        Optional column→value filter (e.g. {"model_name": "ridge_300003"})
        applied AFTER loading.  Note: preset name is not stored in
        model_predictions, only as an MLflow tag — so filtering by preset
        requires looking up run_ids from MLflow first.
    variant_selection : str
        One of "per_sector_honest" (DEFAULT), "global", "per_sector_legacy".
    exclude_baseline : bool
        If True (default), rows whose model_name starts with ``baseline_prefix``
        are removed before any variant selection.
    baseline_prefix : str
        Prefix used to identify baseline rows.

    Returns
    -------
    canonical : DataFrame
        Canonical-schema predictions ready for evaluation_method routines.
        Family-level ``model_name="Pipeline"`` so cross-method Friedman /
        Diebold-Mariano tests work uniformly.
    winners_table : DataFrame
        Columns: sector_code, winning_variant (model_name), winning_run_id,
        inner_mae, n_inner_predictions, n_outer_predictions.
    """
    if variant_selection not in _VALID_MODES:
        raise ValueError(
            f"variant_selection must be one of {_VALID_MODES}, got {variant_selection!r}"
        )

    df, _meta = _load_predictions_df(eval_db_path, table=table, run_filter=run_filter)
    has_run_id   = _meta["has_run_id"]
    variant_keys = _meta["variant_keys"]

    # ── Filter baseline rows (CRITICAL) ───────────────────────────────────
    if exclude_baseline:
        baseline_mask = df["model_name"].str.startswith(baseline_prefix)
        n_baseline_rows = int(baseline_mask.sum())
        if n_baseline_rows > 0:
            n_baseline_models = df.loc[baseline_mask, "model_name"].nunique()
            print(f"  [pipeline_loader] Excluded {n_baseline_rows:,} baseline rows "
                  f"({n_baseline_models} model_names matching '{baseline_prefix}*')")
        df = df[~baseline_mask].copy()
        if df.empty:
            raise ValueError("After excluding baselines, no Pipeline rows remain.")

    inner_df = df[df["fold_set"] == "inner"]
    outer_df = df[df["fold_set"] == "outer"]

    # ── Variant selection ────────────────────────────────────────────────
    if variant_selection == "per_sector_honest":
        canonical, winners_table = _select_per_sector_honest(
            df, inner_df, outer_df, variant_keys, has_run_id
        )
    elif variant_selection == "global":
        canonical, winners_table = _select_global(
            df, inner_df, outer_df, variant_keys, has_run_id
        )
    else:  # "per_sector_legacy"
        canonical, winners_table = _select_per_sector_legacy(
            df, outer_df, variant_keys, has_run_id
        )

    # ── Project to canonical schema (family-level model_name="Pipeline") ──
    if canonical.empty:
        raise ValueError(
            "No canonical predictions produced. Check that the predictions "
            "table contains outer-fold rows for the selected variants. "
            "If only inner rows exist (CV truncated everywhere), re-run with "
            "more data per sector."
        )

    canonical_out = _to_canonical(canonical, "Pipeline")

    return canonical_out, winners_table


# ═══════════════════════════════════════════════════════════════════════════
# Selection strategies
# ═══════════════════════════════════════════════════════════════════════════

def _select_per_sector_honest(
    df: pd.DataFrame,
    inner_df: pd.DataFrame,
    outer_df: pd.DataFrame,
    variant_keys: list,
    has_run_id: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Honest nested-CV per-sector selection.

    For each sector, picks the (model_name, run_id) variant with lowest
    INNER-fold MAE.  Reports OUTER-fold predictions of those winners.
    """
    if inner_df.empty:
        warnings.warn(
            "No inner-fold rows found. Variant selection falls back to "
            "outer-fold MAE (biased — equivalent to per_sector_legacy). "
            "To enable honest selection, re-run the sector sweep "
            "(main.py <table> <model> --all-sectors) with the updated "
            "ml_5_model_evaluation.py.",
            stacklevel=3,
        )
        return _select_per_sector_legacy(df, outer_df, variant_keys, has_run_id)

    n_variants = inner_df.groupby(variant_keys, observed=True).ngroups
    print(f"  [pipeline_loader] per_sector_honest: "
          f"picking variant per sector by INNER-fold MAE "
          f"({len(inner_df):,} inner rows across {n_variants} distinct variants)")

    # Per-sector inner MAE
    inner_mae = (inner_df.groupby(["sector_code"] + variant_keys, observed=True)
                          .agg(inner_mae=("_ae", "mean"),
                               n_inner_predictions=("_ae", "size"))
                          .reset_index())

    # Pick lowest inner MAE per sector
    winners = (inner_mae.sort_values(["sector_code", "inner_mae"])
                         .drop_duplicates("sector_code", keep="first")
                         .reset_index(drop=True))

    # Annotate outer counts
    outer_counts = (outer_df.groupby(["sector_code"] + variant_keys, observed=True)
                             .size().reset_index(name="n_outer_predictions"))
    winners = winners.merge(outer_counts,
                            on=["sector_code"] + variant_keys,
                            how="left")
    winners["n_outer_predictions"] = winners["n_outer_predictions"].fillna(0).astype(int)

    # Restrict outer rows to the chosen variants — joined on the FULL
    # (sector_code, model_name, run_id) tuple
    canonical = outer_df.merge(
        winners[["sector_code"] + variant_keys],
        on=["sector_code"] + variant_keys,
        how="inner",
    )

    # Build user-facing winners_table
    winners_table = winners.rename(columns={"model_name": "winning_variant"})
    if has_run_id:
        winners_table = winners_table.rename(columns={"run_id": "winning_run_id"})

    cols_in_order = ["sector_code", "winning_variant"]
    if has_run_id:
        cols_in_order.append("winning_run_id")
    cols_in_order.extend(["inner_mae", "n_inner_predictions", "n_outer_predictions"])
    winners_table = winners_table[cols_in_order]

    # Warn if any winner has no outer predictions
    no_outer = winners_table[winners_table["n_outer_predictions"] == 0]
    if not no_outer.empty:
        warnings.warn(
            f"{len(no_outer)} sectors selected a variant with NO outer-fold "
            f"predictions; those sectors will be missing from canonical. "
            f"Sectors affected (up to 5 shown): "
            f"{no_outer['sector_code'].tolist()[:5]}",
            stacklevel=3,
        )

    return canonical, winners_table


def _select_global(
    df: pd.DataFrame,
    inner_df: pd.DataFrame,
    outer_df: pd.DataFrame,
    variant_keys: list,
    has_run_id: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Global model-family selection.

    Step 1: pick the FAMILY (model_name with sector suffix stripped) with
            lowest aggregate INNER-fold MAE across all sectors and runs.
    Step 2: within that family, per sector pick the (model_name, run_id) with
            lowest INNER-fold MAE.
    Step 3: report outer-fold predictions of those per-sector winners.

    This is more conservative than per_sector_honest: the family is fixed
    globally, only the run_id (i.e. preset) can vary per sector.
    """
    if inner_df.empty:
        warnings.warn(
            "No inner-fold rows; global selection falls back to outer-fold "
            "aggregate (biased). Re-run pipeline with updated ml_5 to fix.",
            stacklevel=3,
        )
        inner_df = outer_df.copy()

    # Family extraction: strip the trailing _<sector> token from model_name
    inner_df = inner_df.copy()
    inner_df["_family"] = inner_df["model_name"].str.rsplit("_", n=1).str[0]
    outer_df = outer_df.copy()
    outer_df["_family"] = outer_df["model_name"].str.rsplit("_", n=1).str[0]

    family_mae = (inner_df.groupby("_family")["_ae"]
                           .agg(["mean", "count"])
                           .rename(columns={"mean": "family_inner_mae", "count": "n"})
                           .sort_values("family_inner_mae"))
    winning_family = family_mae.index[0]

    print(f"  [pipeline_loader] GLOBAL family selection: '{winning_family}' "
          f"(inner MAE = {family_mae.iloc[0]['family_inner_mae']:.4f} over "
          f"{int(family_mae.iloc[0]['n']):,} inner predictions)")
    print(f"  Other families by inner MAE:")
    for fam, row in family_mae.iloc[1:6].iterrows():
        print(f"    {fam:<30s}: inner MAE = {row['family_inner_mae']:.4f}  "
              f"({int(row['n']):,} preds)")

    # Restrict to winning family, then per-sector best run within that family
    inner_winning_family = inner_df[inner_df["_family"] == winning_family]
    outer_winning_family = outer_df[outer_df["_family"] == winning_family]

    # Reuse per_sector_honest logic on the family-restricted data
    canonical, winners_table = _select_per_sector_honest(
        df, inner_winning_family, outer_winning_family, variant_keys, has_run_id
    )

    # Tag the winners_table with family info
    winners_table = winners_table.copy()
    winners_table["winning_family"] = winning_family
    return canonical, winners_table


def _select_per_sector_legacy(
    df: pd.DataFrame,
    outer_df: pd.DataFrame,
    variant_keys: list,
    has_run_id: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy mode: per-sector variant picked on outer-fold MAE (test-set biased).

    Kept for diagnostic regression — DO NOT use for the final comparison.
    """
    print(f"  [pipeline_loader] ⚠ per_sector_legacy mode — variants chosen on "
          f"outer-fold MAE (test-set biased). For diagnostic use only.")

    base_df = outer_df if not outer_df.empty else df

    sector_variant_mae = (base_df.groupby(["sector_code"] + variant_keys, observed=True)
                                 .agg(legacy_mae=("_ae", "mean"),
                                      n_predictions=("_ae", "size"))
                                 .reset_index())
    winners = (sector_variant_mae.sort_values(["sector_code", "legacy_mae"])
                                  .drop_duplicates("sector_code", keep="first")
                                  .reset_index(drop=True))

    canonical = base_df.merge(
        winners[["sector_code"] + variant_keys],
        on=["sector_code"] + variant_keys,
        how="inner",
    )

    winners_table = winners.rename(columns={"model_name": "winning_variant",
                                             "legacy_mae": "inner_mae"})
    if has_run_id:
        winners_table = winners_table.rename(columns={"run_id": "winning_run_id"})
    winners_table["n_inner_predictions"] = 0  # n/a in legacy mode
    winners_table["n_outer_predictions"] = winners_table.pop("n_predictions")
    return canonical, winners_table


# ═══════════════════════════════════════════════════════════════════════════
# Per-family loader for the cross-method comparison (run_comparison)
# ═══════════════════════════════════════════════════════════════════════════

def load_families_from_eval_db(
    eval_db_path: "str | Path",
    *,
    table: Optional[str] = None,
    variant_selection: str = "per_sector_honest",
    pipeline_families: tuple = _PIPELINE_FAMILIES,
) -> Tuple[list, pd.DataFrame, dict]:
    """Load one canonical-schema DataFrame per model FAMILY from the eval DB.

    A "family" is the model_name with its trailing ``_<sector>`` stripped.
    Within each family, honest per-sector variant selection (default) picks the
    best ``(model_name, run_id)`` per sector by INNER-fold MAE and reports the
    OUTER-fold rows — the same nested-CV discipline as ``load_pipeline_honest``,
    applied per family so families stay distinct for the cross-method comparison.

    The reducer families in ``pipeline_families`` collapse into a single
    ``"Pipeline"`` competitor (best reducer variant per sector).  The stat
    families map to their ``m_evaluation`` operational-table display names.  The
    baseline (``SectorQuarterRollingMean``) is returned separately for skill scores.

    Returns
    -------
    family_dfs : list[pd.DataFrame]
        One canonical-schema DataFrame per family (``model_name`` = display
        name), ready for ``m_evaluation.compare_all_models``.  Families that
        produce no outer-fold rows are skipped.
    baseline_df : pd.DataFrame
        Canonical baseline rows (``model_name="baseline"``); empty if no baseline.
    winners_by_family : dict
        family display name → winners_table from the selector.
    """
    if variant_selection not in _VALID_MODES:
        raise ValueError(
            f"variant_selection must be one of {_VALID_MODES}, got {variant_selection!r}"
        )

    df, meta = _load_predictions_df(eval_db_path, table=table)
    has_run_id   = meta["has_run_id"]
    variant_keys = meta["variant_keys"]
    df["_family"] = df["model_name"].str.rsplit("_", n=1).str[0]

    def _select(sub: pd.DataFrame):
        inner = sub[sub["fold_set"] == "inner"]
        outer = sub[sub["fold_set"] == "outer"]
        if variant_selection == "per_sector_honest":
            return _select_per_sector_honest(sub, inner, outer, variant_keys, has_run_id)
        if variant_selection == "global":
            return _select_global(sub, inner, outer, variant_keys, has_run_id)
        return _select_per_sector_legacy(sub, outer, variant_keys, has_run_id)

    family_dfs: list = []
    winners_by_family: dict = {}

    # ── Baseline (kept separate, for skill scores) ────────────────────────
    base_rows = df[df["_family"] == _BASELINE_PREFIX]
    if not base_rows.empty:
        base_sel, _ = _select(base_rows)
        baseline_df = _to_canonical(base_sel, "baseline")
    else:
        baseline_df = _to_canonical(base_rows, "baseline")  # empty canonical frame

    # ── Pipeline reducers collapsed into one competitor ───────────────────
    pipe_rows = df[df["_family"].isin(pipeline_families)]
    if not pipe_rows.empty:
        pipe_sel, pipe_winners = _select(pipe_rows)
        pipe_canon = _to_canonical(pipe_sel, "Pipeline")
        if not pipe_canon.empty:
            family_dfs.append(pipe_canon)
            winners_by_family["Pipeline"] = pipe_winners

    # ── Every other (stat / unknown) family on its own ────────────────────
    handled = set(pipeline_families) | {_BASELINE_PREFIX}
    for fam in sorted(f for f in df["_family"].unique() if f not in handled):
        sub = df[df["_family"] == fam]
        display = _FAMILY_DISPLAY.get(fam, fam)
        sel, winners = _select(sub)
        canon = _to_canonical(sel, display)
        if not canon.empty:
            family_dfs.append(canon)
            winners_by_family[display] = winners

    return family_dfs, baseline_df, winners_by_family


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.m_pipeline_loader <eval_db_path> "
              "[per_sector_honest|global|per_sector_legacy]")
        sys.exit(1)
    mode = sys.argv[2] if len(sys.argv) > 2 else "per_sector_honest"
    canonical, winners = load_pipeline_honest(sys.argv[1], variant_selection=mode)
    print(f"\nCanonical rows: {len(canonical):,}")
    print(f"Sectors:        {canonical['sector_code'].nunique()}")
    print(f"Horizons:       {sorted(canonical['horizon'].unique())}")
    print(f"\nWinners breakdown:")
    print(winners["winning_variant"].value_counts().to_string())
    if "winning_run_id" in winners.columns:
        n_distinct_runs = winners["winning_run_id"].nunique()
        print(f"\nDistinct winning run_ids across sectors: {n_distinct_runs}")
