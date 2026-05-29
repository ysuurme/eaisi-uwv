"""
Feature Selection Utilities for CBS Panel-Data Forecasting
==========================================================

Pure, reusable functions for statistical feature selection on panel datasets.
Each filter function takes a DataFrame and parameters, returns a standardised
result dictionary.  No global state, no side effects except the explicit
``save_preset_to_json`` function.

Designed for interactive use in Jupyter notebooks::

    from feature_selection_utils import (
        apply_near_constant_filter,
        apply_correlation_filter,
        save_preset_to_json,
    )
    result = apply_near_constant_filter(feature_cols, df, max_fraction=0.95)
    survivors = result["retained"]

Filter result format
--------------------
Every ``apply_*`` function returns a dict with these keys:

- ``"filter"``   – filter name (str)
- ``"params"``   – parameters used (dict)
- ``"input"``    – feature names that entered (list[str])
- ``"retained"`` – feature names that passed (list[str])
- ``"dropped"``  – feature names that were pruned (list[str])
- ``"scores"``   – per-feature scores for visualisation (dict or None)
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import grangercausalitytests

logger = logging.getLogger("feature_selection")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit to match your project
# ═══════════════════════════════════════════════════════════════════════════

#: Columns that are never ML features (keys, dates, identifiers, derived
#: temporal artifacts).  ``"sector"`` is reconstructed from OHE columns by
#: ``DataExtractor.load_full_panel()``.  ``"trend_index"`` is a deterministic
#: linear ramp that becomes constant after differencing.
STRUCTURAL_COLS: set[str] = {
    "silver_id",
    "quarter",
    "year",
    "period_enddate",
    "sector",
    "trend_index",
}

#: Prefix for one-hot-encoded sector columns.
SBI_FILTER_PREFIX: str = "BedrijfskenmerkenSBI2008_"

#: Column identifying the SBI sector for panel-aware operations.
SECTOR_COL: str = "sector"

#: Column used to sort observations chronologically within each sector.
TIME_COL: str = "period_enddate"

#: CBS domain prefixes for domain-based filtering.
DOMAIN_PREFIXES: list[str] = [
    "GewerkteUren",
    "GewerkteUrenPerWerkzamePersoon",
    "GewerkteUrenPerBaan",
    "GewerkteUrenSeizoengecorrigeerd",
    "WerkzamePersonen",
    "WerkzamePersonenSeizoengecorrigeerd",
    "Banen",
    "BanenSeizoengecorrigeerd",
]

#: Maps CBS metric family prefixes → CBS table IDs.
SOURCE_TABLE_LOOKUP: dict[str, str] = {
    "Banen": "85920NED",
    "BanenSeizoengecorrigeerd": "85920NED",
    "GewerkteUren": "85920NED",
    "GewerkteUrenSeizoengecorrigeerd": "85920NED",
    "GewerkteUrenPerBaan": "85920NED",
    "GewerkteUrenPerWerkzamePersoon": "85920NED",
    "WerkzamePersonen": "85920NED",
    "WerkzamePersonenSeizoengecorrigeerd": "85920NED",
}

#: Merges fine-grained metric prefixes → higher-level domain groups.
MERGE_MAP: dict[str, str] = {
    "GewerkteUren": "labor_volume",
    "GewerkteUrenSeizoengecorrigeerd": "labor_volume",
    "GewerkteUrenPerBaan": "labor_volume",
    "GewerkteUrenPerWerkzamePersoon": "labor_volume",
    "WerkzamePersonen": "workforce",
    "WerkzamePersonenSeizoengecorrigeerd": "workforce",
    "Banen": "workforce",
    "BanenSeizoengecorrigeerd": "workforce",
}


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _extract_metric_prefix(column_name: str) -> str:
    """Extract the CBS metric family prefix from a column name.

    CBS gold columns follow ``MetricFamily_SuffixId_Dim1_Dim2``.  Returns
    everything before the first ``_digit`` boundary.

    >>> _extract_metric_prefix("GewerkteUren_3_A045285_4000")
    'GewerkteUren'
    >>> _extract_metric_prefix("BanenSeizoengecorrigeerd_10_A045285")
    'BanenSeizoengecorrigeerd'
    """
    match = re.match(r"^(.+?)_\d", column_name)
    return match.group(1) if match else column_name


def _make_result(
    filter_name: str,
    params: dict,
    input_features: list[str],
    retained: list[str],
    dropped: list[str],
    scores: dict | None = None,
) -> dict[str, Any]:
    """Build the standardised result dict returned by every filter."""
    return {
        "filter": filter_name,
        "params": params,
        "input": list(input_features),
        "retained": list(retained),
        "dropped": list(dropped),
        "scores": scores,
    }


def identify_feature_columns(
    df: pd.DataFrame,
    target: str,
    structural_cols: set[str] | None = None,
    sbi_prefix: str | None = None,
) -> list[str]:
    """Identify feature columns by excluding structural, target, and OHE columns.

    Convenience function to avoid repeating this logic in notebooks.

    Parameters
    ----------
    df : pd.DataFrame
        The full gold panel dataset.
    target : str
        Target column name (e.g. ``"Ziekteverzuimpercentage"``).
    structural_cols : set[str], optional
        Columns to exclude.  Defaults to ``STRUCTURAL_COLS``.
    sbi_prefix : str, optional
        Prefix for OHE sector columns.  Defaults to ``SBI_FILTER_PREFIX``.

    Returns
    -------
    list[str]
        Sorted list of feature column names.
    """
    if structural_cols is None:
        structural_cols = STRUCTURAL_COLS
    if sbi_prefix is None:
        sbi_prefix = SBI_FILTER_PREFIX

    return sorted([
        c for c in df.columns
        if c not in structural_cols
        and c != target
        and not c.startswith(sbi_prefix)
    ])


def identify_yearly_feature_columns(
    feature_cols: list[str],
    prefix: str = "y_",
) -> list[str]:
    """Identify features that originated from yearly CBS tables.

    The gold layer prefixes all yearly-origin feature columns with ``y_``
    during transformation (see ``transform_generic_feature_table`` with
    ``lag_years > 0``).  This function simply filters on that prefix.

    Parameters
    ----------
    feature_cols : list[str]
        All feature column names (e.g. from ``identify_feature_columns()``).
    prefix : str, default ``"y_"``
        The prefix applied to yearly features by the gold layer.

    Returns
    -------
    list[str]
        Feature column names that start with ``prefix``.
    """
    yearly = [c for c in feature_cols if c.startswith(prefix)]
    if yearly:
        logger.info("identify_yearly: %d features with '%s' prefix", len(yearly), prefix)
    return yearly


# ═══════════════════════════════════════════════════════════════════════════
# FILTER 1 — NEAR-CONSTANT
# ═══════════════════════════════════════════════════════════════════════════

def apply_near_constant_filter(
    feature_list: list[str],
    df: pd.DataFrame,
    max_fraction: float = 0.95,
) -> dict[str, Any]:
    """Drop features where a single value dominates the column.

    A feature is dropped when its most frequent value accounts for more than
    ``max_fraction`` of all non-null observations.  Unlike a variance
    threshold, this is scale-independent.

    Parameters
    ----------
    feature_list : list[str]
        Column names to evaluate.
    df : pd.DataFrame
        Full dataset.
    max_fraction : float, default 0.95
        Drop threshold.  A feature is dropped when
        ``mode_count / total_count > max_fraction``.

    Returns
    -------
    dict
        Result dict.  ``scores[feature]`` = fraction of the dominant value.
    """
    scores: dict[str, float] = {}
    retained, dropped = [], []

    for col in feature_list:
        series = df[col].dropna()
        if len(series) == 0:
            dominant_frac = 1.0
        else:
            dominant_frac = float(series.value_counts(normalize=True).iloc[0])
        scores[col] = dominant_frac

        if dominant_frac > max_fraction:
            dropped.append(col)
        else:
            retained.append(col)

    logger.info(
        "near_constant          retained=%d  dropped=%d  (max_fraction=%.2f)",
        len(retained), len(dropped), max_fraction,
    )
    return _make_result("near_constant", {"max_fraction": max_fraction},
                        feature_list, retained, dropped, scores)


# ═══════════════════════════════════════════════════════════════════════════
# FILTER 2 — REDUNDANCY (hierarchical clustering)
# ═══════════════════════════════════════════════════════════════════════════

def apply_redundancy_filter(
    feature_list: list[str],
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.90,
    sector_col: str | None = None,
) -> dict[str, Any]:
    """Remove redundant features using agglomerative clustering on |correlation|.

    Computes a pairwise absolute correlation matrix, applies hierarchical
    clustering (complete linkage), and cuts at distance ``1 - threshold``.
    From each cluster, the feature with the highest target correlation (after
    within-sector differencing if ``sector_col`` is provided) is kept.

    Parameters
    ----------
    feature_list : list[str]
        Column names to evaluate.
    df : pd.DataFrame
        Full dataset.
    target : str
        Target column name.
    threshold : float, default 0.90
        Features with ``|r| >= threshold`` are considered redundant.
    sector_col : str or None
        If provided, target correlation for tie-breaking uses within-sector
        differencing (panel-aware).

    Returns
    -------
    dict
        Result dict.  ``scores[feature]`` = cluster ID assigned to each feature.
    """
    if len(feature_list) <= 1:
        return _make_result("redundancy", {"threshold": threshold, "sector_col": sector_col},
                            feature_list, list(feature_list), [], {f: 0 for f in feature_list})

    with np.errstate(divide="ignore", invalid="ignore"):
        corr_matrix = df[feature_list].corr().abs().fillna(0.0)
    np.fill_diagonal(corr_matrix.values, 1.0)

    distance = 1.0 - corr_matrix
    np.fill_diagonal(distance.values, 0.0)
    distance = distance.clip(lower=0.0)

    condensed = squareform(distance.values, checks=False)
    Z = linkage(condensed, method="complete")
    cluster_labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    # Target correlation for tie-breaking within clusters
    with np.errstate(divide="ignore", invalid="ignore"):
        if sector_col is not None:
            diff_frames = []
            for _, group in df.groupby(sector_col):
                group_sorted = group.sort_values(TIME_COL)
                diff_frame = group_sorted[feature_list + [target]].diff().iloc[1:]
                diff_frames.append(diff_frame)
            df_diff = pd.concat(diff_frames, ignore_index=True)
            target_corr = df_diff[feature_list].corrwith(df_diff[target]).abs().fillna(0.0)
        else:
            target_corr = df[feature_list].corrwith(df[target]).abs().fillna(0.0)

    retained, dropped = [], []
    cluster_map: dict[str, int] = {}
    for cluster_id in np.unique(cluster_labels):
        members = [
            feature_list[i] for i, cl in enumerate(cluster_labels)
            if cl == cluster_id
        ]
        best = max(members, key=lambda f: target_corr.get(f, 0.0))
        retained.append(best)
        dropped.extend([m for m in members if m != best])
        for m in members:
            cluster_map[m] = int(cluster_id)

    logger.info(
        "redundancy             retained=%d  dropped=%d  (threshold=%.2f)",
        len(retained), len(dropped), threshold,
    )
    return _make_result("redundancy", {"threshold": threshold, "sector_col": sector_col},
                        feature_list, retained, dropped, cluster_map)


# ═══════════════════════════════════════════════════════════════════════════
# FILTER 3 — CORRELATION (contemporaneous)
# ═══════════════════════════════════════════════════════════════════════════

def apply_correlation_filter(
    feature_list: list[str],
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.05,
    method: str = "within_sector_differenced",
    sector_col: str | None = None,
) -> dict[str, Any]:
    """Keep features whose absolute correlation with the target meets ``threshold``.

    Two methods are available:

    - ``"pooled"``: plain Pearson correlation on the raw pooled data.
    - ``"within_sector_differenced"``: first-difference within each sector
      (removing sector-level means and trends), then compute correlation on
      the pooled differenced data.  This is the recommended method for panel
      data because it avoids spurious correlations from cross-sectional
      level differences.

    Parameters
    ----------
    feature_list : list[str]
        Column names to evaluate.
    df : pd.DataFrame
        Full dataset.
    target : str
        Target column name.
    threshold : float, default 0.05
        Minimum ``|r|`` to retain a feature.
    method : str, default ``"within_sector_differenced"``
        ``"pooled"`` or ``"within_sector_differenced"``.
    sector_col : str or None
        Required when ``method="within_sector_differenced"``.

    Returns
    -------
    dict
        Result dict.  ``scores[feature]`` = signed Pearson correlation.
    """
    if method == "within_sector_differenced":
        if sector_col is None:
            raise ValueError("sector_col required for within_sector_differenced")
        diff_frames = []
        for _, group in df.groupby(sector_col):
            group_sorted = group.sort_values(TIME_COL)
            diff_frame = group_sorted[feature_list + [target]].diff().iloc[1:]
            diff_frames.append(diff_frame)
        df_work = pd.concat(diff_frames, ignore_index=True)
    elif method == "pooled":
        df_work = df
    else:
        raise ValueError(f"Unknown method {method!r}")

    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = df_work[feature_list].corrwith(df_work[target]).dropna()
    abs_corr = correlations.abs()

    retained = abs_corr[abs_corr >= threshold].index.tolist()
    dropped = abs_corr[abs_corr < threshold].index.tolist()
    # Include features that produced NaN correlation in the dropped list
    nan_features = [f for f in feature_list if f not in correlations.index]
    dropped.extend(nan_features)

    scores = {col: float(correlations.get(col, 0.0)) for col in feature_list}

    logger.info(
        "correlation            retained=%d  dropped=%d  (threshold=%.3f, method=%s)",
        len(retained), len(dropped), threshold, method,
    )
    return _make_result("correlation", {"threshold": threshold, "method": method,
                        "sector_col": sector_col}, feature_list, retained, dropped, scores)


# ═══════════════════════════════════════════════════════════════════════════
# FILTER 4 — TREE IMPORTANCE (permutation-based)
# ═══════════════════════════════════════════════════════════════════════════

def apply_tree_filter(
    feature_list: list[str],
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.005,
    n_estimators: int = 300,
    sector_col: str | None = None,
) -> dict[str, Any]:
    """Score features using permutation importance from an ExtraTreesRegressor.

    Fits an ``ExtraTreesRegressor`` and computes permutation importance (not
    Mean Decrease in Impurity, which is biased toward high-cardinality
    features).  When ``sector_col`` is provided, the data is first-differenced
    within each sector so the tree learns temporal patterns rather than
    cross-sectional level differences.

    Parameters
    ----------
    feature_list : list[str]
        Column names to evaluate.
    df : pd.DataFrame
        Full dataset.
    target : str
        Target column name.
    threshold : float, default 0.005
        Minimum permutation importance to retain.  Measured in R² decrease
        (e.g. 0.005 = keep features whose permutation drops R² by ≥0.5%).
    n_estimators : int, default 300
        Number of trees.
    sector_col : str or None
        If provided, first-difference within each sector before fitting.

    Returns
    -------
    dict
        Result dict.  ``scores[feature]`` = mean permutation importance.
    """
    if sector_col is not None:
        diff_frames = []
        for _, group in df.groupby(sector_col):
            group_sorted = group.sort_values(TIME_COL)
            diff_frame = group_sorted[feature_list + [target]].diff().iloc[1:]
            diff_frames.append(diff_frame)
        df_work = pd.concat(diff_frames, ignore_index=True).dropna()
    else:
        df_work = df[feature_list + [target]].dropna()

    X = df_work[feature_list]
    y = df_work[target]

    model = ExtraTreesRegressor(
        n_estimators=n_estimators, random_state=42, n_jobs=-1,
    )
    model.fit(X, y)

    perm = sk_permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=-1,
    )
    importances = perm.importances_mean
    importance_std = perm.importances_std

    retained = [c for c, imp in zip(feature_list, importances) if imp >= threshold]
    dropped = [c for c, imp in zip(feature_list, importances) if imp < threshold]

    scores = {
        col: {"mean": float(imp), "std": float(std)}
        for col, imp, std in zip(feature_list, importances, importance_std)
    }

    logger.info(
        "tree_importance        retained=%d  dropped=%d  (threshold=%.4f)",
        len(retained), len(dropped), threshold,
    )
    return _make_result("tree_importance", {"threshold": threshold,
                        "n_estimators": n_estimators, "sector_col": sector_col},
                        feature_list, retained, dropped, scores)


# ═══════════════════════════════════════════════════════════════════════════
# FILTER 5 — LAGGED CORRELATION (predictive value)
# ═══════════════════════════════════════════════════════════════════════════

def apply_lagged_correlation_filter(
    feature_list: list[str],
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.10,
    horizons: list[int] | None = None,
    sector_col: str | None = None,
    agg_func: Any = None,
) -> dict[str, Any]:
    """Keep features whose lagged correlation with the target is strong enough.

    For each horizon ``h``, computes ``corr(feature_t, target_{t+h})`` within
    each sector, aggregates across sectors (median by default), and retains
    features where the aggregated score meets ``threshold`` at *any* horizon.

    Parameters
    ----------
    feature_list : list[str]
        Column names to evaluate.
    df : pd.DataFrame
        Full dataset.
    target : str
        Target column name.
    threshold : float, default 0.10
        Minimum aggregated ``|r|`` at any horizon to retain.
    horizons : list[int], optional
        Forecast horizons to evaluate.  Defaults to ``[1, 2, 3]``.
    sector_col : str or None
        Column identifying sectors for panel-aware computation.
    agg_func : callable, optional
        Aggregation across sectors.  Defaults to ``np.median``.

    Returns
    -------
    dict
        Result dict.  ``scores[feature]`` = dict mapping horizon → aggregated |r|.
    """
    if horizons is None:
        horizons = [1, 2, 3]
    if agg_func is None:
        agg_func = np.median

    if sector_col is None:
        sector_groups = [("__all__", df.sort_values(TIME_COL))]
    else:
        sector_groups = [
            (sector, group.sort_values(TIME_COL))
            for sector, group in df.groupby(sector_col)
        ]

    # Pre-compute shifted targets per horizon per sector
    shifted_targets: dict[int, list] = {}
    for h in horizons:
        shifted_targets[h] = [
            (sector, group_sorted, group_sorted[target].shift(-h))
            for sector, group_sorted in sector_groups
        ]

    all_scores: dict[str, dict[int, float]] = {}
    n_features = len(feature_list)
    with np.errstate(divide="ignore", invalid="ignore"):
        for feat_idx, feat in enumerate(feature_list):
            if (feat_idx + 1) % 50 == 0 or feat_idx == 0:
                logger.debug("lagged_correlation     processing feature %d/%d",
                             feat_idx + 1, n_features)
            feat_scores: dict[int, float] = {}
            for h in horizons:
                sector_corrs: list[float] = []
                for sector, group_sorted, future_target in shifted_targets[h]:
                    valid = group_sorted[feat].notna() & future_target.notna()
                    if valid.sum() > 10:
                        r = group_sorted.loc[valid, feat].corr(future_target[valid])
                        if not np.isnan(r):
                            sector_corrs.append(abs(r))
                feat_scores[h] = float(agg_func(sector_corrs)) if sector_corrs else 0.0
            all_scores[feat] = feat_scores

    retained, dropped = [], []
    for feat in feature_list:
        best_score = max(all_scores[feat].values()) if all_scores[feat] else 0.0
        if best_score >= threshold:
            retained.append(feat)
        else:
            dropped.append(feat)

    logger.info(
        "lagged_correlation     retained=%d  dropped=%d  (threshold=%.3f, horizons=%s)",
        len(retained), len(dropped), threshold, horizons,
    )
    return _make_result("lagged_correlation",
                        {"threshold": threshold, "horizons": horizons, "sector_col": sector_col},
                        feature_list, retained, dropped, all_scores)


# ═══════════════════════════════════════════════════════════════════════════
# FILTER 6 — GRANGER CAUSALITY
# ═══════════════════════════════════════════════════════════════════════════

def apply_granger_filter(
    feature_list: list[str],
    df: pd.DataFrame,
    target: str,
    max_lag: int = 4,
    p_threshold: float = 0.05,
    min_sector_fraction: float = 0.20,
    sector_col: str | None = None,
    difference: bool = True,
) -> dict[str, Any]:
    """Keep features that Granger-cause the target in enough sectors.

    For each feature × sector combination, runs a Granger causality test up
    to ``max_lag``.  A feature is retained if it is significant (best p-value
    across lags < ``p_threshold``) in at least ``min_sector_fraction`` of sectors.

    Parameters
    ----------
    feature_list : list[str]
        Column names to evaluate.
    df : pd.DataFrame
        Full dataset.
    target : str
        Target column name.
    max_lag : int, default 4
        Maximum lag order for the Granger test.
    p_threshold : float, default 0.05
        Significance level.
    min_sector_fraction : float, default 0.20
        Minimum fraction of sectors where Granger is significant.
    sector_col : str or None
        Column identifying sectors.
    difference : bool, default True
        First-difference the data before testing (recommended for
        non-stationary panel data).

    Returns
    -------
    dict
        Result dict.  ``scores[feature]`` = fraction of sectors where significant.
    """
    if sector_col is None:
        sector_groups = [("__all__", df.sort_values(TIME_COL))]
    else:
        sector_groups = [
            (sector, group.sort_values(TIME_COL))
            for sector, group in df.groupby(sector_col)
        ]

    scores: dict[str, float] = {}
    n_features = len(feature_list)

    for feat_idx, feat in enumerate(feature_list):
        if (feat_idx + 1) % 50 == 0 or feat_idx == 0:
            logger.debug("granger                processing feature %d/%d",
                         feat_idx + 1, n_features)
        sector_results: list[bool] = []
        n_skipped_gaps = 0
        n_failed = 0
        last_failure_reason = ""

        for sector, group_sorted in sector_groups:
            data = group_sorted[[target, feat]].dropna()

            # Check temporal contiguity before differencing
            full_time = group_sorted[TIME_COL]
            positions = full_time.index.get_indexer(data.index)
            gaps = np.diff(positions)
            if len(positions) > 1 and (gaps > 1).any():
                n_skipped_gaps += 1
                continue

            if difference:
                data = data.diff().iloc[1:]

            if len(data) < 3 * max_lag:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    test_out = grangercausalitytests(
                        data[[target, feat]].values, maxlag=max_lag, verbose=False,
                    )
                min_p = min(
                    test_out[lag][0]["ssr_ftest"][1]
                    for lag in range(1, max_lag + 1)
                )
                sector_results.append(min_p < p_threshold)
            except Exception as exc:
                n_failed += 1
                last_failure_reason = str(exc)
                continue

        if n_skipped_gaps > 0:
            logger.debug("granger  %s: skipped %d/%d sectors (temporal gaps)",
                         feat, n_skipped_gaps, len(sector_groups))
        if n_failed > 0:
            logger.debug("granger  %s: failed %d/%d sectors (%s)",
                         feat, n_failed, len(sector_groups), last_failure_reason)

        if sector_results:
            frac = sum(sector_results) / len(sector_results)
        else:
            frac = 0.0
        scores[feat] = frac

    retained = [f for f in feature_list if scores.get(f, 0.0) >= min_sector_fraction]
    dropped = [f for f in feature_list if scores.get(f, 0.0) < min_sector_fraction]

    logger.info(
        "granger                retained=%d  dropped=%d  "
        "(p=%.2f, min_frac=%.2f, max_lag=%d, difference=%s)",
        len(retained), len(dropped), p_threshold, min_sector_fraction,
        max_lag, difference,
    )
    return _make_result("granger",
                        {"max_lag": max_lag, "p_threshold": p_threshold,
                         "min_sector_fraction": min_sector_fraction,
                         "sector_col": sector_col, "difference": difference},
                        feature_list, retained, dropped, scores)


# ═══════════════════════════════════════════════════════════════════════════
# FILTER 7 — LASSO STABILITY SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def apply_lasso_stability_filter(
    feature_list: list[str],
    df: pd.DataFrame,
    target: str,
    n_bootstraps: int = 50,
    threshold: float = 0.50,
    random_state: int = 42,
    horizons: list[int] | None = None,
    sector_col: str | None = None,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Keep features that Lasso selects consistently across bootstrap samples.

    Runs ``LassoCV`` on ``n_bootstraps`` resampled datasets.  For each
    bootstrap, records which features receive a non-zero coefficient.  A
    feature is retained if its selection probability (fraction of bootstraps
    where it was selected) meets ``threshold``.

    When ``horizons`` is provided, the filter becomes **forecast-aligned**:
    instead of testing ``feature_t → target_t`` (contemporaneous), it tests
    ``feature_t → target_{t+h}`` for each horizon ``h``.  The target is
    shifted within each sector (using ``sector_col``) to avoid cross-sector
    contamination.  A feature's final selection probability is the **maximum**
    across horizons — it passes if Lasso consistently selects it at *any*
    forecast horizon.

    Parameters
    ----------
    feature_list : list[str]
        Column names to evaluate.
    df : pd.DataFrame
        Full dataset.
    target : str
        Target column name.
    n_bootstraps : int, default 50
        Number of bootstrap samples.
    threshold : float, default 0.50
        Minimum selection probability to retain.
    random_state : int, default 42
        Base random seed (incremented per bootstrap and per horizon).
    horizons : list[int] or None, default None
        Forecast horizons to evaluate (e.g. ``[1, 2, 3]``).  When ``None``,
        Lasso tests contemporaneous importance (``feature_t → target_t``).
        When provided, Lasso tests forecast-aligned importance
        (``feature_t → target_{t+h}``).
    sector_col : str or None, default None
        Column identifying sectors.  Required when ``horizons`` is provided
        to shift the target within each sector correctly.
    n_jobs : int, default -1
        Number of parallel jobs for the bootstrap loop.  -1 uses all
        available CPU cores.  1 disables parallelism.

    Returns
    -------
    dict
        Result dict.  ``scores[feature]`` = selection probability [0, 1].
        When ``horizons`` is provided, the result also includes a
        ``"horizon_scores"`` key with per-horizon probabilities:
        ``{feature: {h: probability, ...}}``.
    """
    if horizons is not None and sector_col is None:
        raise ValueError("sector_col is required when horizons is provided "
                         "(needed to shift the target within each sector)")

    if horizons is None:
        # ── Contemporaneous mode (original behaviour) ────────────────
        selection_prob = _lasso_bootstrap_loop(
            df[feature_list].values, df[target].values,
            n_bootstraps, random_state, n_jobs=n_jobs,
        )

        scores = {col: float(p) for col, p in zip(feature_list, selection_prob)}
        retained = [c for c, p in zip(feature_list, selection_prob) if p >= threshold]
        dropped = [c for c, p in zip(feature_list, selection_prob) if p < threshold]

        logger.info(
            "lasso_stability        retained=%d  dropped=%d  "
            "(n_boots=%d, threshold=%.2f, horizons=None)",
            len(retained), len(dropped), n_bootstraps, threshold,
        )
        return _make_result("lasso_stability",
                            {"n_bootstraps": n_bootstraps, "threshold": threshold,
                             "random_state": random_state, "horizons": None},
                            feature_list, retained, dropped, scores)

    # ── Forecast-aligned mode: one stability run per horizon ─────────
    horizon_probs: dict[int, np.ndarray] = {}

    for h in horizons:
        logger.info("lasso_stability        horizon h=%d (%d bootstraps)", h, n_bootstraps)

        # Shift target forward by h within each sector
        shifted = df.groupby(sector_col)[target].shift(-h)
        valid = shifted.notna() & df[feature_list].notna().all(axis=1)
        X_h = df.loc[valid, feature_list].values
        y_h = shifted[valid].values

        # Unique seed per horizon so bootstrap samples are independent
        horizon_seed = random_state + h * 10_000

        horizon_probs[h] = _lasso_bootstrap_loop(
            X_h, y_h, n_bootstraps, horizon_seed, n_jobs=n_jobs,
        )

    # Final probability = max across horizons (pass at ANY horizon)
    max_probs = np.maximum.reduce([horizon_probs[h] for h in horizons])

    scores = {col: float(p) for col, p in zip(feature_list, max_probs)}
    retained = [c for c, p in zip(feature_list, max_probs) if p >= threshold]
    dropped = [c for c, p in zip(feature_list, max_probs) if p < threshold]

    logger.info(
        "lasso_stability        retained=%d  dropped=%d  "
        "(n_boots=%d, threshold=%.2f, horizons=%s)",
        len(retained), len(dropped), n_bootstraps, threshold, horizons,
    )

    result = _make_result("lasso_stability",
                          {"n_bootstraps": n_bootstraps, "threshold": threshold,
                           "random_state": random_state, "horizons": horizons,
                           "sector_col": sector_col},
                          feature_list, retained, dropped, scores)

    # Attach per-horizon detail for downstream visualisation
    result["horizon_scores"] = {
        col: {h: float(horizon_probs[h][i]) for h in horizons}
        for i, col in enumerate(feature_list)
    }
    return result


def _lasso_bootstrap_loop(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstraps: int,
    random_state: int,
    n_jobs: int = -1,
) -> np.ndarray:
    """Run the core Lasso bootstrap loop and return selection probabilities.

    Fits ``LassoCV`` **once** on the full data to find the optimal
    regularisation strength (``alpha``), then runs ``n_bootstraps`` plain
    ``Lasso`` fits (with that fixed alpha) on resampled data in parallel.
    This is ~45× faster than running ``LassoCV`` on every bootstrap.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (will be standardised internally).
    y : np.ndarray, shape (n_samples,)
        Target vector.
    n_bootstraps : int
        Number of bootstrap iterations.
    random_state : int
        Base random seed.
    n_jobs : int, default -1
        Number of parallel jobs.  -1 uses all available CPU cores.
        1 disables parallelism (useful for debugging).

    Returns
    -------
    np.ndarray, shape (n_features,)
        Selection probability for each feature, in [0, 1].
    """
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Step 1: find optimal alpha from full data (one LassoCV, ~500 internal fits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso_cv = LassoCV(cv=5, random_state=random_state, max_iter=10_000, tol=1e-2)
        lasso_cv.fit(X, y)
    alpha = lasso_cv.alpha_

    # Step 2: run bootstraps with fixed alpha (plain Lasso, 1 fit each)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_lasso_bootstrap)(X, y, alpha, random_state, i)
        for i in range(n_bootstraps)
    )

    selection_counts = np.sum(results, axis=0)
    return selection_counts / n_bootstraps


def _single_lasso_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    random_state: int,
    boot_idx: int,
) -> np.ndarray:
    """Fit one Lasso bootstrap with pre-computed alpha and return selection mask.

    This function is called in parallel by ``_lasso_bootstrap_loop``.
    """
    rng = np.random.RandomState(random_state + boot_idx)
    idx = rng.choice(len(X), size=len(X), replace=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso = Lasso(alpha=alpha, max_iter=10_000, tol=1e-2)
        lasso.fit(X[idx], y[idx])

    return (np.abs(lasso.coef_) > 1e-10).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# FILTER 8 — DOMAIN PREFIX (CBS table allowlist)
# ═══════════════════════════════════════════════════════════════════════════

def apply_domain_prefix_filter(
    feature_list: list[str],
    prefixes: list[str] | None = None,
) -> dict[str, Any]:
    """Keep only features whose names start with allowed CBS domain prefixes.

    This is a domain-knowledge filter: you define which CBS metric families
    are relevant (e.g. labour, workforce, hours) and everything else is
    excluded.

    Parameters
    ----------
    feature_list : list[str]
        Column names to evaluate.
    prefixes : list[str], optional
        Allowed name prefixes.  Defaults to ``DOMAIN_PREFIXES``.

    Returns
    -------
    dict
        Result dict.  ``scores`` is ``None`` (no statistical score).
    """
    if prefixes is None:
        prefixes = DOMAIN_PREFIXES

    retained = [f for f in feature_list if any(f.startswith(p) for p in prefixes)]
    dropped = [f for f in feature_list if f not in retained]

    logger.info("domain_prefix          retained=%d  dropped=%d  (prefixes=%d)",
                len(retained), len(dropped), len(prefixes))
    return _make_result("domain_prefix", {"prefixes": prefixes},
                        feature_list, retained, dropped, None)


# ═══════════════════════════════════════════════════════════════════════════
# GROUPING — auto-group survivors by CBS column prefix
# ═══════════════════════════════════════════════════════════════════════════

def build_proposed_groups(
    feature_list: list[str],
    source_table_lookup: dict[str, str] | None = None,
    merge_map: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Auto-generate feature groups from CBS column name prefixes.

    Parses the metric family prefix from each column name, groups by prefix,
    and optionally merges fine-grained groups into higher-level domain groups.

    Parameters
    ----------
    feature_list : list[str]
        Feature column names to group.
    source_table_lookup : dict, optional
        Maps prefix → CBS table ID.  Defaults to ``SOURCE_TABLE_LOOKUP``.
    merge_map : dict, optional
        Maps prefix → merged group name.  Defaults to ``MERGE_MAP``.

    Returns
    -------
    dict[str, dict]
        ``{group_name: {"columns": [...], "source_table": str, "description": str}}``
    """
    if source_table_lookup is None:
        source_table_lookup = SOURCE_TABLE_LOOKUP
    if merge_map is None:
        merge_map = MERGE_MAP

    prefix_groups: dict[str, list[str]] = {}
    for col in feature_list:
        prefix = _extract_metric_prefix(col)
        prefix_groups.setdefault(prefix, []).append(col)

    merged: dict[str, list[str]] = {}
    prefix_to_merged: dict[str, str] = {}
    for prefix, cols in prefix_groups.items():
        group_name = merge_map.get(prefix, prefix)
        merged.setdefault(group_name, []).extend(cols)
        prefix_to_merged[prefix] = group_name

    groups: dict[str, dict[str, Any]] = {}
    for group_name, cols in sorted(merged.items()):
        source_tables = set()
        contributing_prefixes = [p for p, g in prefix_to_merged.items() if g == group_name]
        for prefix in contributing_prefixes:
            table = source_table_lookup.get(prefix, "")
            if table:
                source_tables.add(table)

        source_table_str = ", ".join(sorted(source_tables)) if source_tables else ""
        prefix_list = ", ".join(sorted(contributing_prefixes))
        description = (
            f"CBS metrics: {prefix_list} "
            f"({len(cols)} features, {source_table_str or 'source unknown'})."
        )

        groups[group_name] = {
            "columns": sorted(cols),
            "source_table": source_table_str,
            "description": description,
        }

    return groups


def validate_feature_groups(
    proposed_groups: dict[str, dict[str, Any]],
    surviving_features: list[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Intersect surviving features with proposed groups.

    Filters each group to include only survivors.  Returns both the filtered
    groups and any ungrouped survivors.

    Parameters
    ----------
    proposed_groups : dict
        Group definitions from ``build_proposed_groups()``.
    surviving_features : list[str]
        Features that passed your filter pipeline.

    Returns
    -------
    tuple[dict, list[str]]
        ``(filtered_groups, ungrouped_survivors)``
    """
    survivor_set = set(surviving_features)
    assigned: set[str] = set()
    filtered_groups: dict[str, dict[str, Any]] = {}

    for group_name, meta in proposed_groups.items():
        surviving_cols = [c for c in meta["columns"] if c in survivor_set]
        if surviving_cols:
            filtered_groups[group_name] = {
                "columns": surviving_cols,
                "source_table": meta.get("source_table", ""),
                "description": meta.get("description", ""),
            }
            assigned.update(surviving_cols)

    ungrouped = [c for c in surviving_features if c not in assigned]
    return filtered_groups, ungrouped


# ═══════════════════════════════════════════════════════════════════════════
# JSON EXPORT — write results for downstream model_configs.py
# ═══════════════════════════════════════════════════════════════════════════

def save_preset_to_json(
    preset_name: str,
    output_dir: str | Path,
    survivors: list[str],
    feature_groups: dict[str, dict[str, Any]],
    filter_chain: list[dict[str, Any]],
    input_shape: tuple[int, int] | list[int],
    description: str = "",
    ungrouped_survivors: list[str] | None = None,
) -> Path:
    """Serialise a feature selection preset to a JSON file.

    Writes a JSON artifact that ``model_configs.py`` loads to build
    ``FEATURE_CATALOG``.

    Parameters
    ----------
    preset_name : str
        Name for this preset (used as filename: ``preset_{name}.json``).
    output_dir : str or Path
        Directory to write the JSON file.
    survivors : list[str]
        Final list of selected feature names.
    feature_groups : dict
        Validated group definitions from ``validate_feature_groups()``.
    filter_chain : list[dict]
        Result dicts from each filter applied (in order).
    input_shape : tuple or list
        ``(n_rows, n_columns)`` of the input dataset.
    description : str, optional
        Human-readable description of the preset strategy.
    ungrouped_survivors : list[str], optional
        Survivors not assigned to any group.

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if ungrouped_survivors is None:
        ungrouped_survivors = []

    chain_summary = []
    for entry in filter_chain:
        chain_summary.append({
            "filter": entry["filter"],
            "params": {k: v for k, v in entry["params"].items()
                       if not callable(v)},
            "n_input": len(entry["input"]),
            "n_retained": len(entry["retained"]),
            "n_dropped": len(entry["dropped"]),
        })

    artifact = {
        "preset": preset_name,
        "description": description,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_shape": list(input_shape),
        "n_survivors": len(survivors),
        "surviving_features": survivors,
        "ungrouped_survivors": ungrouped_survivors,
        "filter_chain": chain_summary,
        "feature_groups": feature_groups,
    }

    path = output_dir / f"preset_{preset_name}.json"
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)

    print(f"✅ Preset saved: {path}  ({len(survivors)} features, "
          f"{len(feature_groups)} groups)")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# YEARLY FEATURE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def merge_yearly_features(
    df_quarterly: pd.DataFrame,
    df_yearly: pd.DataFrame,
    yearly_feature_cols: list[str],
    year_col: str = "year",
    lag_years: int = 1,
    sector_col: str | None = None,
    yearly_sector_col: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge yearly CBS data into a quarterly panel with configurable lag.

    Each yearly value is broadcast to all 4 quarters of the **lagged** year.
    With ``lag_years=1`` (default), the 2022 yearly value appears in Q1–Q4
    of 2023 — avoiding look-ahead bias because annual data is typically
    published the following year.

    Parameters
    ----------
    df_quarterly : pd.DataFrame
        The quarterly panel dataset (must contain ``year_col``).
    df_yearly : pd.DataFrame
        The yearly CBS dataset.
    yearly_feature_cols : list[str]
        Column names in ``df_yearly`` to merge.
    year_col : str, default ``"year"``
        Year column present in both datasets.
    lag_years : int, default 1
        How many years to lag.  1 means "use last year's value for this
        year's quarters."  0 means no lag (mild look-ahead bias).
    sector_col : str or None
        Sector column in ``df_quarterly`` (e.g. ``"sector"``).
    yearly_sector_col : str or None
        Sector column in ``df_yearly``.  If ``None`` but ``sector_col``
        is provided, the yearly data is treated as national-level and
        broadcast to all sectors.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        ``(merged_df, new_column_names)`` — the quarterly DataFrame with
        yearly features added, and the list of new column names (suffixed
        with ``_y`` if a name collision would occur).
    """
    df_y = df_yearly.copy()

    # Apply lag: shift the year so yearly 2022 maps to quarterly 2023
    df_y[year_col] = df_y[year_col] + lag_years

    # Determine merge keys
    merge_keys = [year_col]
    if sector_col and yearly_sector_col:
        df_y = df_y.rename(columns={yearly_sector_col: sector_col})
        merge_keys.append(sector_col)

    # Resolve column name collisions
    existing_cols = set(df_quarterly.columns)
    rename_map: dict[str, str] = {}
    final_names: list[str] = []
    for col in yearly_feature_cols:
        new_name = col
        if col in existing_cols:
            new_name = f"{col}_yearly"
            rename_map[col] = new_name
        final_names.append(new_name)

    if rename_map:
        df_y = df_y.rename(columns=rename_map)

    # Select only the columns we need for merging
    merge_cols = merge_keys + final_names
    df_y_slim = df_y[merge_cols].drop_duplicates()

    # Merge
    merged = df_quarterly.merge(df_y_slim, on=merge_keys, how="left")

    n_matched = merged[final_names[0]].notna().sum() if final_names else 0
    n_total = len(merged)
    print(f"Yearly merge: {len(final_names)} features, "
          f"lag={lag_years} year(s), "
          f"{n_matched}/{n_total} rows matched "
          f"({n_matched / n_total:.0%})")

    return merged, final_names


def evaluate_yearly_features(
    yearly_feature_cols: list[str],
    df: pd.DataFrame,
    target: str,
    year_col: str = "year",
    sector_col: str | None = None,
    corr_threshold: float = 0.10,
    lag_threshold: float = 0.10,
) -> dict[str, Any]:
    """Evaluate yearly features at annual granularity.

    Quarterly filters penalise yearly features because within-sector
    differencing produces 75% zeros (the value only changes at Q1
    boundaries).  This function evaluates yearly features fairly by
    aggregating the quarterly target to annual means first.

    Two tests are performed at annual resolution:

    1. **Year-over-year differenced correlation** —
       ``corr(Δfeature_year, Δtarget_year)`` within each sector,
       aggregated across sectors using the median.
    2. **One-year-ahead lagged correlation** —
       ``corr(feature_year, target_{year+1})`` within each sector,
       testing whether this year's feature predicts next year's
       absenteeism.

    A feature passes if **either** test exceeds its threshold.

    Parameters
    ----------
    yearly_feature_cols : list[str]
        Column names to evaluate.
    df : pd.DataFrame
        The quarterly panel with yearly features already merged in.
    target : str
        Target column name.
    year_col : str, default ``"year"``
        Year column for annual aggregation.
    sector_col : str or None
        Sector column for panel-aware computation.
    corr_threshold : float, default 0.10
        Minimum median |r| for the differenced correlation test.
    lag_threshold : float, default 0.10
        Minimum median |r| for the one-year-ahead lagged test.

    Returns
    -------
    dict
        Result dict (same format as other filters).
        ``scores[feature]`` = dict with ``"diff_corr"`` and ``"lag_corr"``
        keys, each containing the median |r| across sectors.
    """
    cols_needed = [year_col, target] + yearly_feature_cols
    if sector_col:
        cols_needed.append(sector_col)

    # Aggregate quarterly target to annual means
    group_keys = [year_col] if not sector_col else [sector_col, year_col]
    df_annual = (
        df[cols_needed]
        .dropna(subset=yearly_feature_cols + [target], how="all")
        .groupby(group_keys)
        .mean(numeric_only=True)
        .reset_index()
    )

    if sector_col:
        sector_groups = [
            (s, g.sort_values(year_col))
            for s, g in df_annual.groupby(sector_col)
        ]
    else:
        sector_groups = [("__all__", df_annual.sort_values(year_col))]

    scores: dict[str, dict[str, float]] = {}

    with np.errstate(divide="ignore", invalid="ignore"):
        for feat in yearly_feature_cols:
            # Test 1: year-over-year differenced correlation
            diff_corrs: list[float] = []
            for _, group in sector_groups:
                diffed = group[[feat, target]].diff().iloc[1:]
                valid = diffed[feat].notna() & diffed[target].notna()
                if valid.sum() > 3:
                    r = diffed.loc[valid, feat].corr(diffed.loc[valid, target])
                    if not np.isnan(r):
                        diff_corrs.append(abs(r))

            # Test 2: one-year-ahead lagged correlation
            lag_corrs: list[float] = []
            for _, group in sector_groups:
                future_target = group[target].shift(-1)
                valid = group[feat].notna() & future_target.notna()
                if valid.sum() > 3:
                    r = group.loc[valid, feat].corr(future_target[valid])
                    if not np.isnan(r):
                        lag_corrs.append(abs(r))

            diff_score = float(np.median(diff_corrs)) if diff_corrs else 0.0
            lag_score = float(np.median(lag_corrs)) if lag_corrs else 0.0
            scores[feat] = {"diff_corr": diff_score, "lag_corr": lag_score}

    retained, dropped = [], []
    for feat in yearly_feature_cols:
        passes_diff = scores[feat]["diff_corr"] >= corr_threshold
        passes_lag = scores[feat]["lag_corr"] >= lag_threshold
        if passes_diff or passes_lag:
            retained.append(feat)
        else:
            dropped.append(feat)

    logger.info(
        "yearly_evaluation      retained=%d  dropped=%d  "
        "(corr_threshold=%.2f, lag_threshold=%.2f)",
        len(retained), len(dropped), corr_threshold, lag_threshold,
    )
    return _make_result(
        "yearly_evaluation",
        {"corr_threshold": corr_threshold, "lag_threshold": lag_threshold,
         "year_col": year_col, "sector_col": sector_col},
        yearly_feature_cols, retained, dropped, scores,
    )
