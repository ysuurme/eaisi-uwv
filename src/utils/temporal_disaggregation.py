"""
Temporal disaggregation utilities for converting yearly CBS data to quarterly.

Provides two methods for distributing annual values across quarters:

1. **Linear interpolation** — simple, no related indicator needed.
   Places annual values at year midpoints and interpolates linearly.
   Preserves annual means via additive adjustment.

2. **Denton-Cholette** — the official statistics standard (Eurostat, CBS).
   Uses a related quarterly indicator to shape the disaggregation.
   Preserves annual means exactly via constrained optimisation.

Both methods replace the naive "repeat 4x" expansion that creates step
functions with 75% zero variance under differencing.

References
----------
Denton, F.T. (1971). "Adjustment of Monthly or Quarterly Series to Annual
    Totals: An Approach Based on Quadratic Minimization."
    Journal of the American Statistical Association, 66(333), 99-102.

Cholette, P.A. (1984). "Adjusting Sub-Annual Series to Yearly Benchmarks."
    Survey Methodology, 10(1), 35-49.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("temporal_disagg")


# ═══════════════════════════════════════════════════════════════════════════
# LINEAR INTERPOLATION
# ═══════════════════════════════════════════════════════════════════════════

def interpolate_linear(
    annual_values: np.ndarray,
    annual_years: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disaggregate annual values to quarterly via linear interpolation.

    Places each annual value at the midpoint of its year (July 1) and
    interpolates linearly to quarterly midpoints.  Applies additive
    adjustment per year to ensure the quarterly mean exactly equals
    the original annual value.

    Parameters
    ----------
    annual_values : np.ndarray, shape (n_years,)
        Annual values to disaggregate.
    annual_years : np.ndarray, shape (n_years,)
        Corresponding calendar years (e.g. [2003, 2004, ...]).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(quarterly_values, quarterly_years, quarterly_quarters)``
        where quarterly_quarters is 1-4.
    """
    annual_values = np.asarray(annual_values, dtype=float)
    annual_years = np.asarray(annual_years, dtype=int)

    if len(annual_values) != len(annual_years):
        raise ValueError("annual_values and annual_years must have the same length")

    # Annual midpoints (July 1 ≈ year + 0.5)
    annual_t = annual_years.astype(float) + 0.5

    # Quarterly midpoints: year + (q - 0.5) / 4
    q_years_list = []
    q_quarters_list = []
    q_t_list = []
    for year in range(int(annual_years.min()), int(annual_years.max()) + 1):
        for q in range(1, 5):
            q_years_list.append(year)
            q_quarters_list.append(q)
            q_t_list.append(year + (q - 0.5) / 4)

    q_t = np.array(q_t_list)
    q_values = np.interp(q_t, annual_t, annual_values)
    q_years = np.array(q_years_list)
    q_quarters = np.array(q_quarters_list)

    # Additive adjustment: ensure quarterly mean = original annual value
    for i, year in enumerate(annual_years):
        mask = q_years == year
        if mask.any():
            mean_q = q_values[mask].mean()
            q_values[mask] += (annual_values[i] - mean_q)

    return q_values, q_years, q_quarters


# ═══════════════════════════════════════════════════════════════════════════
# DENTON-CHOLETTE
# ═══════════════════════════════════════════════════════════════════════════

def denton_cholette(
    annual_values: np.ndarray,
    indicator_quarterly: np.ndarray,
    constraint: str = "mean",
) -> np.ndarray:
    """Denton-Cholette temporal disaggregation.

    Distributes annual values to quarterly frequency using a related
    quarterly indicator to guide the shape.  Solves the constrained
    quadratic minimisation:

        min  || D(q - p) ||²
        s.t.  C q = Y

    where ``q`` is the disaggregated quarterly series, ``p`` is the
    preliminary indicator, ``D`` is the first-difference operator,
    ``C`` is the annual aggregation matrix, and ``Y`` is the vector
    of annual values.

    Parameters
    ----------
    annual_values : np.ndarray, shape (n_years,)
        Annual values to disaggregate.
    indicator_quarterly : np.ndarray, shape (n_years * 4,)
        Quarterly indicator series.  Must have exactly ``4 * n_years``
        values, aligned so that indices 0-3 correspond to Q1-Q4 of
        the first year.
    constraint : str, default ``"mean"``
        ``"mean"`` — quarterly values must average to the annual value.
        ``"sum"``  — quarterly values must sum to the annual value.

    Returns
    -------
    np.ndarray, shape (n_years * 4,)
        Disaggregated quarterly values.
    """
    annual_values = np.asarray(annual_values, dtype=float)
    indicator_quarterly = np.asarray(indicator_quarterly, dtype=float)

    n_years = len(annual_values)
    n_q = 4
    n_total = n_years * n_q

    if len(indicator_quarterly) != n_total:
        raise ValueError(
            f"indicator_quarterly length ({len(indicator_quarterly)}) must be "
            f"4 * n_years ({n_total})"
        )

    p = indicator_quarterly
    Y = annual_values

    # First-difference matrix D: (n_total - 1) x n_total
    D = np.zeros((n_total - 1, n_total))
    for i in range(n_total - 1):
        D[i, i] = -1
        D[i, i + 1] = 1

    # Aggregation matrix C: n_years x n_total
    C = np.zeros((n_years, n_total))
    for i in range(n_years):
        start = i * n_q
        end = start + n_q
        if constraint == "sum":
            C[i, start:end] = 1.0
        else:
            C[i, start:end] = 1.0 / n_q

    # KKT system
    DtD = D.T @ D
    n_sys = n_total + n_years
    KKT = np.zeros((n_sys, n_sys))
    KKT[:n_total, :n_total] = DtD
    KKT[:n_total, n_total:] = C.T
    KKT[n_total:, :n_total] = C

    rhs = np.zeros(n_sys)
    rhs[:n_total] = DtD @ p
    rhs[n_total:] = Y

    try:
        solution = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Denton-Cholette KKT system is singular: {e}")

    return solution[:n_total]


# ═══════════════════════════════════════════════════════════════════════════
# QUARTERLY INDICATOR SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def find_best_quarterly_indicator(
    annual_values: pd.Series,
    quarterly_df: pd.DataFrame,
    quarterly_features: list[str],
    year_col: str = "year",
    min_correlation: float = 0.30,
) -> tuple[str | None, float]:
    """Identify the best quarterly proxy for a yearly feature.

    Aggregates each quarterly feature to annual means, then correlates
    with the yearly feature.  Returns the quarterly feature with the
    highest absolute correlation.

    Parameters
    ----------
    annual_values : pd.Series
        Annual values indexed by year.
    quarterly_df : pd.DataFrame
        Full quarterly panel dataset.
    quarterly_features : list[str]
        Candidate quarterly feature column names.
    year_col : str, default ``"year"``
        Year column in ``quarterly_df``.
    min_correlation : float, default 0.30
        Minimum |r| to accept.

    Returns
    -------
    tuple[str | None, float]
        ``(best_feature_name, correlation)``.  ``None`` if no suitable
        indicator found.
    """
    if not quarterly_features:
        return None, 0.0

    annual_q = quarterly_df.groupby(year_col)[quarterly_features].mean()
    common_years = annual_values.index.intersection(annual_q.index)

    if len(common_years) < 5:
        return None, 0.0

    best_col: str | None = None
    best_corr: float = 0.0

    y_aligned = annual_values.loc[common_years].astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        for col in quarterly_features:
            if col not in annual_q.columns:
                continue
            q_aligned = annual_q.loc[common_years, col].astype(float)
            mask = y_aligned.notna() & q_aligned.notna()
            if mask.sum() < 5:
                continue
            r = y_aligned[mask].corr(q_aligned[mask])
            if not np.isnan(r) and abs(r) > abs(best_corr):
                best_corr = r
                best_col = col

    if abs(best_corr) < min_correlation:
        return None, best_corr

    logger.info("Best quarterly indicator: %s (r=%.3f)", best_col, best_corr)
    return best_col, best_corr


# ═══════════════════════════════════════════════════════════════════════════
# MASTER TABLE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def smooth_yearly_features(
    df: pd.DataFrame,
    year_col: str = "year",
    quarter_col: str = "quarter",
    yearly_prefix: str = "y_",
    method: str = "linear",
    quarterly_features: list[str] | None = None,
    target_col: str | None = None,
    min_indicator_corr: float = 0.30,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Replace step-function yearly features with smoothed versions in-place.

    Operates on the master table after the merge.  Detects yearly-origin
    columns by their ``y_`` prefix, extracts the underlying annual values
    from the step function (all 4 quarters have the same value), and
    replaces them with interpolated or Denton-disaggregated values.

    Parameters
    ----------
    df : pd.DataFrame
        The master panel dataset with step-function yearly columns.
    year_col : str, default ``"year"``
        Year column.
    quarter_col : str, default ``"quarter"``
        Quarter column (1-4).
    yearly_prefix : str, default ``"y_"``
        Prefix identifying yearly-origin columns.
    method : str, default ``"linear"``
        ``"linear"`` — linear interpolation for all yearly columns.
        ``"denton"`` — try Denton-Cholette, fall back to linear.
        ``"auto"``   — Denton if good indicator found, else linear.
    quarterly_features : list[str] or None
        Candidate quarterly features for Denton indicator selection.
        If None and method != "linear", auto-detects non-y_ numeric columns.
    target_col : str or None
        Target column name to EXCLUDE from candidate indicators.
        Using the target as a Denton indicator is circular (data leakage).
    min_indicator_corr : float, default 0.30
        Minimum |r| for Denton indicator selection.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(smoothed_df, report)`` where ``report`` maps each yearly
        column to its disaggregation metadata (method used, indicator, r).
    """
    df = df.copy()

    if year_col not in df.columns or quarter_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{year_col}' and '{quarter_col}' columns")

    yearly_cols = [c for c in df.columns if c.startswith(yearly_prefix) and pd.api.types.is_numeric_dtype(df[c])]

    if not yearly_cols:
        logger.info("No yearly columns (prefix='%s') found. Nothing to smooth.", yearly_prefix)
        return df, {}

    # Auto-detect quarterly features for Denton (exclude target to prevent leakage)
    exclude_from_indicators = {year_col, quarter_col}
    if target_col:
        exclude_from_indicators.add(target_col)

    if quarterly_features is None and method != "linear":
        quarterly_features = [
            c for c in df.columns
            if not c.startswith(yearly_prefix)
            and pd.api.types.is_numeric_dtype(df[c])
            and c not in exclude_from_indicators
        ]

    report: dict[str, dict[str, Any]] = {}

    for col in yearly_cols:
        # Detect broadcast vs sector-specific:
        # broadcast features have zero variance within each (year, quarter) group
        # Use lambda with nanstd to handle NaN values (pre-imputation or partial coverage)
        with np.errstate(divide="ignore", invalid="ignore"):
            group_std = df.groupby([year_col, quarter_col])[col].apply(
                lambda x: np.nanstd(x.dropna().values, ddof=0) if x.notna().any() else 0.0
            )
        is_broadcast = (group_std.fillna(0.0) < 1e-10).all()

        if is_broadcast:
            # All sectors have the same value — extract one annual series
            annual = df.groupby(year_col)[col].first().dropna()
        else:
            # Sector-specific yearly feature — use the mean across sectors
            annual = df.groupby(year_col)[col].mean().dropna()
            logger.warning(
                "Column '%s' varies across sectors within year. "
                "Using cross-sector mean for disaggregation.", col,
            )

        if len(annual) < 2:
            report[col] = {"method_used": "skip", "reason": "< 2 annual values"}
            continue

        years = annual.index.values.astype(int)
        values = annual.values.astype(float)

        # Choose method
        used_method = "linear"
        indicator_col = None
        indicator_corr = 0.0

        if method in ("denton", "auto") and quarterly_features:
            indicator_col, indicator_corr = find_best_quarterly_indicator(
                annual, df, quarterly_features,
                year_col=year_col, min_correlation=min_indicator_corr,
            )

        if indicator_col is not None:
            # Extract indicator's quarterly pattern (mean across sectors per quarter)
            indicator_q = (
                df.groupby([year_col, quarter_col])[indicator_col]
                .mean()
                .reset_index()
                .sort_values([year_col, quarter_col])
            )
            indicator_annual_q = []
            usable = True
            for yr in years:
                yr_q = indicator_q.loc[indicator_q[year_col] == yr, indicator_col]
                if len(yr_q) == 4:
                    indicator_annual_q.extend(yr_q.values)
                else:
                    usable = False
                    break

            if usable:
                try:
                    q_vals = denton_cholette(values, np.array(indicator_annual_q), constraint="mean")
                    used_method = "denton"
                except ValueError:
                    q_vals, q_years, q_quarters = interpolate_linear(values, years)
                    used_method = "linear (denton fallback)"
            else:
                q_vals, q_years, q_quarters = interpolate_linear(values, years)
                used_method = "linear (incomplete quarters)"
        else:
            q_vals, q_years, q_quarters = interpolate_linear(values, years)

        # Map interpolated values back to the DataFrame
        if used_method.startswith("denton"):
            # Denton output is a flat array aligned to years × 4 quarters
            for i, val in enumerate(q_vals):
                yr = years[i // 4]
                q = (i % 4) + 1
                mask = (df[year_col] == yr) & (df[quarter_col] == q)
                df.loc[mask, col] = val
        else:
            # Linear output has explicit year/quarter arrays
            for val, yr, q in zip(q_vals, q_years, q_quarters):
                mask = (df[year_col] == yr) & (df[quarter_col] == q)
                df.loc[mask, col] = val

        report[col] = {
            "method_used": used_method,
            "indicator": indicator_col,
            "indicator_corr": indicator_corr,
        }

    n_linear = sum(1 for r in report.values() if "linear" in str(r.get("method_used", "")))
    n_denton = sum(1 for r in report.values() if r.get("method_used") == "denton")
    n_skip = sum(1 for r in report.values() if r.get("method_used") == "skip")
    logger.info(
        "smooth_yearly_features: %d columns — %d linear, %d denton, %d skipped",
        len(yearly_cols), n_linear, n_denton, n_skip,
    )

    return df, report
