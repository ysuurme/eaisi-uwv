"""
m_gold_target_loader.py — Load per-sector target series for the three CV scripts.

Place at: src/utils/m_gold_target_loader.py

Provides two loader functions that return the same dict {sbi_code: pd.Series}:

* ``load_target_series_from_gold``  — primary loader.  Reads imputed target
  values directly from the gold table (``master_data_ml_preprocessed``).
  Identical inputs to what Pipeline trains on.  Use this for the main
  comparison.

* ``load_target_series_from_silver`` — sensitivity loader.  Queries silver
  and applies contiguous-tail extraction: sectors with internal NaN gaps
  (the 13 reorganized sectors with 2004-2007 or 2004-2005 gaps) keep only
  their longest unbroken stretch after the last gap.  No fabricated values.
  Use this for the sensitivity analysis to quantify how much the
  gold-imputation borrowed-value pattern affects univariate-method accuracy.

Both functions return the same dict shape, so callers can swap them with
a flag without restructuring downstream code.

Why this exists
---------------
The ML pipeline's gold loader runs ``impute_target_variable`` which
forward-fills NaN gaps grouped by ``BedrijfstakkenBranchesSBI2008`` — a
broader classification than the per-sector ``BedrijfskenmerkenSBI2008``
the CV scripts iterate.  Because branches contain multiple sectors,
ffill in gold borrows values from sibling sectors.  Reading from gold
guarantees pixel-perfect input parity with Pipeline; reading from silver
with contiguous-tail extraction gives each univariate method its best
chance on real data only.

Methodological caveat for the gold loader: the imputed values are not
real observations and do enter univariate methods' training, but this
affects all four methods identically.  The forecast window (2024-2025)
is far from the imputed region, so impact on h=4 forecasts is modest.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect as sa_inspect


# Columns we DON'T want as candidate sector OHE columns.
_NATIONAL_TOTAL_OHE = "BedrijfskenmerkenSBI2008_T001081"

# Gold table layout — change these if your gold schema differs.
_DEFAULT_GOLD_TABLE = "master_data_ml_preprocessed"
_OHE_PREFIX         = "BedrijfskenmerkenSBI2008_"


def load_target_series_from_gold(
    gold_db_path: "str | Path",
    target_column: str = "Ziekteverzuimpercentage_1",
    gold_table: str = _DEFAULT_GOLD_TABLE,
    min_history: int = 33,
    include_national_total: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Load per-sector target time series from the gold table.

    Parameters
    ----------
    gold_db_path : str or Path
        Path to the gold SQLite database (DIR_DB_GOLD).
    target_column : str
        Name of the target column in the gold table.
    gold_table : str
        Name of the gold preprocessed table.
    min_history : int
        Minimum number of quarterly observations required per sector.
        Sectors with fewer quarters after imputation are excluded.
    include_national_total : bool
        If True, includes the BedrijfskenmerkenSBI2008_T001081 (national total)
        as one of the series.  If False, restricts to specific-sector OHE columns.
    verbose : bool
        Print a per-sector load summary.

    Returns
    -------
    dict {sbi_code: pd.Series}
        Each Series is indexed by quarter-end timestamps (pd.DatetimeIndex,
        freq='QE') and contains the imputed target values.  The sector code
        is the OHE-stripped key (e.g. "301000" for "BedrijfskenmerkenSBI2008_301000").

    Raises
    ------
    FileNotFoundError
        If the gold DB path doesn't exist.
    ValueError
        If the gold table doesn't have any matching OHE sector columns or
        the target column is missing.
    """
    gold_db_path = Path(gold_db_path)
    if not gold_db_path.exists():
        raise FileNotFoundError(f"Gold DB not found: {gold_db_path}")

    engine = create_engine(f"sqlite:///{gold_db_path.as_posix()}")

    # --- Discover sector OHE columns from the gold schema ---
    insp = sa_inspect(engine)
    all_cols = [c["name"] for c in insp.get_columns(gold_table)]
    if target_column not in all_cols:
        raise ValueError(
            f"Target column '{target_column}' not in {gold_table}. "
            f"Available columns: {all_cols[:20]}..."
        )
    if "period_enddate" not in all_cols:
        raise ValueError(f"'period_enddate' not in {gold_table}")

    ohe_cols = [c for c in all_cols if c.startswith(_OHE_PREFIX)]
    if not include_national_total:
        ohe_cols = [c for c in ohe_cols if c != _NATIONAL_TOTAL_OHE]
    if not ohe_cols:
        raise ValueError(
            f"No OHE columns matching '{_OHE_PREFIX}*' found in {gold_table}"
        )

    # --- Single query for all needed columns ---
    select_cols = ["period_enddate", target_column] + ohe_cols
    quoted_cols = ", ".join(f'"{c}"' for c in select_cols)
    query = f'SELECT {quoted_cols} FROM "{gold_table}" ORDER BY period_enddate'
    df = pd.read_sql(query, engine)
    df["period_enddate"] = pd.to_datetime(df["period_enddate"])

    # --- Build one Series per sector ---
    series_dict = {}
    skipped_short = []
    skipped_empty = []
    for ohe_col in sorted(ohe_cols):
        sbi_code = ohe_col[len(_OHE_PREFIX):]
        # Rows where this OHE flag is 1 belong to this sector
        sector_rows = df[df[ohe_col] == 1]
        if sector_rows.empty:
            skipped_empty.append(sbi_code)
            continue

        # Aggregate to one value per (sector, quarter) — defensive against any
        # accidental duplicates from cross-classification joins
        s = (sector_rows.groupby("period_enddate")[target_column]
                        .first()
                        .sort_index())
        s.index = pd.DatetimeIndex(s.index, name="period_enddate")
        # Snap to quarter-end alignment
        s.index = s.index + pd.offsets.QuarterEnd(0)
        s = s[~s.index.duplicated(keep="last")]

        if len(s) < min_history:
            skipped_short.append((sbi_code, len(s)))
            continue

        # Sanity check: target should be numeric and have no NaN after gold's
        # impute_target_variable pass.  If NaN persists here, that's a real bug
        # in the gold pipeline — we surface it rather than mask it.
        if s.isna().any():
            n_nan = int(s.isna().sum())
            if verbose:
                print(f"  [{sbi_code}] WARNING: {n_nan} NaN values in gold target — "
                      f"gold imputation may have failed for this sector")

        series_dict[sbi_code] = s

    if verbose:
        print(f"  Loaded {len(series_dict)} sector series from {gold_table}")
        if skipped_empty:
            print(f"  Skipped {len(skipped_empty)} empty sectors (no rows with OHE=1): "
                  f"{skipped_empty[:5]}{'...' if len(skipped_empty) > 5 else ''}")
        if skipped_short:
            short_str = ", ".join(f"{s}({n})" for s, n in skipped_short[:5])
            print(f"  Skipped {len(skipped_short)} short sectors "
                  f"(< {min_history} quarters): {short_str}"
                  f"{'...' if len(skipped_short) > 5 else ''}")

    return series_dict


def load_target_series_from_silver(
    silver_db_path: "str | Path",
    silver_table: str = "80072ned_silver",
    sbi_column: str = "BedrijfskenmerkenSBI2008",
    period_column: str = "Perioden",
    target_column: str = "Ziekteverzuimpercentage_1",
    data_start_year: int = 2003,
    min_history: int = 33,
    verbose: bool = True,
) -> dict:
    """
    Load per-sector target series from silver with contiguous-tail extraction.

    Sectors with internal NaN gaps (e.g. SBI reorganizations 2004-2007) keep
    only their longest contiguous tail — the unbroken stretch from after the
    last NaN to the most recent observation.  No fabricated values enter any
    series.  This is the "real data only" complement to the gold loader, used
    for sensitivity analysis.

    Parameters
    ----------
    silver_db_path : str or Path
        Path to the silver SQLite database (DIR_DB_SILVER).
    silver_table : str
        Name of the silver table holding raw absenteeism observations.
    sbi_column : str
        Column with sector codes in silver (the granular per-sector dimension).
    period_column : str
        Column with CBS-format period codes ("2003KW01", etc).
    target_column : str
        Name of the target column in silver.
    data_start_year : int
        Drop any rows before this year (regime-shift filter; matches gold's
        DATA_START_YEAR convention).
    min_history : int
        Minimum number of quarterly observations required per sector after
        contiguous-tail extraction.
    verbose : bool
        Print a per-sector load summary including how many sectors were
        truncated by the contiguous-tail extraction.

    Returns
    -------
    dict {sbi_code: pd.Series}
        Same shape as ``load_target_series_from_gold``'s return.  Sectors
        without gaps get their full series; sectors with gaps get only their
        post-gap contiguous tail (71-79 quarters for the 13 reorganized
        sectors, vs 91 for the others).
    """
    silver_db_path = Path(silver_db_path)
    if not silver_db_path.exists():
        raise FileNotFoundError(f"Silver DB not found: {silver_db_path}")

    engine = create_engine(f"sqlite:///{silver_db_path.as_posix()}")

    # CBS-format period decoding: "2003KW01" → quarter-end date "2003-03-31"
    query = f"""
    SELECT
        "{sbi_column}" as sbi_code,
        DATE(
            printf('%s-%s-01', substr("{period_column}", 1, 4),
                CASE substr("{period_column}", 7, 2)
                    WHEN '01' THEN '01' WHEN '02' THEN '04'
                    WHEN '03' THEN '07' WHEN '04' THEN '10'
                END),
            '+3 months', '-1 day'
        ) AS period_enddate,
        CAST("{target_column}" AS REAL) as target_value
    FROM "{silver_table}"
    WHERE "{period_column}" NOT LIKE '%JJ%'
      AND substr("{period_column}", 1, 4) >= '{data_start_year}'
    ORDER BY sbi_code, period_enddate ASC
    """
    df = pd.read_sql(query, engine)
    df["period_enddate"] = pd.to_datetime(df["period_enddate"])
    df["target_value"]   = pd.to_numeric(df["target_value"], errors="coerce")
    df = df.dropna(subset=["target_value"])

    series_dict   = {}
    skipped_short = []
    n_truncated   = 0
    total_codes   = df["sbi_code"].nunique()

    for code in sorted(df["sbi_code"].dropna().unique()):
        s = (df[df["sbi_code"] == code]
             .sort_values("period_enddate")
             .drop_duplicates("period_enddate")
             .set_index("period_enddate")["target_value"])
        s.index = s.index + pd.offsets.QuarterEnd(0)
        s = s[~s.index.duplicated(keep="last")]

        # Reindex over the sector's observed range; introduces NaN where gaps exist
        full_range  = pd.date_range(s.index.min(), s.index.max(), freq="QE-DEC")
        s_reindexed = s.reindex(full_range)

        if s_reindexed.isna().any():
            # Contiguous-tail extraction: keep only what's after the last NaN
            last_nan_pos = int(np.where(s_reindexed.isna())[0].max())
            s_clean = s_reindexed.iloc[last_nan_pos + 1:]
            if s_clean.isna().any():
                # Defensive — should not happen after slicing past the last NaN
                continue
            n_truncated += 1
        else:
            s_clean = s_reindexed

        if len(s_clean) < min_history:
            skipped_short.append((str(code), len(s_clean)))
            continue

        series_dict[str(code)] = s_clean

    if verbose:
        print(f"  Loaded {len(series_dict)} sector series from {silver_table}")
        print(f"  Contiguous-tail truncation applied to {n_truncated} sector(s) with gaps")
        if skipped_short:
            short_str = ", ".join(f"{s}({n})" for s, n in skipped_short[:5])
            print(f"  Skipped {len(skipped_short)} short sectors "
                  f"(< {min_history} quarters): {short_str}"
                  f"{'...' if len(skipped_short) > 5 else ''}")

    return series_dict


if __name__ == "__main__":
    # CLI self-check
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.m_gold_target_loader "
              "<path/to/gold_data.db> [target_column]")
        sys.exit(1)
    target = sys.argv[2] if len(sys.argv) > 2 else "Ziekteverzuimpercentage_1"
    series_dict = load_target_series_from_gold(sys.argv[1], target_column=target)
    print(f"\nLoaded {len(series_dict)} sectors. First 5:")
    for sbi_code, ts in list(series_dict.items())[:5]:
        print(f"  {sbi_code}: {len(ts)} quarters, "
              f"{ts.index.min().date()} → {ts.index.max().date()}, "
              f"min={ts.min():.2f}, max={ts.max():.2f}")
