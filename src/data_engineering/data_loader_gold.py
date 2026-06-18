"""
Data Loader for the Gold Layer.
Formats tables from silver into clean numerical timeseries features ready for ML ingestion.
"""
from pathlib import Path

# --- Third Party Libraries ---
import pandas as pd
from sqlalchemy import MetaData, create_engine, text

# --- Configuration ---
from src.config import DIR_DB_SILVER, DIR_DB_GOLD, DIR_FEATURE_SELECTION, ML_TARGET_COLUMN, CBS_TABLES_TO_LOAD, CBS_TABLES_YEARLY, DATA_START_YEAR
try:
    from src.config import CBS_TABLES_MONTHLY
except ImportError:
    # Backwards-compatible default — keeps the loader runnable for users
    # who haven't yet added CBS_TABLES_MONTHLY to their config.py.
    CBS_TABLES_MONTHLY: dict = {}

# --- Logging ---
from src.utils.m_log import f_log


class DatabaseGold:
    """
    Manages the SQLite database in the gold layer.
    Formats tables from silver into clean features for analysis and machine learning into a single wide table. 
    """
    def __init__(self, db_silver_path: Path, db_gold_path: Path):
        self.db_silver_path = Path(db_silver_path)
        self.db_gold_path = Path(db_gold_path)
        self.db_gold_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_gold_path.as_posix()}")
        self.metadata = MetaData()


    def process_silver_table(self, identifier: str, transformation_func: callable):
        """Fetches a table from Silver DB, applies transformation, and stores in Gold DB."""
        silver_table_name = f"{identifier}_silver"
        gold_table_name = f"{identifier}_gold"

        f_log(f"Processing {silver_table_name} -> {gold_table_name}", c_type="process")

        with self.engine.connect() as conn:
            conn.execute(text("ATTACH DATABASE :silver_path AS silver"), {"silver_path": str(self.db_silver_path)})
            
            try:
                query = f"SELECT * FROM silver.\"{silver_table_name}\""
                df = pd.read_sql_query(query, conn)
            except Exception as e:
                f_log(f"Error fetching {silver_table_name}: {e}", c_type="error")
                return
            finally:
                conn.execute(text("DETACH DATABASE silver"))

        try:
            df_gold = transformation_func(df)
            
            # --- CENTRALISED GOLD QUALITY GATE ---
            # Enforce zero-nulls and float64 type casting universally
            numeric_cols = df_gold.select_dtypes(include=['number']).columns
            df_gold[numeric_cols] = df_gold[numeric_cols].astype('float64')

            f_log(f"Gold Quality Gate: Processed {len(df_gold)} rows. 0 NaNs remain.", c_type="success")
            
            # Persist to Gold securely inside the explicit try-block to catch SQLite layout boundaries cleanly
            df_gold.to_sql(gold_table_name, self.engine, if_exists='replace', index=False)
            f_log(f"Stored {gold_table_name}", c_type="store")
            
        except Exception as e:
            f_log(f"Error transforming/storing {silver_table_name}: {e}", c_type="error")
            return

    def create_master_training_dataset(self, target_prefix: str = "80072ned") -> pd.DataFrame:
        """
        Synthesizes the Gold Layer into a unified SBI-granular Master Training Matrix.

        Tiered Join Strategy (Broadcast Dimension Pattern):
        - Tier 1 (SBI-specific): Feature has 'BedrijfstakkenBranchesSBI2008'
                                 -> join on [period_enddate, SBI_COL]. Enables branch-level predictions.
        - Tier 2 (Broadcast):    Feature lacks 'BedrijfstakkenBranchesSBI2008'
                                 -> join on [period_enddate] only. The single national value is
                                    broadcast (replicated) across all SBI branch rows for that quarter.
                                    Assumption: national-level features (e.g. stress=1.0) apply uniformly
                                    to every sector until SBI-specific data becomes available.

        The full SBI-granular master table is preserved (~4393 rows). The raw SBI_COL is reconstructed
        from its OHE representation in the target table to re-enable Tier 1 composite joins.
        """
        ALL_SECTORS_KEY = "T001081"
        SBI_COL = "BedrijfstakkenBranchesSBI2008"
        DATE_COL = "period_enddate"
        target_table_name = f"{target_prefix}_gold"

        f_log(f"Starting master dataset creation with target: {target_table_name}", c_type="process")

        with self.engine.connect() as conn:
            # 1. Load full SBI-granular target DataFrame
            try:
                query = f"SELECT * FROM \"{target_table_name}\""
                master_df = pd.read_sql_query(query, conn)
            except Exception as e:
                f_log(f"Failed to load target table {target_table_name}: {e}", c_type="error")
                return pd.DataFrame()

            if master_df.empty:
                f_log("Master target dataframe is empty. Aborting master integration.", c_type="error")
                return master_df

            # 2. Identify and Load all Feature Tables
            try:
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
            except Exception as e:
                f_log(f"Failed reading sqlite_master: {e}", c_type="error")
                return master_df

            excluded = {target_table_name}
            active_gold_tables = (
                {f"{tid}_gold" for tid in CBS_TABLES_TO_LOAD}
                | {f"{tid}_gold" for tid in CBS_TABLES_YEARLY}
                | {f"{tid}_gold" for tid in CBS_TABLES_MONTHLY}
            )
            feature_table_names = [
                t for t in tables['name'].tolist()
                if t.endswith("_gold") and t not in excluded and t in active_gold_tables
            ]

            feature_dfs = {}
            for table in feature_table_names:
                try:
                    feature_dfs[table] = pd.read_sql_query(f"SELECT * FROM \"{table}\"", conn)
                except Exception as e:
                    f_log(f"Error loading feature table {table}: {e}", c_type="error")

        # 3. Tiered Feature Merge using Pure Function
        master_df, sbi_joined, broadcast_joined, column_origin = synthesize_master_features(master_df, feature_dfs, ML_TARGET_COLUMN)

        # Join Summary Report
        f_log("--- Master Join Summary ---", c_type="info")
        f_log(f"Tier 1 (SBI-specific, {len(sbi_joined)} tables): {sbi_joined or 'None'}", c_type="info")
        f_log(f"Tier 2 (Broadcast/no SBI, {len(broadcast_joined)} tables — ADD SBI DIMENSION): {broadcast_joined or 'None'}", c_type="warning")

        # Save column-origin mapping (enables preset-aware feature filtering)
        import json
        origin_path = Path(DIR_FEATURE_SELECTION) / "column_origin.json"
        origin_path.parent.mkdir(parents=True, exist_ok=True)
        with open(origin_path, "w") as f:
            json.dump(column_origin, f, indent=2, sort_keys=True)
        f_log(f"Column-origin mapping: {len(column_origin)} features → {origin_path}", c_type="store")

        # 4. Storage Persistence (joined, pre-imputation)
        try:
            master_df.to_sql("master_data_ml_joined", self.engine, if_exists="replace", index=False)
            f_log("Stored master_data_ml_joined successfully.", c_type="store")
        except Exception as e:
            f_log(f"Failed to save master_data_ml_joined: {e}", c_type="error")

        f_log(f"Completed master dataset join. Shape: {master_df.shape}", c_type="complete")

        # 5. Preprocessing Gate: validate → impute → validate → persist
        from src.ml_engineering.ml_2_data_validation import DataValidator
        from src.utils.m_imputation import impute_missing_values

        DataValidator.validate(master_df, target_column=ML_TARGET_COLUMN, stage="pre_prep")
        preprocessed_df = impute_missing_values(master_df)

        # 5b. Temporal disaggregation: replace yearly step functions with smooth interpolation
        # Runs AFTER imputation so quarterly indicators have no NaN (needed for Denton correlation)
        from src.utils.temporal_disaggregation import smooth_yearly_features
        preprocessed_df, disagg_report = smooth_yearly_features(
            preprocessed_df, method="auto", target_col=ML_TARGET_COLUMN,
        )
        n_denton = sum(1 for r in disagg_report.values() if r.get("method_used") == "denton")
        n_linear = sum(1 for r in disagg_report.values() if "linear" in str(r.get("method_used", "")))
        n_total = n_denton + n_linear
        if n_total > 0:
            f_log(f"Temporal disaggregation: {n_denton} Denton + {n_linear} linear fallback ({n_total} total)", c_type="process")

        DataValidator.validate(preprocessed_df, target_column=ML_TARGET_COLUMN, stage="post_prep")

        try:
            preprocessed_df.to_sql("master_data_ml_preprocessed", self.engine, if_exists="replace", index=False)
            f_log("Stored master_data_ml_preprocessed successfully.", c_type="store")
        except Exception as e:
            f_log(f"Failed to save master_data_ml_preprocessed: {e}", c_type="error")

        f_log(f"Completed full preprocessing pipeline. Final shape: {preprocessed_df.shape}", c_type="complete")
        return master_df


def synthesize_master_features(
    master_df: pd.DataFrame,
    feature_dfs: dict[str, pd.DataFrame],
    target_col: str
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Synthesizes the Gold Layer into a unified SBI-granular Master Training Matrix.
    Pure function avoiding side-effects.
    
    Args:
        master_df: The foundational target DataFrame.
        feature_dfs: Dictionary of feature DataFrames keyed by table name.
        target_col: The target column name for baseline application.

    Returns:
        Tuple of (master_df, sbi_joined_list, broadcast_joined_list, column_origin_dict)
    """
    master_df = master_df.copy()
    SBI_COL = "BedrijfstakkenBranchesSBI2008"
    DATE_COL = "period_enddate"

    if master_df.empty:
        return master_df, [], []

    # Ensure consistent nomenclature for the target table
    master_df = apply_gold_baseline(master_df, target_col)

    # Reconstruct raw SBI_COL from OHE columns so Tier 1 joins are possible.
    sbi_ohe_prefixes = [
        f"{SBI_COL}_",
        "BedrijfstakkenSBI2008_",
        "Bedrijfstakken_SBI2008_",
        "SBI2008_",
    ]
    ohe_prefix = None
    ohe_cols = []
    for candidate_prefix in sbi_ohe_prefixes:
        ohe_cols = [c for c in master_df.columns if c.startswith(candidate_prefix)]
        if ohe_cols:
            ohe_prefix = candidate_prefix
            break

    if ohe_cols and SBI_COL not in master_df.columns:
        master_df[SBI_COL] = master_df[ohe_cols].idxmax(axis=1).str.replace(ohe_prefix, "", regex=False)

    sbi_joined: list[str] = []
    broadcast_joined: list[str] = []
    column_origin: dict[str, str] = {}  # {column_name: table_id}

    for table, feature_df in feature_dfs.items():
        feature_df = feature_df.copy()
        cols_before = set(master_df.columns)

        if DATE_COL not in feature_df.columns:
            continue

        # Ensure datetime type (string after SQLite round-trip)
        if not pd.api.types.is_datetime64_any_dtype(feature_df[DATE_COL]):
            feature_df[DATE_COL] = pd.to_datetime(feature_df[DATE_COL], errors='coerce')

        has_sbi = SBI_COL in feature_df.columns and SBI_COL in master_df.columns
        active_keys = [DATE_COL, SBI_COL] if has_sbi else [DATE_COL]

        if not has_sbi and feature_df.duplicated(subset=[DATE_COL]).any():
            numeric_feature_cols = feature_df.select_dtypes(include="number").columns.tolist()
            feature_df = feature_df.groupby(DATE_COL, as_index=False)[numeric_feature_cols].mean()

        overlap = set(feature_df.columns).intersection(set(master_df.columns)) - set(active_keys)
        if overlap:
            overlap_list = sorted(list(overlap))
            feature_df = feature_df.drop(columns=overlap_list)

        master_df = pd.merge(master_df, feature_df, on=active_keys, how="left")

        # Track which new columns came from this table
        table_id = table.replace("_gold", "")
        for col in set(master_df.columns) - cols_before:
            column_origin[col] = table_id

        if has_sbi:
            sbi_joined.append(table)
        else:
            broadcast_joined.append(table)

    if "silver_id" in master_df.columns:
        master_df = master_df.drop(columns=["silver_id"])

    return master_df, sbi_joined, broadcast_joined, column_origin


def apply_gold_baseline(
    df: pd.DataFrame,
    ml_target_col: str = None,
    lag_years: int = 0,
    monthly_aggregation: str = "mean",
) -> pd.DataFrame:
    """
    Applies the foundational structural baseline logic for Gold datasets.
    Handles temporal filtering, yearly extrapolation, monthly aggregation,
    explicit pruning and zero-filling.

    Parameters
    ----------
    lag_years : int, default 0
        For purely yearly tables, shift the year forward by this many years
        when expanding to quarterly.  Use 1 to avoid look-ahead bias
        (2022 annual value → Q1-Q4 2023).
    monthly_aggregation : {"mean", "sum", "last", "first"}, default "mean"
        How to collapse monthly observations within a quarter.  Choose per
        variable semantics:
          - "mean" : level / index variables (CCI, CPI, unemployment rate)
          - "sum"  : flow variables (new vacancies opened in period)
          - "last" : end-of-period stock variables (vacancies open at quarter close)
          - "first": rarely useful — provided for symmetry
        Only applied to tables that are purely monthly (have MM, no KW).
    """
    df = df.copy()

    # 1. Temporal Standardization & Extrapolation
    if 'Perioden' in df.columns:
        # Defensive: cast to string in case SQLite round-trip changed the type
        # (e.g. when a column is entirely numeric-looking, pandas may infer int).
        periods_str = df['Perioden'].astype(str)
        has_kw = periods_str.str.contains('KW', na=False).any()
        has_jj = periods_str.str.contains('JJ', na=False).any()
        has_mm = periods_str.str.contains('MM', na=False).any()
        n_in = len(df)

        # Priority order: KW > MM > JJ.  This matters because CBS tables routinely
        # publish multiple frequencies in one source (e.g. monthly observations +
        # annual summaries).  Always prefer the highest granularity available;
        # only fall back to yearly expansion when nothing finer exists.
        if has_kw:
            # Pure quarterly (or KW+JJ / KW+MM mix) — keep only KW rows.
            df = df[df['Perioden'].astype(str).str.contains('KW', na=False)].copy()

        elif has_mm:
            # Monthly → quarterly: aggregate the (up to) 3 monthly rows per quarter.
            # CBS encoding: positions 0-3 = YYYY, 4-5 = 'MM', 6-7 = month number.
            f_log(
                f"Aggregating monthly data to quarters "
                f"(method={monthly_aggregation}"
                f"{', lag='+str(lag_years)+'y' if lag_years else ''}, "
                f"{n_in} input rows).",
                c_type="warning",
            )
            mm_df = df[df['Perioden'].astype(str).str.contains('MM', na=False)].copy()
            mm_df['_year']  = mm_df['Perioden'].astype(str).str[:4].astype(int)
            mm_df['_month'] = mm_df['Perioden'].astype(str).str[6:8].astype(int)

            # Drop annual-summary rows encoded as MM00 and any out-of-range months.
            # Without this, _quarter would be 0 or >4 → unparseable "KW00".
            n_before_filter = len(mm_df)
            mm_df = mm_df[(mm_df['_month'] >= 1) & (mm_df['_month'] <= 12)].copy()
            n_dropped = n_before_filter - len(mm_df)
            if n_dropped > 0:
                f_log(
                    f"Monthly aggregation: dropped {n_dropped} rows with month "
                    f"out of [1,12] (likely annual summaries encoded as MMxx).",
                    c_type="process",
                )
            mm_df['_quarter'] = ((mm_df['_month'] - 1) // 3 + 1).astype(int)

            # Group keys: temporal bucket + any SBI variant present.  SBI is
            # preserved so per-sector rows stay distinct after aggregation.
            sbi_variants = [
                'BedrijfstakkenBranchesSBI2008', 'BedrijfstakkenSBI2008',
                'Bedrijfstakken_SBI2008', 'SBI2008',
                'BedrijfstakkenBranches_SBI2008',
            ]
            sbi_in_df = [c for c in sbi_variants if c in mm_df.columns]
            group_keys = ['_year', '_quarter'] + sbi_in_df

            # Defensive numeric coercion.  Silver stores everything as
            # SQLAlchemy String — usually SQLite type affinity preserves the
            # underlying float/int values so pandas reads them as numeric,
            # but CBS sometimes encodes missing values as "." or "       ."
            # which would force a string column.  Coerce so groupby.mean()
            # has something to operate on.
            structural_cols = set(group_keys) | {
                'Perioden', '_month',
                'silver_id', 'bronze_pk', '_source_file', 'ID',
            } | set(sbi_variants)
            feature_candidates = [
                c for c in mm_df.columns
                if c not in structural_cols
                and not pd.api.types.is_numeric_dtype(mm_df[c])
            ]
            for c in feature_candidates:
                mm_df[c] = pd.to_numeric(mm_df[c], errors='coerce')

            numeric_cols = [
                c for c in mm_df.select_dtypes(include='number').columns
                if c not in ('_year', '_month', '_quarter')
            ]
            if not numeric_cols:
                f_log(
                    "Monthly aggregation: no numeric feature columns to aggregate — "
                    "table will produce only structural keys (this is the bug if "
                    "the source table is supposed to have data columns).",
                    c_type="warning",
                )

            agg_methods = {"mean", "sum", "last", "first"}
            if monthly_aggregation not in agg_methods:
                raise ValueError(
                    f"monthly_aggregation must be one of {agg_methods}, "
                    f"got {monthly_aggregation!r}"
                )

            if monthly_aggregation in ("last", "first"):
                # Order by _month so first/last pick the chronologically right month.
                mm_df = mm_df.sort_values('_month')
            agg = (
                mm_df.groupby(group_keys, as_index=False)[numeric_cols]
                     .agg(monthly_aggregation)
            )

            # Reconstruct Perioden in CBS KW format ("YYYYKWqq"), applying lag.
            target_year = agg['_year'] + lag_years if lag_years else agg['_year']
            agg['Perioden'] = (
                target_year.astype(str)
                + 'KW'
                + agg['_quarter'].apply(lambda q: f"{q:02d}")
            )
            df = agg.drop(columns=['_year', '_quarter'])
            f_log(
                f"Monthly aggregation complete: {n_in} rows → {len(df)} quarterly "
                f"rows × {len(numeric_cols)} feature columns.",
                c_type="success",
            )

        elif has_jj:
            # Yearly → quarterly: replicate each yearly row 4× (one per quarter).
            # Reached only when no MM and no KW are present.
            lag_label = f", lag={lag_years}y" if lag_years else ""
            f_log(f"Expanding yearly data to quarters (values repeated 4x{lag_label}).", c_type="warning")
            df_list = []
            for quarter in ['KW01', 'KW02', 'KW03', 'KW04']:
                temp = df[df['Perioden'].astype(str).str.contains('JJ', na=False)].copy()
                if lag_years:
                    # Shift year forward: 2022JJ00 with lag=1 → 2023KWxx
                    temp['Perioden'] = temp['Perioden'].apply(
                        lambda p: str(int(str(p)[:4]) + lag_years) + quarter
                    )
                else:
                    temp['Perioden'] = temp['Perioden'].astype(str).str.replace(r'JJ.*', quarter, regex=True)
                df_list.append(temp)
            df = pd.concat(df_list, ignore_index=True)

        else:
            # No recognizable CBS period encoding — surface this loudly so it
            # isn't mistaken for an empty source.
            f_log(
                f"Perioden column present but none of KW/JJ/MM detected "
                f"(sample: {df['Perioden'].head(3).tolist()}). "
                f"No temporal transformation applied.",
                c_type="warning",
            )

        # Diagnostic for the silent-zero case
        if len(df) == 0 and n_in > 0:
            f_log(
                f"Period transformation produced 0 rows from {n_in} input rows. "
                f"Likely cause: branch selection filtered everything out.",
                c_type="error",
            )

        # Datetime Parsing
        def parse_quarter(val):
            if pd.isna(val): return pd.NaT
            parts = str(val).split('KW')
            if len(parts) == 2:
                try:
                    return pd.Timestamp(year=int(parts[0]), month=int(parts[1])*3, day=1) + pd.offsets.MonthEnd(0)
                except Exception:
                    return pd.NaT
            return pd.NaT

        df['period_enddate'] = df['Perioden'].apply(parse_quarter)



    # 3. Column Pruning
    meta_cols = ["silver_id", "bronze_pk", "_source_file", "ID", "Perioden"]
    descriptive_suffixes = ('_Description', '_Title', '_Status', '_CategoryGroupID')
    
    cols_to_drop = [c for c in meta_cols if c in df.columns]
    cols_to_drop += [c for c in df.columns if c.endswith(descriptive_suffixes)]
    
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # 4. Normalize SBI Column Nomenclature
    # Standardizes various CBS sector column names into a project-wide structural key consistent for joins.
    sbi_variants = ['BedrijfstakkenSBI2008', 'Bedrijfstakken_SBI2008', 'SBI2008', 'BedrijfstakkenBranches_SBI2008']
    target_sbi = 'BedrijfstakkenBranchesSBI2008'
    
    if target_sbi not in df.columns:
        for variant in sbi_variants:
            if variant in df.columns:
                df = df.rename(columns={variant: target_sbi})
                f_log(f"Normalized structural key '{variant}' -> '{target_sbi}'", c_type="process")
                break

    # 5. Drop NaNs on specific crucial identifier 
    if 'period_enddate' in df.columns:
        # Ensure datetime type (may be string after SQLite round-trip)
        if not pd.api.types.is_datetime64_any_dtype(df['period_enddate']):
            df['period_enddate'] = pd.to_datetime(df['period_enddate'], errors='coerce')
        df = df.dropna(subset=['period_enddate'])

        # 6. Structural break filter: exclude data before DATA_START_YEAR
        #    (WIA law 2003 caused a regime shift in Dutch absenteeism data)
        if DATA_START_YEAR:
            before = len(df)
            df = df[df['period_enddate'].dt.year >= DATA_START_YEAR].copy()
            n_dropped = before - len(df)
            if n_dropped > 0:
                f_log(f"Temporal filter: dropped {n_dropped} rows before {DATA_START_YEAR}", c_type="process")

    return df


def transform_generic_feature_table(
    df: pd.DataFrame,
    lag_years: int = 0,
    filters: dict = None,
    exclude_metrics: list = None,
    keep_metrics: list = None,
    monthly_aggregation: str = "mean",
) -> pd.DataFrame:
    """
    Transforms observational feature datasets dynamically into ML features.
    Applies strict pivoting by translating granular rows across all available dimensions into distinct column features.

    Parameters
    ----------
    lag_years : int
        Year shift for yearly tables (0 = no shift, 1 = last year's value).
    filters : dict or None
        Optional ``{column: value}`` row filters applied before pivoting.
        Use to select a specific CBS aggregation level (e.g. ``{"Marges": "MW00000"}``).
    exclude_metrics : list[str] or None
        Regex patterns for metric columns to **exclude** before pivoting.
        Matching uses ``re.search`` (case-insensitive, matches anywhere in
        column name).  Example: ``["Mannen", "Vrouwen"]`` excludes any
        column containing those strings.
    keep_metrics : list[str] or None
        Regex patterns for metric columns to **keep**.  Only columns matching
        at least one pattern are retained; all others are dropped.  Matching
        uses ``re.match`` (anchored to the start of the column name).
        Applied before ``exclude_metrics``.
        Example: ``["Score"]`` keeps only columns starting with "Score".
    monthly_aggregation : {"mean", "sum", "last", "first"}, default "mean"
        How to collapse monthly observations into quarters when the source
        table is purely monthly (CBS ``MM`` Perioden).  ``"mean"`` for
        index/rate variables, ``"sum"`` for flows, ``"last"`` for stocks.
    """
    df = apply_gold_baseline(df, lag_years=lag_years, monthly_aggregation=monthly_aggregation)

    # Apply table-specific dimensional filters (e.g., Marges == "MW00000")
    if filters:
        for col, value in filters.items():
            if col in df.columns:
                before = len(df)
                if isinstance(value, (list, tuple)):
                    # Multi-value filter: keep rows matching ANY of the values
                    str_values = [str(v) for v in value]
                    df = df[df[col].astype(str).isin(str_values)].copy()
                else:
                    df = df[df[col].astype(str) == str(value)].copy()
                f_log(f"Filter {col}={value}: {before} → {len(df)} rows", c_type="process")

    import re
    # Extract all categorical dimension columns targeted for pivoting
    # Safety Check: Prevent unparsed floating metrics ('3.7') stored as text from being treated as dimensions!
    # CBS strictly flags metrics by appending an underscore and number (e.g., '_1', '_2') to the column name.
    object_cols = df.select_dtypes(include=['object', 'string']).columns
    categorical_cols = [c for c in object_cols if c != 'period_enddate' and not re.search(r'_\d+$', c)]
    
    # Filter out Top-level Aggregations globally (CBS commonly uses 'T00xxxx' for root domain 'Totals')
    for col in categorical_cols:
        # Strip CBS total aggregations to explicitly focus on granular breakdown only
        # Safety Check: If the table ONLY contains T00... (e.g. National level isolated statistics), do not drop them!
        non_total_mask = ~df[col].astype(str).str.startswith('T00', na=False)
        if non_total_mask.any():
            df = df[non_total_mask]

    # Isolate structural integration keys (e.g. Branches) so they represent vertical analytical axes alongside Time.
    # Pivoting 90+ Branches horizontally creates 3000+ column Cartesian explosions that break SQLite!
    structural_keys = [c for c in categorical_cols if 'SBI' in c or 'Branche' in c or 'Bedrijf' in c]
    
    index_cols = ['period_enddate'] + structural_keys
    pivot_cols = [c for c in categorical_cols if c not in structural_keys]
    
    # Cast unparsed metrics proactively
    for c in df.columns:
        if c not in index_cols + pivot_cols and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    value_cols = [c for c in df.columns if c not in index_cols + pivot_cols and pd.api.types.is_numeric_dtype(df[c])]

    # Whitelist: keep only metric columns matching the patterns
    if keep_metrics:
        before_n = len(value_cols)
        value_cols = [
            c for c in value_cols
            if any(re.match(pattern, c) for pattern in keep_metrics)
        ]
        f_log(f"keep_metrics: {before_n} → {len(value_cols)} metric columns "
              f"(patterns: {keep_metrics})", c_type="process")

    # Blacklist: exclude target-proxy or unwanted metric columns before pivoting
    if exclude_metrics:
        before_n = len(value_cols)
        value_cols = [
            c for c in value_cols
            if not any(re.search(pattern, c, re.IGNORECASE) for pattern in exclude_metrics)
        ]
        n_excluded = before_n - len(value_cols)
        if n_excluded > 0:
            f_log(f"Excluded {n_excluded} metric columns matching {exclude_metrics}", c_type="process")

    if pivot_cols and not df.empty:
        # Duplicate detection: warn if the pivot would aggregate multiple rows
        # into a single cell.  For clean data (no duplicates) mean == sum == the
        # single value, so the choice is invisible.  For dirty data, mean is safe
        # for both additive metrics (volumes) and non-additive metrics (rates, %).
        dup_key = index_cols + pivot_cols
        n_dups = df.duplicated(subset=dup_key, keep=False).sum()
        if n_dups > 0:
            f_log(
                f"Pivot has {n_dups} duplicate (index × dimension) rows. "
                f"Using mean aggregation. Consider adding a 'filters' parameter "
                f"to select a specific aggregation level.",
                c_type="warning",
            )

        df_pivoted = df.pivot_table(index=index_cols, columns=pivot_cols, values=value_cols, aggfunc='mean')
        # Flatten MultiIndex hierarchical columns
        df_pivoted.columns = ['_'.join(map(str, col)).strip() for col in df_pivoted.columns.values]
        df_pivoted = df_pivoted.reset_index()
        df = df_pivoted
    else:
        # No pivot — apply keep/exclude directly on the DataFrame
        cols_to_keep = index_cols + value_cols
        df = df[cols_to_keep].copy()

    # Prefix yearly-origin feature columns so downstream code can identify them
    if lag_years > 0:
        structural = {'period_enddate', 'BedrijfstakkenBranchesSBI2008', 'year', 'quarter'}
        rename_map = {c: f"y_{c}" for c in df.columns if c not in structural}
        df = df.rename(columns=rename_map)
        f_log(f"Prefixed {len(rename_map)} feature columns with 'y_' (yearly origin).", c_type="process")

    return df


def add_covid_period_flags(df: pd.DataFrame, date_col: str = "period_enddate") -> pd.DataFrame:
    """
    Appends two mutually exclusive boolean period flags based on quarter-end dates.

    Definitions
    -----------
    covid_period : Q1 2020 <= period_enddate <= Q4 2022  (2020-03-31 … 2022-12-31)
    post_covid   : period_enddate > Q4 2022  (i.e. after 2022-12-31)

    Parameters
    ----------
    df       : DataFrame containing a datetime ``date_col`` column.
    date_col : Name of the quarter-end date column (default: 'period_enddate').

    Returns
    -------
    DataFrame with two new int columns: covid_period, post_covid.
    """
    if date_col not in df.columns:
        f_log(f"add_covid_period_flags: '{date_col}' not found — skipping flags.", c_type="warning")
        return df

    dates = pd.to_datetime(df[date_col], errors="coerce")

    # Inclusive quarter boundaries stored as period end-dates
    COVID_START = pd.Timestamp("2020-03-31")  # end of Q1 2020
    COVID_END   = pd.Timestamp("2022-12-31")  # end of Q4 2022

    df = df.copy()
    df["covid_period"] = ((dates >= COVID_START) & (dates <= COVID_END)).astype(int)
    df["post_covid"]   = (dates > COVID_END).astype(int)

    n_cov  = df["covid_period"].sum()
    n_post = df["post_covid"].sum()
    f_log(
        f"COVID period flags: covid_period={n_cov}, post_covid={n_post} rows",
        c_type="process",
    )
    return df


def add_continuous_regime_features(df: pd.DataFrame, date_col: str = "period_enddate") -> pd.DataFrame:
    """
    Appends two continuous regime-shape features (in quarters).

    Definitions
    -----------
    covid_depth       : 0 before Q1 2020, ramps 1..12 through the 12 COVID
                        quarters (Q1 2020 → Q4 2022), then stays at 12.
                        Encodes "how deep into the COVID disruption we are".
    recovery_quarters : 0 during and before COVID; ramps 1, 2, 3, ... from
                        Q1 2023 onwards.  Encodes "how long since COVID ended".

    Why continuous over the existing binary flags
    ---------------------------------------------
    The binary `covid_period` / `post_covid` are step functions: they shift
    the intercept but cannot model a gradual onset or recovery.  Linear
    models with these continuous features can fit a *slope* of disruption
    and a *recovery slope*, capturing the shape of the shock rather than
    just its presence.

    Day count assumes a 91.3125-day quarter (365.25 / 4).  Cap at 12 for
    covid_depth so the feature plateaus after the COVID window closes.
    """
    if date_col not in df.columns:
        f_log(
            f"add_continuous_regime_features: '{date_col}' not found — skipping.",
            c_type="warning",
        )
        return df

    dates = pd.to_datetime(df[date_col], errors="coerce")
    QUARTER_DAYS = 91.3125  # 365.25 / 4

    # Reference points: end of Q4 2019 (last pre-COVID quarter) and end of
    # Q4 2022 (last COVID quarter).  Q1 2020 = 1 quarter into COVID; Q4 2022
    # = 12 quarters into COVID.
    PRE_COVID_END = pd.Timestamp("2019-12-31")
    COVID_END     = pd.Timestamp("2022-12-31")

    qtrs_into_covid = (
        ((dates - PRE_COVID_END).dt.days / QUARTER_DAYS)
        .clip(lower=0.0, upper=12.0)
    )
    qtrs_since_covid = (
        ((dates - COVID_END).dt.days / QUARTER_DAYS)
        .clip(lower=0.0)
    )

    df = df.copy()
    df["covid_depth"]       = qtrs_into_covid.astype(float)
    df["recovery_quarters"] = qtrs_since_covid.astype(float)

    f_log(
        f"Continuous regime features: covid_depth ∈ "
        f"[{df['covid_depth'].min():.2f}, {df['covid_depth'].max():.2f}], "
        f"recovery_quarters ∈ "
        f"[{df['recovery_quarters'].min():.2f}, {df['recovery_quarters'].max():.2f}]",
        c_type="process",
    )
    return df


def add_regime_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends two regime × structural-feature interactions.

    Definitions
    -----------
    trend_x_post_covid   : trend_index × post_covid.  Lets the post-COVID
                           trend slope differ from the pre-COVID slope —
                           i.e. captures "the underlying drift changed
                           after the regime broke".
    quarter_x_post_covid : quarter × post_covid.  Lets the seasonal pattern
                           shift post-COVID — plausible since WFH changed
                           the seasonality of sick leave reporting.

    Why
    ---
    Linear models cannot discover interactions on their own.  Manually
    constructing the most theoretically motivated ones (trend slope and
    seasonal shape, both regime-dependent) gives the model the lever it
    needs to fit different dynamics in the post-COVID regime without
    expanding parameter count uncontrollably.

    Both features are zero pre-2023 and become their parent feature value
    afterwards, so the model effectively learns:
        y_pred = β_trend × trend + β_trend_post × trend × post_covid
               = β_trend × trend                    (pre-2023)
               = (β_trend + β_trend_post) × trend   (post-2023)
    """
    required = ["trend_index", "quarter", "post_covid"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        f_log(
            f"add_regime_interactions: missing columns {missing} — skipping.",
            c_type="warning",
        )
        return df

    df = df.copy()
    df["trend_x_post_covid"]   = (df["trend_index"] * df["post_covid"]).astype(int)
    df["quarter_x_post_covid"] = (df["quarter"]     * df["post_covid"]).astype(int)

    n_active = int((df["post_covid"] == 1).sum())
    f_log(
        f"Regime interactions: trend_x_post_covid and quarter_x_post_covid "
        f"active on {n_active} post-COVID rows",
        c_type="process",
    )
    return df


def transform_target_fact_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the foundational ML Target/Fact table (e.g. 80072ned).
    Instead of pivoting (which explodes targets), it maintains discrete dimension rows per quarter
    and dynamically applies One-Hot Encoding (OHE) to represent dimensions internally for Machine Learning models.
    """
    df = apply_gold_baseline(df, ML_TARGET_COLUMN)

    # 1. Target Normalization: Handle explicit Dutch comma formatting dynamically
    if ML_TARGET_COLUMN in df.columns:
        if getattr(df[ML_TARGET_COLUMN], 'dtype', None) == object:
            df[ML_TARGET_COLUMN] = df[ML_TARGET_COLUMN].str.replace(',', '.')
        df[ML_TARGET_COLUMN] = pd.to_numeric(df[ML_TARGET_COLUMN], errors='coerce')
        from src.utils.m_imputation import impute_target_variable
        df = impute_target_variable(df, ML_TARGET_COLUMN, 'BedrijfstakkenBranchesSBI2008', 'period_enddate')

    # 2. Temporal Feature Engineering
    if 'period_enddate' in df.columns and pd.api.types.is_datetime64_any_dtype(df['period_enddate']):
        df['year'] = df['period_enddate'].dt.year.astype(int)
        df['quarter'] = df['period_enddate'].dt.quarter.astype(int)
        
        min_date = df['period_enddate'].min()
        if pd.notnull(min_date):
            df['trend_index'] = (
                (df['period_enddate'].dt.year - min_date.year) * 4 + 
                (df['period_enddate'].dt.quarter - min_date.quarter) + 1
            ).astype(int)

    # 3. COVID period flags (binary)
    df = add_covid_period_flags(df)

    # 3b. Continuous regime shape (quarters into / since COVID)
    df = add_continuous_regime_features(df)

    # 3c. Regime × structural interactions (lets linear models fit
    # post-COVID-specific trend slope and seasonal shape)
    df = add_regime_interactions(df)

    import re
    # 2. Dynamic One-Hot Encoding for all dimension bounds remaining
    # Omit unparsed metrics safely utilizing the CBS metric suffix identifier
    object_cols = df.select_dtypes(include=['object', 'string']).columns
    categorical_cols = [c for c in object_cols if c != 'period_enddate' and c != ML_TARGET_COLUMN and not re.search(r'_\d+$', c)]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
    
    return df


# Dynamic Registry Maps Model Identity -> Bound Transformation Function
from functools import partial

# Only the target needs an explicit entry — everything else is auto-registered
TRANSFORMATION_REGISTRY = {
    "80072ned": transform_target_fact_table,
}

# Auto-register yearly tables with their publication lag
for _tid, _lag in CBS_TABLES_YEARLY.items():
    if _tid not in TRANSFORMATION_REGISTRY:
        TRANSFORMATION_REGISTRY[_tid] = partial(transform_generic_feature_table, lag_years=_lag)

# Auto-register monthly tables with their aggregation method.
# CBS_TABLES_MONTHLY maps table_id -> {"mean", "sum", "last", "first"}.
# transform_generic_feature_table forwards monthly_aggregation through to
# apply_gold_baseline, which collapses 3 monthly rows per quarter into one.
for _tid, _agg in CBS_TABLES_MONTHLY.items():
    if _tid not in TRANSFORMATION_REGISTRY:
        TRANSFORMATION_REGISTRY[_tid] = partial(
            transform_generic_feature_table, monthly_aggregation=_agg,
        )

# Auto-register quarterly tables with default transformation
for _tid in CBS_TABLES_TO_LOAD:
    if _tid not in TRANSFORMATION_REGISTRY:
        TRANSFORMATION_REGISTRY[_tid] = transform_generic_feature_table

# ── Manual overrides (tables that need filters) ──────────────────────────
TRANSFORMATION_REGISTRY["86009NED"] = partial(
    transform_generic_feature_table, lag_years=1,
    filters={"Marges": "MW00000"},
    exclude_metrics=[
        "Ziekteverzuimpercentage",
        "AantalWerkdagenVerzuimd",
        "k_1Tot5Werkdagen",
        "k_5Tot20Werkdagen",
        "k_20WerkdagenOfMeer",
        "VerzuimGevallen",
        "Meldingsfrequentie",
        "ZiekteverzuimFrequentie",        
    ],
)
TRANSFORMATION_REGISTRY["85542NED"] = partial(
    transform_generic_feature_table, lag_years=1,
    filters={"Marges": "MW00000", "Kenmerken": "T009002"},
    keep_metrics=["Score"],
)
TRANSFORMATION_REGISTRY["85920NED"] = partial(
    transform_generic_feature_table, filters={"TypeWerkenden": "T001413"}
)
TRANSFORMATION_REGISTRY["80590ned"] = partial(
    transform_generic_feature_table, filters={"Geslacht": "T001038"}
)
TRANSFORMATION_REGISTRY["81433ned"] = partial(
    transform_generic_feature_table, lag_years=1,  # ← was missing
    filters={"GeslachtWerknemer": "T001038", "Dienstverband": "T001007", "KenmerkenBaan": "10000"}
)
TRANSFORMATION_REGISTRY["85278NED"] = partial(
    transform_generic_feature_table, lag_years=1,  # ← was missing
    filters={"Geslacht": "T001038", "Persoonskenmerken": "T009002"}
)
TRANSFORMATION_REGISTRY["85916NED"] = partial(
    transform_generic_feature_table, filters={"Geslacht": "T001038"}
)

TRANSFORMATION_REGISTRY["85917NED"] = partial(
    transform_generic_feature_table,
    exclude_metrics=["Mannen_", "Vrouwen_"]  
)


if __name__ == "__main__":
    db = DatabaseGold(DIR_DB_SILVER, DIR_DB_GOLD)

    # Dedup across the three frequency collections; a table id appearing in
    # more than one (shouldn't happen, but be defensive) is processed once.
    all_tables = list(dict.fromkeys(
        list(CBS_TABLES_TO_LOAD)
        + [t for t in CBS_TABLES_YEARLY if t not in CBS_TABLES_TO_LOAD]
        + [t for t in CBS_TABLES_MONTHLY
           if t not in CBS_TABLES_TO_LOAD and t not in CBS_TABLES_YEARLY]
    ))
    for table_id in all_tables:
        if table_id in TRANSFORMATION_REGISTRY:
            db.process_silver_table(table_id, TRANSFORMATION_REGISTRY[table_id])
        else:
            f_log(f"Setup Warning: Transformation logic not registered for target table: {table_id}", c_type="warning")

    # Synthesize Master Dataset
    db.create_master_training_dataset()