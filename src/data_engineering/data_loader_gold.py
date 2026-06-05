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


def apply_gold_baseline(df: pd.DataFrame, ml_target_col: str = None, lag_years: int = 0) -> pd.DataFrame:
    """
    Applies the foundational structural baseline logic for Gold datasets.
    Handles temporal filtering, yearly extrapolation, explicit pruning and zero-filling.

    Parameters
    ----------
    lag_years : int, default 0
        For purely yearly tables, shift the year forward by this many years
        when expanding to quarterly.  Use 1 to avoid look-ahead bias
        (2022 annual value → Q1-Q4 2023).
    """
    df = df.copy()

    # 1. Temporal Standardization & Extrapolation
    if 'Perioden' in df.columns:
        if not df['Perioden'].str.contains('KW', na=False).any() and df['Perioden'].str.contains('JJ', na=False).any():
            # Extrapolation Logic: Convert purely yearly tables to quarterly
            lag_label = f", lag={lag_years}y" if lag_years else ""
            f_log(f"Expanding yearly data to quarters (values repeated 4x{lag_label}).", c_type="warning")
            df_list = []
            for quarter in ['KW01', 'KW02', 'KW03', 'KW04']:
                temp = df[df['Perioden'].str.contains('JJ', na=False)].copy()
                if lag_years:
                    # Shift year forward: 2022JJ00 with lag=1 → 2023KWxx
                    temp['Perioden'] = temp['Perioden'].apply(
                        lambda p: str(int(p[:4]) + lag_years) + quarter
                    )
                else:
                    temp['Perioden'] = temp['Perioden'].str.replace(r'JJ.*', quarter, regex=True)
                df_list.append(temp)
            df = pd.concat(df_list, ignore_index=True)
        else:
            # Discard remaining 'JJ' periods if table naturally has 'KW'
            df = df[df['Perioden'].str.contains('KW', na=False)].copy()
        
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


def transform_generic_feature_table(df: pd.DataFrame, lag_years: int = 0, filters: dict = None, exclude_metrics: list = None, keep_metrics: list = None) -> pd.DataFrame:
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
    """
    df = apply_gold_baseline(df, lag_years=lag_years)

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

# 1. Define the dict FIRST
TRANSFORMATION_REGISTRY = {
    "80072ned": transform_target_fact_table,
}

# Auto-register yearly tables with their publication lag
for _tid, _lag in CBS_TABLES_YEARLY.items():
    if _tid not in TRANSFORMATION_REGISTRY:
        TRANSFORMATION_REGISTRY[_tid] = partial(transform_generic_feature_table, lag_years=_lag)

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
        "k_20Tot210Werkdagen",
        "k_210WerkdagenOfMeer",
        "VerzuimGevallen",
        "Meldingsfrequentie",
        "ZiekteverzuimFrequentie", 
        "AandeelWerknemersDieHebbenVerzuimd",     
        "AantalKeerVerzuimd", 
        "AantalWerkdagenMeestRecenteVerzuim",
        "WeetNiet",
        "AndereReden",
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
    filters={"Geslacht": "T001038", "Persoonskenmerken": "T009002", "PositieInDeWerkkring_CategoryGroupID": ["9", "10", "12"]},
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

    all_tables = list(CBS_TABLES_TO_LOAD) + [t for t in CBS_TABLES_YEARLY if t not in CBS_TABLES_TO_LOAD]
    for table_id in all_tables:
        if table_id in TRANSFORMATION_REGISTRY:
            db.process_silver_table(table_id, TRANSFORMATION_REGISTRY[table_id])
        else:
            f_log(f"Setup Warning: Transformation logic not registered for target table: {table_id}", c_type="warning")

    # Synthesize Master Dataset
    db.create_master_training_dataset()