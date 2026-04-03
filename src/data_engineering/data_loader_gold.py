"""
Data Loader for the Gold Layer.
Formats tables from silver into clean numerical timeseries features ready for ML ingestion.
"""
from pathlib import Path

# --- Third Party Libraries ---
import pandas as pd
from sqlalchemy import MetaData, create_engine, text

# --- Configuration ---
from src.config import DIR_DB_SILVER, DIR_DB_GOLD, ML_TARGET_COLUMN

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
        self.engine = create_engine(f"sqlite:///{self.db_gold_path}")
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
            if df_gold[numeric_cols].isnull().any().any():
                f_log(f"WARNING: Imputing missing values with 0 for {identifier}. Validate optimal imputation logic.", c_type="warning")
            df_gold[numeric_cols] = df_gold[numeric_cols].fillna(0).astype('float64')

            f_log(f"Gold Quality Gate: Processed {len(df_gold)} rows. 0 NaNs remain.", c_type="success")
            
            # Persist to Gold securely inside the explicit try-block to catch SQLite layout boundaries cleanly
            df_gold.to_sql(gold_table_name, self.engine, if_exists='replace', index=False)
            f_log(f"Stored {gold_table_name}", c_type="store")
            
        except Exception as e:
            f_log(f"Error transforming/storing {silver_table_name}: {e}", c_type="error")
            return


def apply_gold_baseline(df: pd.DataFrame, ml_target_col: str = None) -> pd.DataFrame:
    """
    Applies the foundational structural baseline logic for Gold datasets.
    Handles temporal filtering, yearly extrapolation, explicit pruning and zero-filling.
    """
    df = df.copy()

    # 1. Temporal Standardization & Extrapolation
    if 'Perioden' in df.columns:
        if not df['Perioden'].str.contains('KW', na=False).any() and df['Perioden'].str.contains('JJ', na=False).any():
            # Extrapolation Logic: Convert purely yearly tables to quarterly
            f_log("WARNING: Extrapolating yearly data to quarters. Values repeated 4x! Replace with proper imputation strategy/percentages if needed.", c_type="warning")
            df_list = []
            for quarter in ['KW01', 'KW02', 'KW03', 'KW04']:
                temp = df[df['Perioden'].str.contains('JJ', na=False)].copy()
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

    # 2. Target Column Guard
    if ml_target_col and ml_target_col in df.columns:
        df = df.dropna(subset=[ml_target_col])

    # 3. Column Pruning
    meta_cols = ["silver_id", "bronze_pk", "_source_file", "ID", "Perioden"]
    descriptive_suffixes = ('_Description', '_Title', '_Status', '_CategoryGroupID')
    
    cols_to_drop = [c for c in meta_cols if c in df.columns]
    cols_to_drop += [c for c in df.columns if c.endswith(descriptive_suffixes)]
    
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Drop NaNs on specific crucial identifier 
    if 'period_enddate' in df.columns:
        df = df.dropna(subset=['period_enddate'])

    return df


def transform_generic_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms observational feature datasets dynamically into ML features.
    Applies strict pivoting by translating granular rows across all available dimensions into distinct column features.
    """
    df = apply_gold_baseline(df)

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

    if pivot_cols and not df.empty:
        df_pivoted = df.pivot_table(index=index_cols, columns=pivot_cols, values=value_cols, aggfunc='sum')
        # Flatten MultiIndex hierarchical columns
        df_pivoted.columns = ['_'.join(map(str, col)).strip() for col in df_pivoted.columns.values]
        df_pivoted = df_pivoted.reset_index()
        df = df_pivoted

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
        # Hard drop rule enforcing valid target constraints
        df = df.dropna(subset=[ML_TARGET_COLUMN])

    import re
    # 2. Dynamic One-Hot Encoding for all dimension bounds remaining
    # Omit unparsed metrics safely utilizing the CBS metric suffix identifier
    object_cols = df.select_dtypes(include=['object', 'string']).columns
    categorical_cols = [c for c in object_cols if c != 'period_enddate' and c != ML_TARGET_COLUMN and not re.search(r'_\d+$', c)]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
    
    return df


# Dynamic Registry Maps Model Identity -> Bound Transformation Function
TRANSFORMATION_REGISTRY = {
    "80072ned": transform_target_fact_table,
    "83415NED": transform_generic_feature_table,
    "85916NED": transform_generic_feature_table,
    "85918NED": transform_generic_feature_table,
    "85919NED": transform_generic_feature_table,
    "85920NED": transform_generic_feature_table
}

if __name__ == "__main__":
    db = DatabaseGold(DIR_DB_SILVER, DIR_DB_GOLD)
    
    # Retrieve all tables dynamically from the Silver Database directly
    import sqlite3
    with sqlite3.connect(DIR_DB_SILVER) as conn:
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    
    # Parse identifiers ('85916NED_silver' -> '85916NED')
    tables_to_process = [t.replace("_silver", "") for t in tables['name'].tolist() if t.endswith("_silver")]
    
    for table_id in tables_to_process:
        if table_id in TRANSFORMATION_REGISTRY:
            db.process_silver_table(table_id, TRANSFORMATION_REGISTRY[table_id])
        else:
            f_log(f"Setup Warning: Transformation logic not registered for target table: {table_id}", c_type="warning")