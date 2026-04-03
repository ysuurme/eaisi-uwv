from pathlib import Path

# --- Third Party Libraries ---
import pandas as pd
from sqlalchemy import MetaData, create_engine, text

# --- Configuration ---
from src.config import DIR_DB_SILVER, DIR_DB_GOLD

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
        except Exception as e:
            f_log(f"Error transforming {silver_table_name}: {e}", c_type="error")
            return

        df_gold.to_sql(gold_table_name, self.engine, if_exists='replace', index=False)
        f_log(f"Stored {gold_table_name}", c_type="store")


def transform_80072ned(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and reformats the 80072ned table with a 'Zero-Null' policy.
    Missing values in critical columns are now captured as soft errors in the Silver layer.
    """
    df = df.copy()

    # 1. Reformat Target to float64
    if 'Ziekteverzuimpercentage_1' in df.columns:
        if df['Ziekteverzuimpercentage_1'].dtype == object:
            df['Ziekteverzuimpercentage_1'] = df['Ziekteverzuimpercentage_1'].str.replace(',', '.')
        df['Ziekteverzuimpercentage_1'] = pd.to_numeric(df['Ziekteverzuimpercentage_1'], errors='coerce')

    # 2. Temporal Processing
    if 'Perioden' in df.columns:
        df = df[df['Perioden'].str.contains('KW', na=False)]
        def parse_quarter(val):
            parts = val.split('KW')
            if len(parts) == 2:
                return pd.Timestamp(year=int(parts[0]), month=(int(parts[1]) - 1) * 3 + 1, day=1)
            return pd.NaT
        df['Perioden_dt'] = df['Perioden'].apply(parse_quarter)

    # 3. Numeric Casting
    if 'BedrijfskenmerkenSBI2008_CategoryGroupID' in df.columns:
        df['BedrijfskenmerkenSBI2008_CategoryGroupID'] = pd.to_numeric(df['BedrijfskenmerkenSBI2008_CategoryGroupID'], errors='coerce')

    # 4. One-hot Encoding
    if 'BedrijfskenmerkenSBI2008' in df.columns:
        df = pd.get_dummies(df, columns=['BedrijfskenmerkenSBI2008'], prefix='SBI', dtype=int)
    
    if 'Perioden_Status' in df.columns:
        df = pd.get_dummies(df, columns=['Perioden_Status'], prefix='PeriodenStatus', dtype=int)

    # 5. --- GOLD QUALITY GATE ---
    columns_to_drop = ["bronze_pk", "_source_file", "ID", "Perioden", "Perioden_Description", "Perioden_Title", 
                       "BedrijfskenmerkenSBI2008_Title", "BedrijfskenmerkenSBI2008_Description"]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    df = df.dropna(subset=['Ziekteverzuimpercentage_1', 'Perioden_dt'])

    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[numeric_cols] = df[numeric_cols].astype('float64')

    f_log(f"Gold Quality Gate: Processed {len(df)} rows. 0 NaNs remain.", c_type="success")
    return df


if __name__ == "__main__":
    db = DatabaseGold(DIR_DB_SILVER, DIR_DB_GOLD)
    db.process_silver_table("80072ned", transform_80072ned)