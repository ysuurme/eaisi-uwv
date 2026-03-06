import logging
from pathlib import Path


# --- Third Party Libraries ---
import pandas as pd
from sqlalchemy import MetaData, create_engine, text

# --- Configuration ---
try:
    from config import DIR_DB_SILVER, DIR_DB_GOLD
except ImportError:
    raise ImportError("Configuration file 'config.py' not found or missing required variables.")

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# --- Gold Manager ---
class DatabaseGold:
    """
    Manages the SQLite database in the gold layer.
    Formats tables from silver into clean features for analysis and machine learning into a single wide table. 
    """
    def __init__(self, db_silver_path: Path, db_gold_path: Path):
        self.db_silver_path = Path(db_silver_path)
        self.db_gold_path = Path(db_gold_path)
        
        # Ensure directory exists
        self.db_gold_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.engine = create_engine(f"sqlite:///{self.db_gold_path}")
        self.metadata = MetaData()


    def process_silver_table(self, identifier: str, transformation_func: callable):
        """
        Fetches a table from Silver DB, applies transformation, and stores in Gold DB.
        """
        silver_table_name = f"{identifier}_silver"
        gold_table_name = f"{identifier}_gold"

        logger.info(f"Processing {silver_table_name} -> {gold_table_name}")

        with self.engine.connect() as conn:
            # Attach Silver Database
            conn.execute(text("ATTACH DATABASE :silver_path AS silver"), {"silver_path": str(self.db_silver_path)})
            
            try:
                # Fetch data using pandas
                query = f"SELECT * FROM silver.\"{silver_table_name}\""
                df = pd.read_sql_query(query, conn)
            except Exception as e:
                logger.error(f"Error fetching {silver_table_name}: {e}")
                return
            finally:
                conn.execute(text("DETACH DATABASE silver"))

        # Apply Transformation
        try:
            df_gold = transformation_func(df)
        except Exception as e:
            logger.error(f"Error transforming {silver_table_name}: {e}")
            return

        # Store in Gold
        df_gold.to_sql(gold_table_name, self.engine, if_exists='replace', index=False)
        logger.info(f"Successfully stored {gold_table_name}")


# --- Transformation Functions ---
def transform_80072ned(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and reformats the 80072ned table.
    """
    df = df.copy()

    # 1. Reformat 'Ziekteverzuimpercentage_1' to float
    if 'Ziekteverzuimpercentage_1' in df.columns:
        # Handle potential string formatting (Dutch commas)
        if df['Ziekteverzuimpercentage_1'].dtype == object:
            df['Ziekteverzuimpercentage_1'] = df['Ziekteverzuimpercentage_1'].str.replace(',', '.')
        df['Ziekteverzuimpercentage_1'] = pd.to_numeric(df['Ziekteverzuimpercentage_1'], errors='coerce')

    # 2. Reformat 'Perioden' to datetime format per quarter
    if 'Perioden' in df.columns:
        # Filter only quarters (KW)
        df = df[df['Perioden'].str.contains('KW', na=False)]
        
        def parse_quarter(val):
            # Expected format: YYYYKWQQ or YYYYKWQ
            parts = val.split('KW')
            if len(parts) == 2:
                year = int(parts[0])
                quarter = int(parts[1])
                # Convert to datetime (start of quarter)
                return pd.Timestamp(year=year, month=(quarter - 1) * 3 + 1, day=1)
            return pd.NaT

        df['Perioden_dt'] = df['Perioden'].apply(parse_quarter)

    # 3. Reformat 'BedrijfskenmerkenSBI2008_CategoryGroupID' to numeric
    if 'BedrijfskenmerkenSBI2008_CategoryGroupID' in df.columns:
        df['BedrijfskenmerkenSBI2008_CategoryGroupID'] = pd.to_numeric(df['BedrijfskenmerkenSBI2008_CategoryGroupID'], errors='coerce')

    # 4. One-hot encode 'BedrijfskenmerkenSBI2008'
    if 'BedrijfskenmerkenSBI2008' in df.columns:
        # Create dummies, prefixing with Bedrijfskenmerken 'SBI' for clarity, using 0/1 integers
        df = pd.get_dummies(df, columns=['BedrijfskenmerkenSBI2008'], prefix='SBI', dtype=int)

    # 5. One-hot encode 'Perioden_Status'
    if 'Perioden_Status' in df.columns:
        df = pd.get_dummies(df, columns=['Perioden_Status'], prefix='PeriodenStatus', dtype=int)

    # Finally, drop unwanted columns
    columns_to_drop = ["bronze_pk", "_source_file", "ID", "Perioden", "Perioden_Description", "Perioden_Title", "BedrijfskenmerkenSBI2008_Title", "BedrijfskenmerkenSBI2008_Description"]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    return df


# --- Main execution ---
if __name__ == "__main__":
    db = DatabaseGold(DIR_DB_SILVER, DIR_DB_GOLD)
    # Process 80072ned
    db.process_silver_table("80072ned", transform_80072ned)