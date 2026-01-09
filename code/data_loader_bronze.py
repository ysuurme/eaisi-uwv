import json
import logging
from pathlib import Path

# --- Third Party Libraries ---
from sqlalchemy import create_engine, insert, MetaData, Table, Column, String, Integer, Float


# --- Configuration ---
try:
    from config import DIR_DATA_RAW, DIR_DB_BRONZE, CBS_TABLES_T3
except ImportError:
    raise ImportError("Configuration file 'config.py' not found or missing required variables.")


# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# --- "Data Lake" 0_raw to Database 1_bronze Manager Class ---
class DatabaseBronze:
    """
    Manages the SQLite database in the bronze layer using SQLAlchemy Core.
    Dynamically maps JSON structures to SQLite tables.
    """
    def __init__(self, data_raw_path: Path, db_path: Path):
        self.data_raw_path = data_raw_path
        self.db_path = db_path
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.metadata = MetaData()


    def _infer_primary_key(self, file_name, data_sample: dict) -> str:
        # Determine the Primary Key Name, for Facts == "ID", for Dimensions try "Key" or "DimensionKey"
        potential_keys = ["ID", "Key", "DimensionKey"]
        primary_key = next((k for k in potential_keys if k in data_sample), None)
        if not primary_key:
            logger.error(f"No ID found in {file_name}, skipping table insertion.")
            return
        return primary_key
    

    def _clean_data(self, file_name, data, primary_key) -> list:
        """Cleans string fields by stripping whitespace and add key columns."""
        cleaned_data = []
        for record in data:
            clean_record = {k: (v.strip() if isinstance(v, str) else v) for k, v in record.items()}

            # Create the specific Concatenated Primary Key
            pk = clean_record.get(primary_key)
            clean_record["bronze_pk"] = f"{file_name}_{pk}"
            clean_record["_source_file"] = file_name
            cleaned_data.append(clean_record)

        return cleaned_data


    def _infer_column_type(self, value):
            """Simple type inference for 'Raw' data integrity."""
            if isinstance(value, int):
                return Integer
            if isinstance(value, float):
                return Float
            return String  # Default to String for codes/keys/NULL


    def ingest_0_raw_folder(self, identifier: str):
        """
        Scans a specific identifier folder and ingests Fact and Dimension .json files.
        """
        folder_path = Path(self.data_raw_path, identifier)
        if not folder_path.exists():
            logger.error(f"Identifier folder {identifier} not found in {self.data_raw_path}")
            return

        for json_path in folder_path.glob("*.json"):
            # Determine Fact table or Dimension table
            if json_path.name == "TypedDataSet.json":
                table_name = f"{identifier}_fact"
            else:
                table_name = f"{identifier}_dim_{json_path.stem}"

            self.insert_json_data(json_path, table_name)

    def insert_json_data(self, json_path: Path, table_name: str):
        """
        Reads JSON, dynamically creates table based on keys, and bulk inserts.
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)          
            if not data:
                return
            
        except Exception as e:
            logger.error(f"Failed to read {json_path}: {e}")
            return

        # Inspect the first record to define the table schema dynamically
        file_name = json_path.name
        data_sample = data[0]
        columns = []

        # Determine the Table Primary Key Name, for Facts == "ID", for Dimensions try "Key" or "DimensionKey"
        primary_key = self._infer_primary_key(file_name, data_sample)

        # Pre-process data: clean strings and add supporting columns
        cleaned_data = self._clean_data(file_name, data, primary_key)   
      
        # Ensure bronze_pk is the Primary Key and the first column, subsequently append list with other columns
        columns.append(Column("bronze_pk", String, primary_key=True))
        for key, value in cleaned_data[0].items():
            if key == "bronze_pk":
                continue # Skip bronze_pk as it's already added to the column list first
            col_type = self._infer_column_type(value) # Infer column type from sample value
            columns.append(Column(key, col_type, primary_key=False))

        # Create Table
        table = Table(table_name, self.metadata, *columns, extend_existing=True)
        table.drop(self.engine, checkfirst=True) # Reset Bronze for fresh landing
        table.create(self.engine)

        # Bulk Insert using SQLAlchemy Core for speed
        with self.engine.begin() as conn:
            try:
                conn.execute(insert(table), cleaned_data)
                logger.info(f"Loaded {len(cleaned_data)} rows into {table_name}")
            except Exception as e:
                logger.error(f"Bulk insert failed for {table_name}: {e}")


# --- Main execution ---
db = DatabaseBronze(DIR_DATA_RAW, DIR_DB_BRONZE)

# Process tables defined in config
for table_id in CBS_TABLES_T3:
    db.ingest_0_raw_folder(table_id)