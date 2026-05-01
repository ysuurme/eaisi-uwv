import json
from pathlib import Path

# --- Third Party Libraries ---
from sqlalchemy import create_engine, insert, MetaData, Table, Column, String, Integer, Float

# --- Configuration ---
from src.config import DIR_DATA_RAW, DIR_DB_BRONZE, CBS_TABLES_T65

# --- Logging ---
from src.utils.m_log import f_log


def infer_primary_key(file_name: str, data_sample: dict) -> str:
    """Determines the Primary Key Name from the data sample."""
    potential_keys = ["ID", "Key", "DimensionKey"]
    primary_key = next((k for k in potential_keys if k in data_sample), None)
    if not primary_key:
        f_log(f"No ID found in {file_name}, skipping table insertion.", c_type="error")
        return None
    return primary_key

def clean_bronze_data(file_name: str, data: list, primary_key: str) -> list:
    """Cleans string fields by stripping whitespace and add key columns."""
    cleaned_data = []
    for record in data:
        clean_record = {k: (v.strip() if isinstance(v, str) else v) for k, v in record.items()}
        pk = clean_record.get(primary_key)
        clean_record["bronze_pk"] = f"{file_name}_{pk}"
        clean_record["_source_file"] = file_name
        cleaned_data.append(clean_record)
    return cleaned_data

def infer_column_type(value):
    """Simple type inference for 'Raw' data integrity."""
    if isinstance(value, int):
        return Integer
    if isinstance(value, float):
        return Float
    return String


class DatabaseBronze:
    """
    Manages the SQLite database in the bronze layer using SQLAlchemy Core.
    Dynamically maps JSON structures to SQLite tables.
    """
    def __init__(self, data_raw_path: Path, db_path: Path):
        self.data_raw_path = data_raw_path
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path.as_posix()}")
        self.metadata = MetaData()


    def ingest_0_raw_folder(self, identifier: str):
        """Scans a specific identifier folder and ingests Fact and Dimension .json files."""
        folder_path = Path(self.data_raw_path, identifier)
        if not folder_path.exists():
            f_log(f"Identifier folder {identifier} not found in {self.data_raw_path}", c_type="error")
            return

        for json_path in folder_path.glob("*.json"):
            if json_path.name == "TypedDataSet.json":
                table_name = f"{identifier}_fact"
            else:
                table_name = f"{identifier}_dim_{json_path.stem}"

            self.insert_json_data(json_path, table_name)

    def insert_json_data(self, json_path: Path, table_name: str):
        """Reads JSON, dynamically creates table based on keys, and bulk inserts."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)          
            if not data:
                return
            
        except Exception as e:
            f_log(f"Failed to read {json_path}: {e}", c_type="error")
            return

        file_name = json_path.name
        data_sample = data[0]
        columns = []

        primary_key = infer_primary_key(file_name, data_sample)
        if not primary_key:
            return

        cleaned_data = clean_bronze_data(file_name, data, primary_key)   
      
        columns.append(Column("bronze_pk", String, primary_key=True))
        for key, value in cleaned_data[0].items():
            if key == "bronze_pk":
                continue
            col_type = infer_column_type(value)
            columns.append(Column(key, col_type, primary_key=False))

        table = Table(table_name, self.metadata, *columns, extend_existing=True)
        table.drop(self.engine, checkfirst=True)
        table.create(self.engine)

        with self.engine.begin() as conn:
            try:
                conn.execute(insert(table), cleaned_data)
                f_log(f"Loaded {len(cleaned_data)} rows into {table_name}", c_type="success")
            except Exception as e:
                f_log(f"Bulk insert failed for {table_name}: {e}", c_type="error")


if __name__ == "__main__":
    db = DatabaseBronze(DIR_DATA_RAW, DIR_DB_BRONZE)
    for table_id in CBS_TABLES_T65:
        db.ingest_0_raw_folder(table_id)