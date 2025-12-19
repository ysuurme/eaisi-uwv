import os
import logging
from pathlib import Path


# --- Third Party Libraries ---
import cbsodata


# --- Configuration ---
DIR_DATA_RAW = "data//0_raw"
DIR_DB_BRONZE = "sqlite:///data//1_bronze//bronze_data.db"
TABLES = ["82070ENG", "83765NED", "47003NED"]



# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --- Data ingestion to "Data Lake" 0_raw ---
class CBSDataLoader:
    """
    Responsible for retrieving data from CBS Open Data and storing it as raw JSON. It is an industry standard to save the raw JSON first, 
    hence creating a "Data Lake". This setup allows for re-processing raw data without re-querying the API—saving time and being a 
    "good citizen" to the CBS servers.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def fetch_table(self, table_id: str) -> Path:
        """Fetches a dataset from CBS by its Table ID and saves it to the output directory."""
        
        output_dir_table = Path(self.output_dir, table_id)

        if output_dir_table.exists():
            logger.info(f"File '{output_dir_table}' already exists, table not fetched.")
            return output_dir_table

        logger.debug(f"Fetching data for table '{table_id}'...")
        
        try:
            data = cbsodata.get_data(table_id, dir=output_dir_table)
            logger.info(f"Successfully saved table with {len(data)} records to {output_dir_table}")
            return output_dir_table
        except Exception as e:
            logger.error(f"Failed to save data for table '{table_id}': {e}")
            raise


# Ingest Raw Data
loader = CBSDataLoader(output_dir=DIR_DATA_RAW)

for table in TABLES:
    loader.fetch_table(table)
