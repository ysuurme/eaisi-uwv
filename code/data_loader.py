import os
import json
import logging
from pathlib import Path

# --- Third Party Libraries ---
import cbsodata


# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

DIR_DATA_RAW = "data//0_raw"
DIR_DB_BRONZE = "sqlite:///data//1_bronze//bronze_data.db"


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
        
        file_name = f"{table_id}.json"
        output_path = self.output_dir / file_name

        if output_path.exists():
            logger.info(f"File '{output_path}' already exists. Skipping download.")
            return output_path

        logger.info(f"Fetching data for table '{table_id}'...")
        
        try:
            data = cbsodata.get_data(table_id)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Successfully saved {len(data)} records to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to fetch or save data for table '{table_id}': {e}")
            raise


# 1. Ingest Raw Data
loader = CBSDataLoader(output_dir=DIR_DATA_RAW)
loader.fetch_table("82070ENG")
