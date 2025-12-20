import os
import logging
from pathlib import Path
import json

# --- Third Party Libraries ---
import cbsodata


# --- Configuration ---
DIR_DATA_RAW = "data//0_raw"
CBS_TABLES_T3 = ["80072ned", "83415NED", "83157NED"] 
CBS_TABLES_T65 = [
    "80072ned", "83415NED", "86009NED", "86010NED", "85998NED", "86011NED", "86168NED", 
    "83156NED", "84434NED", "83157NED", "84436NED", "83159NED", "84435NED", "84031NED", 
    "84030NED", "83158NED", "85718NED", "85786NED", "86047NED", "86067NED", "81628ENG", 
    "83005ENG", "81174ENG", "81177eng", "81175eng", "85647NED", "80590ned", "85264NED", 
    "85224NED", "85312NED", "83752NED", "85271NED", "83583NED", "83582NED", "85918NED", 
    "85920NED", "81431ned", "83451NED", "85274NED", "85275NED", "85276NED", "85278NED", 
    "81178ENG", "82470NED", "85544NED", "83700ENG", "84432ENG", "71541eng", "82801ENG", 
    "86092NED", "85900NED", "84939NED", "84671NED", "84669NED", "83913NED", "83648NED", 
    "83734NED", "81414NED", "84566NED", "82439NED", "82325NED", "81408NED", "83738NED", 
    "82623NED", "81567NED"
]

"""
Based on the attributes provided and the metadata from the cbs_table_list.json, these are the most relevant tables for predicting sick leave.
Example CBS table IDs can be found at https://github.com/J535D165/cbsodata/blob/main/README.md
"""

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

    
    def _file_exists(self, file_dir: Path) -> bool:
        """Checks if a file directory already exists."""
        if file_dir.exists():
            logger.info(f"File '{file_dir}' already exists, data not fetched.")
            return True

    
    def get_table_list(self) -> list:
        """Fetches the list of available CBS tables."""
        if self._file_exists(Path(self.output_dir, "cbs_table_list.json")):
            return
        
        try:
            tables = cbsodata.get_table_list()
            logger.info(f"Fetched {len(tables)} tables from CBS.")
            table_info = [
                {
                    "Identifier": table['Identifier'],
                    "Title": table['Title'],
                    "ShortDescription": table['ShortDescription']
                } for table in tables
            ]

            output_path = Path(self.output_dir, "cbs_table_list.json")
            with open(output_path, 'w') as f:
                json.dump(table_info, f, indent=4)
            logger.info(f"Saved table list to: {output_path}")
            return tables
        
        except Exception as e:
            logger.error(f"Failed to fetch table list: {e}")
            return []


    def get_table(self, table_id: str) -> Path:
        """Fetches a dataset from CBS by its Table ID and saves it to the output directory."""
        output_dir_table = Path(self.output_dir, table_id)
        if self._file_exists(output_dir_table):
            return
       
        try:
            logger.debug(f"Getting data for table '{table_id}'...")
            data = cbsodata.get_data(table_id, dir=output_dir_table)
            logger.info(f"Successfully saved table with {len(data)} records to {output_dir_table}")
            return output_dir_table
        
        except Exception as e:
            logger.error(f"Failed to save data for table '{table_id}': {e}")
            raise


# Ingest Raw Data
loader = CBSDataLoader(output_dir=DIR_DATA_RAW)

loader.get_table_list()

for table in CBS_TABLES_T3:
    loader.get_table(table)