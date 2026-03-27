import json
from pathlib import Path

# --- Third Party Libraries ---
import cbsodata

# --- Configuration ---
from src.config import DIR_DATA_RAW, CBS_TABLES_T3, CBS_TABLES_T65

# --- Logging ---
from src.utils.m_log import f_log


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
            f_log(f"File '{file_dir}' already exists, data not fetched.")
            return True

    
    def get_table_list(self) -> list:
        """Fetches the list of available CBS tables."""
        if self._file_exists(Path(self.output_dir, "cbs_table_list.json")):
            return
        
        try:
            tables = cbsodata.get_table_list()
            f_log(f"Fetched {len(tables)} tables from CBS.", c_type="success")
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
            f_log(f"Saved table list to: {output_path}", c_type="store")
            return tables
        
        except Exception as e:
            f_log(f"Failed to fetch table list: {e}", c_type="error")
            return []


    def get_table(self, table_id: str) -> Path:
        """Fetches a dataset from CBS by its Table ID and saves it to the output directory."""
        output_dir_table = Path(self.output_dir, table_id)
        if self._file_exists(output_dir_table):
            return
       
        try:
            f_log(f"Getting data for table '{table_id}'...", c_type="debug")
            data = cbsodata.get_data(table_id, dir=output_dir_table)
            f_log(f"Saved table with {len(data)} records to {output_dir_table}", c_type="store")
            return output_dir_table
        
        except Exception as e:
            f_log(f"Failed to save data for table '{table_id}': {e}", c_type="error")
            raise


if __name__ == "__main__":
    loader = CBSDataLoader(output_dir=DIR_DATA_RAW)
    loader.get_table_list()
    for table in CBS_TABLES_T65:
        loader.get_table(table)