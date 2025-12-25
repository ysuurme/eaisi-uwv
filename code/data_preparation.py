"""
Data Loader for the Silver Layer.
Implements the Star Schema strategy by joining Fact and Dimension tables from Bronze.
"""
import logging
from pathlib import Path

# --- Third Party Libraries ---
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float, select, insert

# --- Configuration ---
try:
    from config import DIR_DB_BRONZE, DIR_DB_SILVER
except ImportError:
    raise ImportError("Configuration file 'config.py' not found or missing required variables.")

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class DatabaseSilver:
    """
    Manages the SQLite database in the silver layer.
    Joins Fact and Dimension tables from Bronze into a single wide table.
    """
    def __init__(self, db_bronze_path: Path, db_silver_path: Path):
        self.db_bronze_path = db_bronze_path
        self.db_silver_path = db_silver_path
        
        # Ensure directory exists
        if isinstance(self.db_silver_path, Path):
            self.db_silver_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.engine_bronze = create_engine(f"sqlite:///{self.db_bronze_path}")
        self.engine_silver = create_engine(f"sqlite:///{self.db_silver_path}")
        self.metadata_bronze = MetaData()
        self.metadata_silver = MetaData()

    def _infer_column_type(self, value):
        """Simple type inference for Silver schema."""
        if isinstance(value, int):
            return Integer
        if isinstance(value, float):
            return Float
        return String

    def create_silver_table(self, identifier: str):
        """
        Joins Fact and Dimensions for a given identifier and creates a Silver table.
        """
        logger.info(f"Starting Silver transformation for identifier: {identifier}")

        # 1. Reflect Bronze Tables
        self.metadata_bronze.reflect(bind=self.engine_bronze)
        
        fact_table_name = f"{identifier}_fact"
        if fact_table_name not in self.metadata_bronze.tables:
            logger.error(f"Fact table {fact_table_name} not found in Bronze.")
            return

        fact_table = self.metadata_bronze.tables[fact_table_name]
        
        # 2. Identify Dimension Tables
        dim_tables = []
        dim_prefix = f"{identifier}_dim_"
        for name, table in self.metadata_bronze.tables.items():
            if name.startswith(dim_prefix):
                dim_tables.append(table)

        logger.info(f"Found {len(dim_tables)} dimension tables for {fact_table_name}.")

        # 3. Build Query: Start with Fact Table
        query = select(fact_table)
        
        # 4. Join Dimensions
        for dim_table in dim_tables:
            # Heuristic: Dimension name suffix (e.g. 'RegioS' from '..._dim_RegioS')
            # matches a column in Fact table.
            dim_suffix = dim_table.name.replace(dim_prefix, "")
            
            if dim_suffix in fact_table.c:
                fact_join_col = fact_table.c[dim_suffix]
                
                # Infer PK in Dim (Bronze strategy used "ID", "Key", "DimensionKey")
                dim_pk_col = None
                for candidate in ["Key", "DimensionKey", "ID", "Code"]:
                    if candidate in dim_table.c:
                        dim_pk_col = dim_table.c[candidate]
                        break
                
                if dim_pk_col is not None:
                    # Perform Left Join to preserve Fact records even if Dim is missing
                    query = query.join(dim_table, fact_join_col == dim_pk_col, isouter=True)
                    
                    # Select descriptive columns from Dimension
                    # Exclude keys and system columns to keep the table clean
                    for col in dim_table.c:
                        if col.name not in [dim_pk_col.name, "bronze_pk", "_source_file"]:
                            # Alias column to avoid collision (e.g. RegioS_Title)
                            query = query.add_columns(col.label(f"{dim_suffix}_{col.name}"))
                else:
                    logger.warning(f"Skipping join for {dim_table.name}: No Primary Key found.")
            else:
                logger.debug(f"Skipping join for {dim_table.name}: No matching FK column '{dim_suffix}' in Fact.")

        # 5. Execute Query
        with self.engine_bronze.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()
            
            if not rows:
                logger.warning(f"No data found for {identifier} after join.")
                return
                
            # 6. Infer Schema for Silver Table
            column_names = result.keys()
            sample_row = rows[0]
            
            silver_columns = []
            for col_name, value in zip(column_names, sample_row):
                col_type = self._infer_column_type(value)
                # Use 'ID' from Fact as the Primary Key for Silver if present
                is_pk = (col_name == "ID")
                silver_columns.append(Column(col_name, col_type, primary_key=is_pk))

            # 7. Create Silver Table
            silver_table_name = f"{identifier}_silver"
            silver_table = Table(silver_table_name, self.metadata_silver, *silver_columns, extend_existing=True)
            
            silver_table.drop(self.engine_silver, checkfirst=True)
            silver_table.create(self.engine_silver)
            
            # 8. Bulk Insert
            data_to_insert = [dict(zip(column_names, row)) for row in rows]
            
            with self.engine_silver.begin() as silver_conn:
                silver_conn.execute(insert(silver_table), data_to_insert)
                logger.info(f"Loaded {len(data_to_insert)} rows into {silver_table_name}")


# --- Main execution ---
db = DatabaseSilver(DIR_DB_BRONZE, DIR_DB_SILVER)
db.create_silver_table("80072ned")
