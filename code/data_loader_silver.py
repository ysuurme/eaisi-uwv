"""
Data Loader for the Silver Layer.
Implements the Star Schema strategy by joining Fact and Dimension tables from Bronze.
"""
import logging
from pathlib import Path
from typing import List, Any, Optional

# --- Third Party Libraries ---
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float, select, insert

# --- Configuration ---
try:
    from config import DIR_DB_BRONZE, DIR_DB_SILVER, CBS_TABLES_T3
except ImportError:
    # Fallback defaults if config is missing
    DIR_DB_BRONZE = Path("data/1_bronze/bronze_data.db")
    DIR_DB_SILVER = Path("data/2_silver/silver_data.db")
    CBS_TABLES_T3 = ["80072ned"]  # Default fallback

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

    def _clean_value(self, value):
        """Standardization: Trim whitespace and handle empty strings as None."""
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
        return value

    def _infer_column_type(self, value):
        """Simple type inference for Silver schema."""
        if value is None:
            return String  # Default to String for NULLs if unknown
        if isinstance(value, int):
            return Integer
        if isinstance(value, float):
            return Float
        # Try to see if string is actually a number
        if isinstance(value, str):
            if value.isdigit():
                return Integer
            try:
                float(value)
                return Float
            except ValueError:
                pass
        return String

    def create_silver_table(self, identifier: str):
        """
        Joins Fact and Dimensions for a given identifier and creates a Silver table.
        """
        logger.info(f"Starting Silver transformation for identifier: {identifier}")

        # 1. Reflect Bronze Tables
        self.metadata_bronze.clear()
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
            # Heuristic: Dimension name suffix matches a column in Fact table.
            # e.g. '80072ned_dim_RegioS' -> 'RegioS'
            dim_suffix = dim_table.name.replace(dim_prefix, "")
            
            # Find matching column in Fact (handle potential spacing issues)
            match_col = None
            if dim_suffix in fact_table.c:
                match_col = fact_table.c[dim_suffix]
            else:
                # Try removing spaces from suffix to match column (e.g. 'Regio S' -> 'RegioS')
                normalized_suffix = dim_suffix.replace(" ", "")
                for col in fact_table.c:
                    if col.name == normalized_suffix:
                        match_col = col
                        break
            
            if match_col is not None:
                # Infer PK in Dim (Key, ID, Code)
                dim_pk_col = None
                for candidate in ["Key", "DimensionKey", "ID", "Code"]:
                    if candidate in dim_table.c:
                        dim_pk_col = dim_table.c[candidate]
                        break
                
                if dim_pk_col is not None:
                    # Left Join to preserve Fact records
                    query = query.join(dim_table, match_col == dim_pk_col, isouter=True)
                    
                    # Select descriptive columns from Dimension
                    for col in dim_table.c:
                        # Exclude keys and system columns
                        if col.name not in [dim_pk_col.name, "bronze_pk", "_source_file", "ID"]:
                            # Alias column to avoid collision
                            query = query.add_columns(col.label(f"{dim_suffix}_{col.name}"))
            else:
                logger.debug(f"Skipping join for {dim_table.name}: No matching FK column in Fact.")

        # 5. Execute Query
        with self.engine_bronze.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()
            keys = result.keys()
            
            if not rows:
                logger.warning(f"No data found for {identifier} after join.")
                return
            
            # 6. Process Data (Cleaning & Type Inference)
            cleaned_data = []
            
            # Clean all rows
            for row in rows:
                cleaned_row = {}
                for col_name, val in zip(keys, row):
                    cleaned_row[col_name] = self._clean_value(val)
                cleaned_data.append(cleaned_row)

            # Infer Schema
            silver_columns = []
            # Add a specific Silver Primary Key
            silver_columns.append(Column("silver_id", Integer, primary_key=True, autoincrement=True))

            for col_name in keys:
                inferred_type = String  # Default
                
                # Check first few rows for a non-null value to infer type
                for row in cleaned_data[:100]:
                    val = row[col_name]
                    if val is not None:
                        inferred_type = self._infer_column_type(val)
                        break
                
                # Avoid duplicate primary keys if 'ID' exists in source
                if col_name == "silver_id":
                    continue
                    
                silver_columns.append(Column(col_name, inferred_type))

            # 7. Create Silver Table
            silver_table_name = f"{identifier}_silver"
            
            # Drop if exists in metadata to avoid conflict
            if silver_table_name in self.metadata_silver.tables:
                self.metadata_silver.remove(self.metadata_silver.tables[silver_table_name])
                
            silver_table = Table(silver_table_name, self.metadata_silver, *silver_columns, extend_existing=True)
            
            silver_table.drop(self.engine_silver, checkfirst=True)
            silver_table.create(self.engine_silver)
            
            # 8. Bulk Insert
            with self.engine_silver.begin() as silver_conn:
                silver_conn.execute(insert(silver_table), cleaned_data)
                logger.info(f"Loaded {len(cleaned_data)} rows into {silver_table_name}")


# --- Main execution ---
db = DatabaseSilver(DIR_DB_BRONZE, DIR_DB_SILVER)
    
# Process tables defined in config
for table_id in CBS_TABLES_T3:
    db.create_silver_table(table_id)