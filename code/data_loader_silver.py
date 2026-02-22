"""
Data Loader for the Silver Layer.
Implements the Star Schema strategy by joining Fact and Dimension tables from Bronze.
"""
import logging
from pathlib import Path

# --- Third Party Libraries ---
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, select, insert

# --- Configuration ---
try:
    from config import DIR_DB_BRONZE, DIR_DB_SILVER, CBS_TABLES_T3
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
    Joins Fact and Dimension tables from bronze into a single wide table.
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


    def create_silver_table(self, identifier: str):
        """
        Joins Fact and Dimensions for a given identifier and creates a Silver table.
        """
        logger.info(f"Starting Silver transformation for CBS identifier: {identifier}")

        # 1. Identify Fact Table
        self.metadata_bronze.reflect(bind=self.engine_bronze)
        fact_table = self.metadata_bronze.tables.get(f"{identifier}_fact")
        
        if fact_table is None:
            logger.error(f"Fact table {identifier}_fact not found in Bronze Database.")
            return
        logger.info(f"Found fact table for {identifier}.")

        # 2. Identify Dimension Tables
        dim_tables = [t for n, t in self.metadata_bronze.tables.items() if n.startswith(f"{identifier}_dim_")]
        logger.info(f"Found {len(dim_tables)} dimension tables for {identifier}.")

        # 3. Build Query: Start with Fact Table
        query = select(fact_table)
        for dim_table in dim_tables:
            query = self._apply_dim_join(query, fact_table, dim_table, identifier)
            logger.debug(f"Query after join {str(query)}.") # Debug: print the query after each join
        
        # 4. Process and Save
        with self.engine_bronze.connect() as conn:
            data = conn.execute(query).fetchall()
            if data:
                self._save_to_silver(identifier, data)
            

    def _apply_dim_join(self, query, fact_table, dim_table, identifier):
            """Helper to find join keys and attach dimension columns to the query."""
            dim_prefix = f"{identifier}_dim_"
            dim_suffix = dim_table.name.replace(dim_prefix, "")
            
            # Determine the Foreign Key column in the Fact Table
            fk_col_fact = self._find_matching_column(fact_table, dim_suffix)
            # Determine the Foreign Key columns in the Dimension Table
            fk_col_dim = self._find_foreign_key(dim_table)

            if fk_col_fact is not None and fk_col_dim is not None:
                # Join and add descriptive columns (aliased to avoid collisions)
                query = query.join(dim_table, fk_col_fact == fk_col_dim, isouter=True)
                for col in dim_table.c:
                    if col.name not in [fk_col_dim.name, "bronze_pk", "_source_file"]:  # Exclude FK and metadata columns
                        query = query.add_columns(col.label(f"{dim_suffix}_{col.name}"))
            
            return query

    def _find_matching_column(self, table, column_name: str):
        """Matches dimension names to fact columns, handling spaces."""
        if column_name in table.c:
            return table.c[column_name]

    def _find_foreign_key(self, table):
        """Detects the Foreign Key column in a dimension table."""
        candidates = ["Key", "DimensionKey", "ID", "Code"]
        return next((table.c[c] for c in candidates if c in table.c), None)

    def _save_to_silver(self, identifier, rows):
        """Handles table creation and bulk insertion into Silver layer."""
        silver_table_name = f"{identifier}_silver"

        # Drop if exists in metadata to avoid conflict
        if silver_table_name in self.metadata_silver.tables:
            self.metadata_silver.remove(self.metadata_silver.tables[silver_table_name])

        # Simple schema: Use columns from the first row of results. In a real scenario, you'd use your _infer_column_type logic here
        cols = [Column("silver_id", Integer, primary_key=True, autoincrement=True)]
        for key in rows[0]._fields:
            if key != "silver_id":
                cols.append(Column(key, String)) # Simplifying to String for brevity

        # Create and Load
        silver_table = Table(silver_table_name, self.metadata_silver, *cols, extend_existing=True)
        silver_table.drop(self.engine_silver, checkfirst=True)
        silver_table.create(self.engine_silver)

        with self.engine_silver.begin() as conn:
            # Convert rows to dicts for insertion
            conn.execute(insert(silver_table), [row._asdict() for row in rows])
            logger.info(f"Successfully loaded {len(rows)} rows into {silver_table_name}")


# --- Main execution ---
db = DatabaseSilver(DIR_DB_BRONZE, DIR_DB_SILVER)
    
# Process tables defined in config
for table_id in CBS_TABLES_T3:
    db.create_silver_table(table_id)