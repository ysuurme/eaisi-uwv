"""
Data Loader for the Silver Layer.
Implements the Star Schema strategy by joining Fact and Dimension tables from Bronze.
"""
from pathlib import Path

# --- Third Party Libraries ---
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, select, insert

# --- Configuration ---
from src.config import DIR_DB_BRONZE, DIR_DB_SILVER, CBS_TABLES_T3, CBS_TABLES_T65

# --- Logging ---
from src.utils.m_log import f_log


class DatabaseSilver:
    """
    Manages the SQLite database in the silver layer.
    Joins Fact and Dimension tables from bronze into a single wide table.
    """
    def __init__(self, db_bronze_path: Path, db_silver_path: Path):
        self.db_bronze_path = db_bronze_path
        self.db_silver_path = db_silver_path
        
        if isinstance(self.db_silver_path, Path):
            self.db_silver_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.engine_bronze = create_engine(f"sqlite:///{self.db_bronze_path}")
        self.engine_silver = create_engine(f"sqlite:///{self.db_silver_path}")
        self.metadata_bronze = MetaData()
        self.metadata_silver = MetaData()


    def create_silver_table(self, identifier: str):
        """Joins Fact and Dimensions for a given identifier and creates a Silver table."""
        f_log(f"Starting Silver transformation for CBS identifier: {identifier}", c_type="process")

        # 1. Identify Fact Table
        self.metadata_bronze.reflect(bind=self.engine_bronze)
        fact_table = self.metadata_bronze.tables.get(f"{identifier}_fact")
        
        if fact_table is None:
            f_log(f"Fact table {identifier}_fact not found in Bronze Database.", c_type="error")
            return
        f_log(f"Found fact table for {identifier}.")

        # 2. Identify Dimension Tables
        dim_tables = [t for n, t in self.metadata_bronze.tables.items() if n.startswith(f"{identifier}_dim_")]
        f_log(f"Found {len(dim_tables)} dimension tables for {identifier}.")

        # 3. Build Query: Start with Fact Table
        query = select(fact_table)
        for dim_table in dim_tables:
            query = self._apply_dim_join(query, fact_table, dim_table, identifier)
            f_log(f"Query after join {str(query)}.", c_type="debug")
        
        # 4. Process and Save
        with self.engine_bronze.connect() as conn:
            data = conn.execute(query).fetchall()
            if data:
                self._save_to_silver(identifier, data)
            

    def _apply_dim_join(self, query, fact_table, dim_table, identifier):
            """Helper to find join keys and attach dimension columns to the query."""
            dim_prefix = f"{identifier}_dim_"
            dim_suffix = dim_table.name.replace(dim_prefix, "")
            
            fk_col_fact = self._find_matching_column(fact_table, dim_suffix)
            fk_col_dim = self._find_foreign_key(dim_table)

            if fk_col_fact is not None and fk_col_dim is not None:
                query = query.join(dim_table, fk_col_fact == fk_col_dim, isouter=True)
                for col in dim_table.c:
                    if col.name not in [fk_col_dim.name, "bronze_pk", "_source_file"]:
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

        # --- SILVER VALIDATION GATE (Soft Error) ---
        self._validate_silver_data(identifier, rows)

        if silver_table_name in self.metadata_silver.tables:
            self.metadata_silver.remove(self.metadata_silver.tables[silver_table_name])

        cols = [Column("silver_id", Integer, primary_key=True, autoincrement=True)]
        for key in rows[0]._fields:
            if key != "silver_id":
                cols.append(Column(key, String))

        silver_table = Table(silver_table_name, self.metadata_silver, *cols, extend_existing=True)
        silver_table.drop(self.engine_silver, checkfirst=True)
        silver_table.create(self.engine_silver)

        with self.engine_silver.begin() as conn:
            conn.execute(insert(silver_table), [row._asdict() for row in rows])
            f_log(f"Loaded {len(rows)} rows into {silver_table_name}", c_type="success")

    def _validate_silver_data(self, identifier: str, rows: list):
        """Scans rows for missing data in critical columns and logs a soft error (warning)."""
        critical_cols = ['Ziekteverzuimpercentage_1', 'Perioden']
        missing_stats = {col: 0 for col in critical_cols}
        
        for row in rows:
            row_dict = row._asdict()
            for col in critical_cols:
                if col in row_dict and (row_dict[col] is None or row_dict[col] == ''):
                    missing_stats[col] += 1

        for col, count in missing_stats.items():
            if count > 0:
                f_log(
                    f"SOFT ERROR: Missing data in Silver layer for '{identifier}'. "
                    f"Column '{col}' has {count} missing values. "
                    f"Gold layer expects complete data; check upstream Bronze/Raw sources.",
                    c_type="warning",
                )


if __name__ == "__main__":
    db = DatabaseSilver(DIR_DB_BRONZE, DIR_DB_SILVER)
    for table_id in CBS_TABLES_T65:
        db.create_silver_table(table_id)