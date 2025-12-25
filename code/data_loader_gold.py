import logging
from pathlib import Path


# --- Third Party Libraries ---
from sqlalchemy import create_engine, MetaData


# --- Configuration ---
try:
    from config import DIR_DB_SILVER, DIR_DB_GOLD
except ImportError:
    raise ImportError("Configuration file 'config.py' not found or missing required variables.")


# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Gold Manager ---
class DatabaseGold:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.metadata = MetaData()

    def create_wide_table(self, silver_engine):
        """
        Flattens the Silver Star Schema into a single Gold table.
        """
        # Using a raw SQL Join for 'Wide' table creation efficiency
        query = """
        CREATE TABLE gold_ml_features AS
        SELECT 
            f.silver_id,
            f.periods,
            f.value as labour_value,
            d.title as region_name,
            f.bronze_id
        FROM fact_labour f
        LEFT JOIN dim_region d ON f.region_key = d.key
        """
        with self.engine.begin() as gold_conn:
            # We attach the silver database to the gold connection to join across SQLite .db files
            gold_conn.execute(f"ATTACH DATABASE '{silver_engine.url.database}' AS silver")
            gold_conn.execute("DROP TABLE IF EXISTS gold_ml_features")
            gold_conn.execute(query.replace("fact_labour", "silver.fact_labour").replace("dim_region", "silver.dim_region"))
            logger.info("Gold Wide Table 'gold_ml_features' created successfully.")