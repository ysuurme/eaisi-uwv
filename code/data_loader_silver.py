import logging
from pathlib import Path
from typing import List
from datetime import datetime


# --- Third Party Libraries ---
from sqlalchemy import create_engine, MetaData, String, ForeignKey, Integer, Float, DateTime
from sqlalchemy.orm import mapped_column, relationship, DeclarativeBase, Mapped, Session


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


# --- Schema definitions from Database 1_bronze to Database 2_silver (Declarative ORM) ---
class Base(DeclarativeBase):
    """ Common meta for all SilverORM models """
    pass


class DimRegion(Base):
    __tablename__ = "dim_region"
    key: Mapped[str] = mapped_column(String(50), primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(String(1000), nullable=True)


class FactLabour(Base):
    __tablename__ = "fact_labour"
    
    # Primary Key for Silver (Clean, internal ID)
    silver_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign Key linking to DimRegion
    region_key: Mapped[str] = mapped_column(ForeignKey("dim_region.key"))
    
    # Data Columns
    periods: Mapped[str] = mapped_column(String(20))
    employed_internat: Mapped[float] = mapped_column(Float, nullable=True)
    employed_national: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Metadata for Data Lineage
    source_bronze_id: Mapped[str] = mapped_column(String(255))
    processed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    # Relationships (Allows for easy .join() in queries)
    region: Mapped["DimRegion"] = relationship()


class SampleTable(Base):
    __tablename__ = "sample_table"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50))
    value: Mapped[float] = mapped_column(Float)


# --- The Database 1_bronze to Database 2_silver Manager Class ---
class DatabaseSilver:
    """
    Manages the SQLite database in the Silver layer using SQLAlchemy ORM (strict/typed data).
    Consumer data from SQLite bronze tables and populates structured ORM tables into SQLite silver tables.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.metadata = MetaData()

    def init_database(self):
        """Standardized table creation pattern."""
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        logger.info(f"Silver DB initialized at {self.db_path}")

    def load_from_bronze(self, bronze_data: List[dict], table_type: str):
        """
        Transformation logic: Maps Bronze dicts to Silver ORM objects.
        This is where 'Data Cleaning' happens.
        """
        with Session(self.engine) as session:
            try:
                for row in bronze_data:
                    if table_type == "fact":
                        # Mapping Bronze keys to Silver naming conventions
                        obj = FactLabour(
                            region_key=row.get("CaribbeanNetherlands"),
                            periods=row.get("Periods"),
                            employed_internat=row.get("EmployedLabourForceInternatDef_1"),
                            employed_national=row.get("EmployedLabourForceNationalDef_2"),
                            source_bronze_id=row.get("bronze_id")
                        )
                    elif table_type == "dim_region":
                        obj = DimRegion(
                            key=row.get("Key"),
                            title=row.get("Title"),
                            description=row.get("Description")
                        )
                    
                    session.add(obj)
                
                session.commit()
                logger.info(f"Successfully moved {len(bronze_data)} records to Silver {table_type}")
            except Exception as e:
                session.rollback()
                logger.error(f"Silver load failed: {e}")
                raise


# --- Main execution ---
if __name__ == "__main__":
    db = DatabaseSilver(DIR_DB_SILVER)
    db.init_database()

    # Add sample data
    with Session(db.engine) as session:
        sample1 = SampleTable(name="Test A", value=10.5)
        sample2 = SampleTable(name="Test B", value=20.0)
        session.add_all([sample1, sample2])
        session.commit()
        logger.info("Sample table populated with dummy data.")