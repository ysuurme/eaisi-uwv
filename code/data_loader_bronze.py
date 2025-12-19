import logging
import json
from pathlib import Path
from datetime import datetime

# --- Third Party Libraries ---
from sqlalchemy import create_engine, String
from sqlalchemy.orm import DeclarativeBase, Session, Mapped, mapped_column

# --- Configuration ---
DIR_DB_BRONZE = "data/1_bronze/bronze_data.db"
DIR_JSON_DATA = "data/0_raw/82070ENG/TypedDataSet.json"


# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# --- "Data Lake" 0_raw to Database 1_bronze ---
class DatabaseBronze:
    """
    Manages the SQLite database in the bronze layer (data/1_bronze) using SQLAlchemy.
    """
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.engine = create_engine(f"sqlite:///{self.db_path}")


    def create_table(self):
        """
        Creates tables based on the ORM models.
        Drops existing tables first to ensure schema is up to date.
        """
        # Drop existing table to ensure we have the correct schema
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        logger.info(f"Table '{TransactionORM.__tablename__}' created/reset in {self.db_path}")


    def insert_json_data(self, json_path: str):
        """
        Reads a JSON file, validates it against the TransactionORM,
        and inserts the data into the database.
        
        Args:
            json_path (str): The path to the JSON file.
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error reading JSON file {json_path}: {e}")
            return
            
        with Session(self.engine) as session:
            try:
                for item in data:
                    orm_object = TransactionORM(**item)
                    session.add(orm_object)
                session.commit()
                logger.info(f"Successfully inserted {len(data)} records from {json_path}")
            except Exception as e:
                session.rollback()
                logger.error(f"Error inserting data from {json_path}: {e}")
                raise


    def fetch_all(self, table_orm):
        """Fetches all rows from a table represented by an ORM class."""
        with Session(self.engine) as session:
            try:
                return session.query(table_orm).all()
            except Exception as e:
                logger.error(f"Failed to fetch data from '{table_orm.__tablename__}': {e}")
                return []


# --- Database Layer (SQLAlchemy) ---
class Base(DeclarativeBase):
    pass
    

class TransactionORM(Base):
    """Represents the SQL Table structure for TypedDataSet."""
    __tablename__ = "TypedDataSet"

    key: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ID: Mapped[int] = mapped_column(unique=True)
    Gender: Mapped[str] = mapped_column(String(255))
    PersonalCharacteristics: Mapped[str] = mapped_column(String(255))
    CaribbeanNetherlands: Mapped[str] = mapped_column(String(255))
    Periods: Mapped[str] = mapped_column(String(255))
    EmployedLabourForceInternatDef_1: Mapped[int] = mapped_column(nullable=True)
    EmployedLabourForceNationalDef_2: Mapped[int] = mapped_column(nullable=True)
    # insert_datetime: Mapped[datetime] = mapped_column(default=datetime.now)


# --- Main execution ---
if __name__ == "__main__":
    # Initialize
    db = DatabaseBronze(DIR_DB_BRONZE)

    # Create table
    db.create_table()
    
    # Insert data from JSON
    db.insert_json_data(DIR_JSON_DATA)

    # Verify
    print(f"\n--- Data in {TransactionORM.__tablename__} ---")
    all_data = db.fetch_all(TransactionORM)
    for row in all_data:
        print(row)