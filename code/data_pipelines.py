import os
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

# --- Third Party Libraries ---
from pydantic import BaseModel, ValidationError, Field
from sqlalchemy import create_engine, String, Text
from sqlalchemy.orm import DeclarativeBase, Session, mapped_column, Mapped
from sqlalchemy.exc import SQLAlchemyError


# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

DB_FILE = "sqlite:///data//1_bronze//bronze_data.db"
DATA_DIR = "data//0_raw//TypedDataSet.json"

# --- Data Validation Layer (Pydantic) ---
class TypedDataSchema(BaseModel):
    key: int
    id: int
    gender: str
    personal_characteristics: str
    caribbean_netherlands: str
    periods: str
    employed_labour_def_1: int
    employed_labour_def_2: int
    insert_datetime: Optional[datetime] = Field(default_factory=datetime.now)

# --- Database Layer (SQLAlchemy) ---
class Base(DeclarativeBase):
    pass
    

class TransactionORM(Base):
    """Represents the SQL Table structure."""
    __tablename__ = "TypedDataSet"

    key: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    id: Mapped[int] = mapped_column(unique=True)
    gender: Mapped[str] = mapped_column(String(255))
    personal_characteristics: Mapped[str] = mapped_column(String(255))
    caribbean_netherlands: Mapped[str] = mapped_column(String(255))
    periods: Mapped[str] = mapped_column(String(255))
    employed_labour_def_1: Mapped[int] = mapped_column(nullable=True)
    employed_labour_def_2: Mapped[int] = mapped_column(nullable=True)
    insert_datetime: Mapped[datetime] = mapped_column(default=datetime.now)

    # Helper to create ORM object from Pydantic model
    @classmethod
    def from_schema(cls, schema: TypedDataSchema):
        return cls(**schema.model_dump())

# --- The Pipeline Class ---
class DataIngestionPipeline:
    def __init__(self, db_url: str, source_dir: str):
        self.source_dir = Path(source_dir)
        self.engine = create_engine(db_url)
        
        # Ensure DB tables exist
        Base.metadata.create_all(self.engine)
        logger.info("Database initialized.")

    def process_directory(self):
        """Orchestrates the retrieval and storage of data."""
        if not self.source_dir.exists():
            logger.error(f"Directory {self.source_dir} not found.")
            return

        logger.info(f"Found {self.source_dir} JSON files to process.")

        with Session(self.engine) as session:
            self._process_single_file(session, self.source_dir)

    def _process_single_file(self, session, file_path: Path):
        """Reads, Validates, and Loads a single file."""
        try:
            with open(file_path, "r") as f:
                raw_data = json.load(f)

            # A. VALIDATION (Pydantic)
            # If JSON is bad, this raises ValidationError and skips DB insertion
            validated_data = TypedDataSchema(**raw_data)

            # B. TRANSFORMATION & LOADING (SQLAlchemy)
            # Convert validated Pydantic model to SQLAlchemy ORM object
            orm_record = TransactionORM.from_schema(validated_data)
            
            # Merge handles "insert or update" if primary key exists, 
            # though here we rely on unique transaction_id constraints.
            session.add(orm_record)
            
            # Commit per file or batch (here per file for safety)
            session.commit()
            logger.info(f"Successfully ingested: {file_path.name}")
            
            # Optional: Move file to 'processed' folder here

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON: {file_path.name}")
        except ValidationError as e:
            logger.error(f"Validation failed for {file_path.name}: {e}")
        except SQLAlchemyError as e:
            session.rollback() # Important! Rollback on DB errors
            logger.error(f"Database error for {file_path.name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path.name}: {e}")


# --- Main Execution ---
# Initialize Pipeline
pipeline = DataIngestionPipeline(
    db_url=DB_FILE,
    source_dir=DATA_DIR
)

# Run
pipeline.process_directory()

print("\nCheck the logs above to see how invalid data was rejected and valid data stored.")