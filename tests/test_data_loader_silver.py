import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from sqlalchemy import MetaData, Table, Column, String
from src.data_engineering.data_loader_silver import DatabaseSilver

class TestDataLoaderSilver(unittest.TestCase):
    @patch("src.data_engineering.data_loader_silver.create_engine")
    def setUp(self, mock_engine):
        self.bronze_path = Path("fake_bronze.db")
        self.silver_path = Path("fake_silver.db")
        self.silver = DatabaseSilver(self.bronze_path, self.silver_path)

    @patch("src.data_engineering.data_loader_silver.DatabaseSilver._save_to_silver")
    @patch("src.data_engineering.data_loader_silver.MetaData.reflect")
    def test_create_silver_table_success(self, mock_reflect, mock_save):
        # Create a REAL table object instead of MagicMock for SQLAlchemy select()
        metadata = MetaData()
        fact_table = Table(
            "80072ned_fact", metadata,
            Column("bronze_pk", String, primary_key=True),
            Column("Perioden", String)
        )
        self.silver.metadata_bronze.tables = {"80072ned_fact": fact_table}
        
        # Mock engine connection and result fetching
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_row = MagicMock()
        mock_row._asdict.return_value = {"bronze_pk": "1", "Perioden": "2023"}
        mock_row._fields = ["bronze_pk", "Perioden"]
        mock_result.fetchall.return_value = [mock_row]
        mock_conn.execute.return_value = mock_result
        self.silver.engine_bronze.connect.return_value.__enter__.return_value = mock_conn

        self.silver.create_silver_table("80072ned")
        
        mock_save.assert_called_once()

if __name__ == "__main__":
    unittest.main()
