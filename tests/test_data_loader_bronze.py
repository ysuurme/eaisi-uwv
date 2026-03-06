import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.data_engineering.data_loader_bronze import DatabaseBronze

class TestDataLoaderBronze(unittest.TestCase):
    @patch("src.data_engineering.data_loader_bronze.create_engine")
    def setUp(self, mock_engine):
        self.raw_dir = Path("fake_raw")
        self.db_path = Path("fake_bronze.db")
        self.bronze = DatabaseBronze(self.raw_dir, self.db_path)

    @patch("src.data_engineering.data_loader_bronze.DatabaseBronze.insert_json_data")
    @patch("src.data_engineering.data_loader_bronze.Path.exists")
    @patch("src.data_engineering.data_loader_bronze.Path.glob")
    def test_ingest_0_raw_folder_success(self, mock_glob, mock_exists, mock_insert):
        mock_exists.return_value = True
        mock_glob.return_value = [Path("fake_raw/80072ned/TypedDataSet.json")]
        
        self.bronze.ingest_0_raw_folder("80072ned")
        
        mock_insert.assert_called_once()

if __name__ == "__main__":
    unittest.main()
