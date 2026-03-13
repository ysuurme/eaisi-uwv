import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from pathlib import Path
from src.data_engineering.data_loader_gold import DatabaseGold

class TestDataLoaderGold(unittest.TestCase):
    @patch("src.data_engineering.data_loader_gold.create_engine")
    def setUp(self, mock_engine):
        self.silver_path = Path("fake_silver.db")
        self.gold_path = Path("fake_gold.db")
        self.gold = DatabaseGold(self.silver_path, self.gold_path)

    @patch("src.data_engineering.data_loader_gold.pd.read_sql_query")
    @patch("src.data_engineering.data_loader_gold.pd.DataFrame.to_sql")
    @patch("src.data_engineering.data_loader_gold.text")
    def test_process_silver_table_success(self, mock_text, mock_to_sql, mock_read_sql):
        # Mocking data reading
        mock_read_sql.return_value = pd.DataFrame({"feat": [1]})
        
        # Mock transformation function
        mock_transform = MagicMock(return_value=pd.DataFrame({"feat_gold": [1]}))
        
        # Mock engine connection
        mock_conn = MagicMock()
        self.gold.engine.connect.return_value.__enter__.return_value = mock_conn

        self.gold.process_silver_table("80072ned", mock_transform)
        
        mock_transform.assert_called_once()
        mock_to_sql.assert_called_once()

if __name__ == "__main__":
    unittest.main()
