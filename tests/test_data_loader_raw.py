import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.data_engineering.data_loader_raw import CBSDataLoader

class TestDataLoaderRaw(unittest.TestCase):
    @patch("src.data_engineering.data_loader_raw.Path.mkdir")
    def setUp(self, mock_mkdir):
        self.output_dir = "fake_raw"
        self.loader = CBSDataLoader(self.output_dir)

    @patch("src.data_engineering.data_loader_raw.cbsodata.get_table_list")
    @patch("src.data_engineering.data_loader_raw.Path.exists")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_get_table_list_success(self, mock_file, mock_exists, mock_get_list):
        # Mocking: file doesn't exist, API returns one table
        mock_exists.return_value = False
        mock_get_list.return_value = [{"Identifier": "80072ned", "Title": "Test", "ShortDescription": "Desc"}]
        
        tables = self.loader.get_table_list()
        
        self.assertEqual(len(tables), 1)
        mock_get_list.assert_called_once()
        mock_file.assert_called()

    @patch("src.data_engineering.data_loader_raw.cbsodata.get_data")
    @patch("src.data_engineering.data_loader_raw.Path.exists")
    def test_get_table_skips_if_exists(self, mock_exists, mock_get_data):
        mock_exists.return_value = True
        
        self.loader.get_table("80072ned")
        
        # Should NOT call API if folder exists
        mock_get_data.assert_not_called()

if __name__ == "__main__":
    unittest.main()
