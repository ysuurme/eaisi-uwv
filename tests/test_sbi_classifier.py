import unittest
from unittest.mock import patch, MagicMock
import json
import pandas as pd
from pathlib import Path

from src.utils.m_sbi_classifier import (
    _f_detect_sbi_format,
    _f_load_sbi_reference,
    _f_load_cbs_dimension_lookup,
    _f_ensure_sbi_reference,
    f_split_by_sbi,
)


# --- Sample data fixtures ---

SAMPLE_CBS_DIMENSION = [
    {"Key": "T001081", "Title": "A-U Alle economische activiteiten", "Description": "", "CategoryGroupID": 1},
    {"Key": "300003 ", "Title": "B-F Nijverheid en energie", "Description": "", "CategoryGroupID": 2},
    {"Key": "301000 ", "Title": "A Landbouw, bosbouw en visserij", "Description": "", "CategoryGroupID": 2},
    {"Key": "307500 ", "Title": "C Industrie", "Description": "", "CategoryGroupID": 3},
    {"Key": "307610 ", "Title": "10-12 Voedings-, genotmiddelenindustrie", "Description": "", "CategoryGroupID": 4},
    {"Key": "WP19078", "Title": "1 tot 10 werkzame personen", "Description": "", "CategoryGroupID": 5},
]

SAMPLE_JSONLD = {
    "@context": {},
    "@graph": [
        {
            "@id": "https://opendata.cbs.nl/id/sbi/01",
            "@type": "cbs:SBI",
            "rdfs:label": [
                {"@language": "nl", "@value": "Landbouw"},
                {"@language": "en", "@value": "Agriculture"},
            ],
            "schema:identifier": "01",
            "schema:nace": "01",
        },
        {
            "@id": "https://opendata.cbs.nl/id/sbi/45_2",
            "@type": "cbs:SBI",
            "rdfs:label": [
                {"@language": "nl", "@value": "Handel in en reparatie van auto's"},
                {"@language": "en", "@value": "Trade and repair of motor vehicles"},
            ],
            "schema:identifier": "45.2",
            "schema:nace": "45.2",
        },
        {
            "@id": "https://opendata.cbs.nl/id/sbi/69_10_1",
            "@type": "cbs:SBI",
            "rdfs:label": [
                {"@language": "nl", "@value": "Advocatenkantoren"},
                {"@language": "en", "@value": "Lawyers"},
            ],
            "schema:identifier": "69.10.1",
            "schema:nace": "69.10",
        },
        {
            "@id": "https://opendata.cbs.nl/id/sbi/86",
            "@type": "cbs:SBI",
            "rdfs:label": [
                {"@language": "nl", "@value": "Gezondheidszorg"},
                {"@language": "en", "@value": "Healthcare"},
            ],
            "schema:identifier": "86",
            "schema:nace": "86",
        },
        {
            "@id": "https://opendata.cbs.nl/id/sbi/45_20",
            "@type": "cbs:SBI",
            "rdfs:label": [
                {"@language": "nl", "@value": "Gespecialiseerde reparatie van auto's"},
                {"@language": "en", "@value": "Specialized repair of motor vehicles"},
            ],
            "schema:identifier": "45.20",
            "schema:nace": "45.20",
        },
    ],
}


class TestDetectSbiFormat(unittest.TestCase):
    """Tests for the auto-detection of SBI code formats."""

    def test_detect_numeric_sbi_codes(self):
        series = pd.Series(["01", "45.20", "69.10.1", "86"])
        result = _f_detect_sbi_format(series)
        self.assertEqual(result, "numeric")

    def test_detect_cbs_keys(self):
        series = pd.Series(["T001081", "307500 ", "WP19078", "301000 "])
        result = _f_detect_sbi_format(series)
        self.assertEqual(result, "cbs_key")

    def test_detect_empty_series_defaults_to_cbs_key(self):
        series = pd.Series([], dtype=str)
        result = _f_detect_sbi_format(series)
        self.assertEqual(result, "cbs_key")


class TestLoadSbiReference(unittest.TestCase):
    """Tests for loading and parsing the SBI 2008 linked data file."""

    def test_parses_jsonld_correctly(self):
        # Write sample jsonld to a temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonld", delete=False, encoding="utf-8") as f:
            json.dump(SAMPLE_JSONLD, f)
            tmp_path = Path(f.name)

        try:
            lookup = _f_load_sbi_reference(tmp_path)

            self.assertEqual(len(lookup), 5)
            self.assertIn("identifier", lookup.columns)
            self.assertIn("sbi_level", lookup.columns)
            self.assertIn("sbi_section_letter", lookup.columns)
            self.assertIn("label_nl", lookup.columns)
        finally:
            tmp_path.unlink()

    def test_level_classification_by_dot_count(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonld", delete=False, encoding="utf-8") as f:
            json.dump(SAMPLE_JSONLD, f)
            tmp_path = Path(f.name)

        try:
            lookup = _f_load_sbi_reference(tmp_path)
            levels = dict(zip(lookup["identifier"], lookup["sbi_level"]))

            self.assertEqual(levels["01"], "division")       # 0 dots
            self.assertEqual(levels["86"], "division")       # 0 dots
            self.assertEqual(levels["45.2"], "group")        # 1 dot
            self.assertEqual(levels["45.20"], "class")       # 2 dots
            self.assertEqual(levels["69.10.1"], "subclass")  # 3 dots
        finally:
            tmp_path.unlink()

    def test_section_letter_derived_from_division(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonld", delete=False, encoding="utf-8") as f:
            json.dump(SAMPLE_JSONLD, f)
            tmp_path = Path(f.name)

        try:
            lookup = _f_load_sbi_reference(tmp_path)
            sections = dict(zip(lookup["identifier"], lookup["sbi_section_letter"]))

            self.assertEqual(sections["01"], "A")   # Division 01 -> Section A
            self.assertEqual(sections["86"], "Q")   # Division 86 -> Section Q
            self.assertEqual(sections["45.2"], "G")  # Division 45 -> Section G
        finally:
            tmp_path.unlink()


class TestLoadCbsDimensionLookup(unittest.TestCase):
    """Tests for loading and classifying CBS dimension JSON entries."""

    def test_classifies_levels_by_title_regex(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(SAMPLE_CBS_DIMENSION, f)
            tmp_path = Path(f.name)

        try:
            lookup = _f_load_cbs_dimension_lookup(tmp_path)
            levels = dict(zip(lookup["Key"], lookup["sbi_level"]))

            self.assertEqual(levels["T001081"], "totaal")
            self.assertEqual(levels["300003"], "sector")       # trailing space stripped
            self.assertEqual(levels["301000"], "section")      # "A Landbouw"
            self.assertEqual(levels["307500"], "section")      # "C Industrie"
            self.assertEqual(levels["307610"], "subdivision")  # "10-12 ..."
            self.assertEqual(levels["WP19078"], "size")
        finally:
            tmp_path.unlink()

    def test_trailing_spaces_stripped_from_keys(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(SAMPLE_CBS_DIMENSION, f)
            tmp_path = Path(f.name)

        try:
            lookup = _f_load_cbs_dimension_lookup(tmp_path)
            # All keys should be stripped
            for key in lookup["Key"]:
                self.assertEqual(key, key.strip())
        finally:
            tmp_path.unlink()

    def test_section_letter_extracted(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(SAMPLE_CBS_DIMENSION, f)
            tmp_path = Path(f.name)

        try:
            lookup = _f_load_cbs_dimension_lookup(tmp_path)
            section_rows = lookup[lookup["sbi_level"] == "section"]
            letters = section_rows["sbi_section_letter"].tolist()

            self.assertIn("A", letters)
            self.assertIn("C", letters)
        finally:
            tmp_path.unlink()


class TestEnsureSbiReference(unittest.TestCase):
    """Tests for the download-and-cache helper."""

    def test_finds_existing_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a dummy file
            ref_path = Path(tmp_dir) / "SBI_2008_upd.2019.jsonld"
            ref_path.write_text("{}")

            result = _f_ensure_sbi_reference(Path(tmp_dir))
            self.assertEqual(result, ref_path)

    @patch("src.utils.m_sbi_classifier.urllib.request.urlretrieve")
    def test_downloads_when_missing(self, mock_urlretrieve):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            expected_path = Path(tmp_dir) / "SBI_2008_upd.2019.jsonld"

            result = _f_ensure_sbi_reference(Path(tmp_dir))

            mock_urlretrieve.assert_called_once()
            self.assertEqual(result, expected_path)


class TestSplitBySbiCbsKeys(unittest.TestCase):
    """Tests for f_split_by_sbi with CBS internal keys."""

    def _make_cbs_df(self):
        """Create a sample DataFrame with CBS internal keys."""
        return pd.DataFrame({
            "BedrijfskenmerkenSBI2008": [
                "T001081",   # totaal
                "300003 ",   # sector (trailing space)
                "301000 ",   # section A
                "307500 ",   # section C
                "307610 ",   # subdivision
                "WP19078",   # size
            ],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })

    def _write_cbs_dimension(self):
        """Write sample CBS dimension JSON to a temp file."""
        import tempfile
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(SAMPLE_CBS_DIMENSION, f)
        f.close()
        return Path(f.name)

    def test_split_produces_correct_levels(self):
        df = self._make_cbs_df()
        dim_path = self._write_cbs_dimension()

        try:
            result = f_split_by_sbi(df, dimension_json_path=dim_path)

            self.assertIn("df_totaal", result)
            self.assertIn("df_sector", result)
            self.assertIn("df_section", result)
            self.assertIn("df_subdivision", result)
            self.assertIn("df_size", result)
        finally:
            dim_path.unlink()

    def test_all_rows_accounted_for(self):
        df = self._make_cbs_df()
        dim_path = self._write_cbs_dimension()

        try:
            result = f_split_by_sbi(df, dimension_json_path=dim_path)
            total_rows = sum(len(v) for v in result.values())
            self.assertEqual(total_rows, len(df))
        finally:
            dim_path.unlink()

    def test_trailing_spaces_handled(self):
        df = self._make_cbs_df()
        dim_path = self._write_cbs_dimension()

        try:
            result = f_split_by_sbi(df, dimension_json_path=dim_path)
            # "300003 " should match after stripping
            self.assertIn("df_sector", result)
            self.assertEqual(len(result["df_sector"]), 1)
        finally:
            dim_path.unlink()

    def test_size_classes_included(self):
        df = self._make_cbs_df()
        dim_path = self._write_cbs_dimension()

        try:
            result = f_split_by_sbi(df, dimension_json_path=dim_path)
            self.assertIn("df_size", result)
            self.assertEqual(len(result["df_size"]), 1)
        finally:
            dim_path.unlink()


class TestSplitBySbiNumeric(unittest.TestCase):
    """Tests for f_split_by_sbi with numeric SBI codes."""

    def _make_numeric_df(self):
        return pd.DataFrame({
            "sbi_code": ["01", "86", "45.2", "45.20", "69.10.1"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

    @patch("src.utils.m_sbi_classifier._f_ensure_sbi_reference")
    def test_split_numeric_produces_correct_levels(self, mock_ensure):
        import tempfile
        # Write sample jsonld
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonld", delete=False, encoding="utf-8") as f:
            json.dump(SAMPLE_JSONLD, f)
            tmp_path = Path(f.name)

        mock_ensure.return_value = tmp_path

        try:
            df = self._make_numeric_df()
            result = f_split_by_sbi(df, sbi_column="sbi_code")

            # Should have division (01, 86), group (45.2), class (45.20), subclass (69.10.1)
            self.assertIn("df_division", result)
            self.assertIn("df_group", result)
            self.assertIn("df_class", result)
            self.assertIn("df_subclass", result)
        finally:
            tmp_path.unlink()


class TestSplitBySbiValidation(unittest.TestCase):
    """Tests for input validation and error handling."""

    def test_missing_column_raises_valueerror(self):
        df = pd.DataFrame({"other_column": [1, 2, 3]})
        with self.assertRaises(ValueError):
            f_split_by_sbi(df, sbi_column="sbi_code")

    def test_non_dataframe_raises_typeerror(self):
        with self.assertRaises(TypeError):
            f_split_by_sbi("not a dataframe")

    def test_empty_dataframe_returns_empty_dict(self):
        df = pd.DataFrame({"BedrijfskenmerkenSBI2008": pd.Series([], dtype=str)})
        result = f_split_by_sbi(df)
        self.assertEqual(result, {})

    def test_unmatched_rows_collected(self):
        df = pd.DataFrame({
            "BedrijfskenmerkenSBI2008": ["T001081", "UNKNOWN_KEY"],
            "value": [1.0, 2.0],
        })
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(SAMPLE_CBS_DIMENSION, f)
            dim_path = Path(f.name)

        try:
            result = f_split_by_sbi(df, dimension_json_path=dim_path, include_unmatched=True)
            self.assertIn("df__unmatched", result)
            self.assertEqual(len(result["df__unmatched"]), 1)
        finally:
            dim_path.unlink()


if __name__ == "__main__":
    unittest.main()
