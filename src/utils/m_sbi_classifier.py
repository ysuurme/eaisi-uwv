"""
Purpose:    Reusable utility to split a DataFrame by SBI (Standaard Bedrijfsindeling
            2008) categories using CBS Open Data dimension tables or the official
            SBI 2008 linked data taxonomy as reference.

            The SBI 2008 is the Dutch national standard for classifying businesses
            by their main economic activity. It is based on the EU NACE Rev. 2
            classification, which itself derives from the UN ISIC Rev. 4.

            CBS (Centraal Bureau voor de Statistiek) encodes SBI codes as opaque
            internal keys like "T001081" or "307500 " (note: trailing spaces).
            Other datasets may use actual SBI numeric codes like "01", "45.20",
            or "69.10.1". This utility handles both formats via auto-detection.

            SBI Hierarchy (numeric codes):
            - Section  : Letter A-U (derived from 2-digit division ranges)
            - Division : 2-digit code, e.g. "45" (Afdeling)
            - Group    : 3-digit code, e.g. "45.2" (Groep)
            - Class    : 4-digit code, e.g. "45.20" (Klasse)
            - Subclass : 5-digit code, e.g. "69.10.1" (Subklasse, NL-specific)

            CBS Hierarchy (internal keys):
            - Totaal      : Grand total ("A-U Alle economische activiteiten")
            - Sector      : Aggregates like "B-F Nijverheid en energie"
            - Section     : Individual letters A through U
            - Subdivision : Numeric sub-groups like "10-12", "45"
            - Size        : Business size classes (not SBI!), keys start with "WP"
"""

import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Literal

import pandas as pd

# --- Configuration ---
try:
    from config import DIR_DATA_RAW
except ImportError:
    DIR_DATA_RAW = None

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---

# URL to the official SBI 2008 linked data taxonomy (complete, ~500+ entries)
_SBI_REFERENCE_URL = (
    "https://raw.githubusercontent.com/KnowSyms/sbi-linked-data/"
    "master/SBI_2008_upd.2019.jsonld"
)
_SBI_REFERENCE_FILENAME = "SBI_2008_upd.2019.jsonld"

# Standard mapping from 2-digit SBI division codes to section letters (A-U).
# Source: CBS / Eurostat NACE Rev. 2
_DIVISION_TO_SECTION = {}
_section_ranges = {
    "A": range(1, 4),    "B": range(5, 10),   "C": range(10, 34),
    "D": range(35, 36),  "E": range(36, 40),  "F": range(41, 44),
    "G": range(45, 48),  "H": range(49, 54),  "I": range(55, 57),
    "J": range(58, 64),  "K": range(64, 67),  "L": range(68, 69),
    "M": range(69, 76),  "N": range(77, 83),  "O": range(84, 85),
    "P": range(85, 86),  "Q": range(86, 89),  "R": range(90, 94),
    "S": range(94, 97),  "T": range(97, 99),  "U": range(99, 100),
}
for _letter, _rng in _section_ranges.items():
    for _div in _rng:
        _DIVISION_TO_SECTION[_div] = _letter

# Regex patterns for classifying CBS dimension Title fields.
# These are stable across all CBS tables (unlike CategoryGroupID which varies).
_RE_TOTAAL = re.compile(r"^A-U\s")
_RE_SECTOR = re.compile(r"^[A-Z]-[A-Z]\s")
_RE_SECTION = re.compile(r"^([A-U])\s")
_RE_SUBDIVISION = re.compile(r"^\d")

# Regex for detecting numeric SBI codes (e.g. "01", "45.20", "69.10.1")
_RE_NUMERIC_SBI = re.compile(r"^\d{1,2}(\.\d+)*$")


# ---------------------------------------------------------------------------
# Helper functions (private)
# ---------------------------------------------------------------------------

def _f_ensure_sbi_reference(raw_dir: Path) -> Path:
    """
    Ensure the SBI 2008 reference file exists locally. Download if missing.

    This implements the 'download-and-cache' pattern: check if the file
    already exists before fetching it from the internet. This is idempotent
    — calling it multiple times has the same effect as calling it once.

    Parameters
    ----------
    raw_dir : Path
        Directory where raw data is stored (e.g. config.DIR_DATA_RAW).

    Returns
    -------
    Path
        Absolute path to the local SBI reference file.
    """
    raw_dir = Path(raw_dir)
    local_path = raw_dir / _SBI_REFERENCE_FILENAME

    if local_path.exists():
        logger.info(f"SBI reference file found at {local_path}")
        return local_path

    # Download from GitHub
    logger.info(f"SBI reference file not found. Downloading from {_SBI_REFERENCE_URL}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_SBI_REFERENCE_URL, local_path)
    logger.info(f"SBI reference file saved to {local_path}")
    return local_path


def _f_load_sbi_reference(jsonld_path: Path) -> pd.DataFrame:
    """
    Load the SBI 2008 linked data file and derive hierarchy classifications.

    Parses the JSON-LD ``@graph`` array and extracts:
    - ``identifier``: the SBI code (e.g. "01", "45.20", "69.10.1")
    - ``label_nl``: Dutch description
    - ``nace``: corresponding EU NACE code
    - ``sbi_level``: hierarchy level derived from dot count in identifier
    - ``sbi_section_letter``: the A-U section letter derived from the
      2-digit division prefix

    Parameters
    ----------
    jsonld_path : Path
        Path to the SBI_2008_upd.2019.jsonld file.

    Returns
    -------
    pd.DataFrame
        Lookup table with one row per SBI code.
    """
    with open(jsonld_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = data.get("@graph", [])
    records = []

    for entry in graph:
        identifier = entry.get("schema:identifier", "")
        if not identifier:
            continue

        # Extract Dutch label from the bilingual rdfs:label array
        labels = entry.get("rdfs:label", [])
        label_nl = ""
        for label in labels:
            if isinstance(label, dict) and label.get("@language") == "nl":
                label_nl = label.get("@value", "")
                break

        nace = entry.get("schema:nace", "")

        # Determine hierarchy level from identifier structure.
        # SBI codes use a dotted notation where the structure (not just dot count)
        # determines the level:
        #   "45"     → 1 part               → division  (Afdeling)
        #   "45.2"   → 2 parts, 1-char tail → group     (Groep)
        #   "45.20"  → 2 parts, 2-char tail → class     (Klasse)
        #   "69.10.1"→ 3 parts              → subclass  (Subklasse, NL-specific)
        parts = identifier.split(".")
        if len(parts) == 1:
            sbi_level = "division"
        elif len(parts) == 2:
            sbi_level = "group" if len(parts[1]) == 1 else "class"
        else:
            sbi_level = "subclass"

        # Derive section letter from the 2-digit division prefix
        # E.g. "45.20" -> division 45 -> section "G"
        try:
            division_num = int(identifier.split(".")[0])
            sbi_section_letter = _DIVISION_TO_SECTION.get(division_num, None)
        except ValueError:
            sbi_section_letter = None

        records.append({
            "identifier": identifier,
            "label_nl": label_nl,
            "nace": nace,
            "sbi_level": sbi_level,
            "sbi_section_letter": sbi_section_letter,
        })

    df_lookup = pd.DataFrame(records)
    logger.info(f"Loaded SBI reference: {len(df_lookup)} entries across levels "
                f"{df_lookup['sbi_level'].value_counts().to_dict()}")
    return df_lookup


def _f_load_cbs_dimension_lookup(dimension_json_path: Path) -> pd.DataFrame:
    """
    Load a CBS SBI dimension JSON file and classify entries by Title regex.

    CBS dimension tables use opaque internal keys (e.g. "T001081", "307500 ")
    rather than standard SBI codes. The Title field contains human-readable
    descriptions that reveal the hierarchy level. This function parses those
    titles using regex patterns that are stable across all CBS tables.

    Why Title parsing instead of CategoryGroupID?
    Because CategoryGroupID values are inconsistent: table 80072ned uses
    IDs 1-5, while 83157NED uses IDs 1-10. The Title format is the same
    across all tables.

    Parameters
    ----------
    dimension_json_path : Path
        Path to a CBS dimension JSON file (e.g. BedrijfskenmerkenSBI2008.json).

    Returns
    -------
    pd.DataFrame
        Lookup table with columns: Key, Title, sbi_level, sbi_section_letter.
    """
    with open(dimension_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df_lookup = pd.DataFrame(data)

    # CBS keys often have trailing whitespace — strip it for reliable matching
    df_lookup["Key"] = df_lookup["Key"].astype(str).str.strip()

    # Classify each entry by parsing the Title field
    levels = []
    section_letters = []

    for _, row in df_lookup.iterrows():
        key = row["Key"]
        title = str(row.get("Title", ""))

        # Check order matters: "WP" first, then "A-U" before general sector pattern
        if key.startswith("WP"):
            levels.append("size")
            section_letters.append(None)
        elif _RE_TOTAAL.search(title):
            levels.append("totaal")
            section_letters.append(None)
        elif _RE_SECTOR.search(title):
            levels.append("sector")
            section_letters.append(None)
        elif match := _RE_SECTION.match(title):
            levels.append("section")
            section_letters.append(match.group(1))
        elif _RE_SUBDIVISION.search(title):
            levels.append("subdivision")
            section_letters.append(None)
        else:
            levels.append("other")
            section_letters.append(None)

    df_lookup["sbi_level"] = levels
    df_lookup["sbi_section_letter"] = section_letters

    logger.info(f"Loaded CBS dimension lookup: {len(df_lookup)} entries, "
                f"levels: {df_lookup['sbi_level'].value_counts().to_dict()}")
    return df_lookup


def _f_detect_sbi_format(series: pd.Series) -> Literal["numeric", "cbs_key"]:
    """
    Auto-detect whether a Series contains numeric SBI codes or CBS internal keys.

    Samples up to 10 non-null values and checks if the majority match the
    numeric SBI pattern (digits with optional dots, e.g. "01", "45.20").

    Parameters
    ----------
    series : pd.Series
        The column to inspect.

    Returns
    -------
    str
        Either "numeric" or "cbs_key".
    """
    sample = series.dropna().head(10).astype(str).str.strip()
    if sample.empty:
        return "cbs_key"

    numeric_matches = sample.apply(lambda x: bool(_RE_NUMERIC_SBI.match(x)))
    ratio = numeric_matches.sum() / len(sample)

    detected = "numeric" if ratio > 0.5 else "cbs_key"
    logger.info(f"Auto-detected SBI format: '{detected}' "
                f"({numeric_matches.sum()}/{len(sample)} values matched numeric pattern)")
    return detected


# ---------------------------------------------------------------------------
# Main function (public)
# ---------------------------------------------------------------------------

def f_split_by_sbi(
    df: pd.DataFrame,
    sbi_column: str = "BedrijfskenmerkenSBI2008",
    dimension_json_path: "Path | str | None" = None,
    include_unmatched: bool = True,
) -> "dict[str, pd.DataFrame]":
    """
    Split a DataFrame into sub-DataFrames by SBI hierarchy level.

    Automatically detects whether the SBI column contains numeric SBI codes
    (e.g. "01", "45.20") or CBS internal keys (e.g. "T001081", "307500 ")
    and applies the appropriate classification strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a column with SBI codes or CBS keys.
    sbi_column : str, default "BedrijfskenmerkenSBI2008"
        Name of the column containing SBI codes or CBS keys.
    dimension_json_path : Path or str or None, default None
        Path to a CBS dimension JSON file. Only used when the column
        contains CBS internal keys. If None, the function searches
        ``config.DIR_DATA_RAW`` for a file matching ``{sbi_column}.json``.
    include_unmatched : bool, default True
        Whether to include rows that could not be matched to any SBI
        category. These are stored under the key ``"df__unmatched"``.

    Returns
    -------
    dict[str, pd.DataFrame]
        One entry per hierarchy level found. Keys follow the naming
        convention ``df_{level_name}``.

        For **numeric SBI codes**, possible keys:
        ``df_division``, ``df_group``, ``df_class``, ``df_subclass``,
        ``df_section`` (grouped by A-U letter).

        For **CBS internal keys**, possible keys:
        ``df_totaal``, ``df_sector``, ``df_section``, ``df_subdivision``,
        ``df_size``.

        Only levels that actually exist in the data appear in the output.

    Raises
    ------
    ValueError
        If ``sbi_column`` is not found in the DataFrame.
    FileNotFoundError
        If the required reference file cannot be found or downloaded.

    Examples
    --------
    Split CBS Silver data by SBI category:

    >>> from src.utils.m_sbi_classifier import f_split_by_sbi
    >>> splits = f_split_by_sbi(df_silver)
    >>> for name, sub_df in splits.items():
    ...     print(f"{name}: {len(sub_df)} rows")
    df_totaal: 116 rows
    df_sector: 348 rows
    df_section: 2204 rows
    df_subdivision: 1160 rows
    df_size: 348 rows

    Split a DataFrame with numeric SBI codes:

    >>> import pandas as pd
    >>> df = pd.DataFrame({"sbi_code": ["01", "45.20", "69.10.1"]})
    >>> splits = f_split_by_sbi(df, sbi_column="sbi_code")
    >>> list(splits.keys())
    ['df_division', 'df_subclass']
    """
    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")

    if sbi_column not in df.columns:
        raise ValueError(
            f"Column '{sbi_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if df.empty:
        logger.warning("Input DataFrame is empty. Returning empty dict.")
        return {}

    # --- Auto-detect format ---
    sbi_format = _f_detect_sbi_format(df[sbi_column])

    # --- Helper columns we'll add and later remove ---
    _helper_cols = ["_sbi_level", "_sbi_section_letter"]

    if sbi_format == "numeric":
        result = _split_numeric(df, sbi_column, include_unmatched, _helper_cols)
    else:
        result = _split_cbs_keys(df, sbi_column, dimension_json_path,
                                 include_unmatched, _helper_cols)

    # --- Log summary ---
    summary = ", ".join(f"{k} ({len(v)} rows)" for k, v in result.items())
    logger.info(f"Split into {len(result)} DataFrames: {summary}")

    return result


# ---------------------------------------------------------------------------
# Split strategies (private)
# ---------------------------------------------------------------------------

def _split_numeric(
    df: pd.DataFrame,
    sbi_column: str,
    include_unmatched: bool,
    helper_cols: list[str],
) -> "dict[str, pd.DataFrame]":
    """Split by numeric SBI codes using the jsonld reference."""

    # Ensure reference file exists
    if DIR_DATA_RAW is None:
        raise FileNotFoundError(
            "config.DIR_DATA_RAW is not available. Cannot locate SBI reference file. "
            "Ensure config.py is accessible or pass dimension_json_path explicitly."
        )

    jsonld_path = _f_ensure_sbi_reference(Path(DIR_DATA_RAW))
    lookup = _f_load_sbi_reference(jsonld_path)

    # Prepare merge: strip whitespace from input column
    df_work = df.copy()
    df_work["_sbi_key_stripped"] = df_work[sbi_column].astype(str).str.strip()

    # Left merge to get level info
    df_work = df_work.merge(
        lookup[["identifier", "sbi_level", "sbi_section_letter"]],
        left_on="_sbi_key_stripped",
        right_on="identifier",
        how="left",
    )
    df_work.rename(columns={
        "sbi_level": "_sbi_level",
        "sbi_section_letter": "_sbi_section_letter",
    }, inplace=True)

    # Build result dict — one DataFrame per level
    result = {}
    cols_to_drop = helper_cols + ["_sbi_key_stripped", "identifier"]

    # Split by hierarchy level (division, group, class, subclass)
    matched = df_work[df_work["_sbi_level"].notna()]
    for level_name, group_df in matched.groupby("_sbi_level"):
        clean_df = group_df.drop(columns=cols_to_drop, errors="ignore").reset_index(drop=True)
        result[f"df_{level_name}"] = clean_df

    # Also create section-level split (grouping divisions by A-U letter)
    divisions_with_section = matched[
        (matched["_sbi_level"] == "division") & (matched["_sbi_section_letter"].notna())
    ]
    if not divisions_with_section.empty:
        section_frames = {}
        for letter, group_df in divisions_with_section.groupby("_sbi_section_letter"):
            clean_df = group_df.drop(columns=cols_to_drop, errors="ignore").reset_index(drop=True)
            section_frames[letter] = clean_df
        if section_frames:
            # Combine all section sub-frames into one df_section with all divisions grouped
            result["df_section"] = pd.concat(section_frames.values(), ignore_index=True)

    # Handle unmatched
    unmatched = df_work[df_work["_sbi_level"].isna()]
    if not unmatched.empty:
        if include_unmatched:
            clean_df = unmatched.drop(columns=cols_to_drop, errors="ignore").reset_index(drop=True)
            result["df__unmatched"] = clean_df
        logger.warning(
            f"{len(unmatched)} rows could not be matched to any SBI code. "
            f"Sample values: {unmatched[sbi_column].head(5).tolist()}"
        )

    return result


def _split_cbs_keys(
    df: pd.DataFrame,
    sbi_column: str,
    dimension_json_path: "Path | str | None",
    include_unmatched: bool,
    helper_cols: list[str],
) -> "dict[str, pd.DataFrame]":
    """Split by CBS internal keys using a CBS dimension JSON."""

    # Resolve dimension JSON path
    if dimension_json_path is not None:
        dim_path = Path(dimension_json_path)
    else:
        # Auto-detect: search config.DIR_DATA_RAW for {sbi_column}.json
        if DIR_DATA_RAW is None:
            raise FileNotFoundError(
                "config.DIR_DATA_RAW is not available and no dimension_json_path "
                "was provided. Ensure config.py is accessible or pass the path explicitly."
            )
        candidates = list(Path(DIR_DATA_RAW).rglob(f"{sbi_column}.json"))
        if len(candidates) == 1:
            dim_path = candidates[0]
        elif len(candidates) > 1:
            raise ValueError(
                f"Multiple CBS dimension files found for '{sbi_column}': {candidates}. "
                f"Pass dimension_json_path explicitly to disambiguate."
            )
        else:
            raise FileNotFoundError(
                f"No '{sbi_column}.json' found under {DIR_DATA_RAW}. "
                f"Ensure the raw data has been downloaded first."
            )

    logger.info(f"Using CBS dimension file: {dim_path}")
    lookup = _f_load_cbs_dimension_lookup(dim_path)

    # Prepare merge: strip whitespace from input column
    df_work = df.copy()
    df_work["_sbi_key_stripped"] = df_work[sbi_column].astype(str).str.strip()

    # Left merge to get level info
    df_work = df_work.merge(
        lookup[["Key", "sbi_level", "sbi_section_letter"]],
        left_on="_sbi_key_stripped",
        right_on="Key",
        how="left",
    )
    df_work.rename(columns={
        "sbi_level": "_sbi_level",
        "sbi_section_letter": "_sbi_section_letter",
    }, inplace=True)

    # Build result dict — one DataFrame per level
    result = {}
    cols_to_drop = helper_cols + ["_sbi_key_stripped", "Key"]

    matched = df_work[df_work["_sbi_level"].notna()]
    for level_name, group_df in matched.groupby("_sbi_level"):
        clean_df = group_df.drop(columns=cols_to_drop, errors="ignore").reset_index(drop=True)
        result[f"df_{level_name}"] = clean_df

    # Handle unmatched
    unmatched = df_work[df_work["_sbi_level"].isna()]
    if not unmatched.empty:
        if include_unmatched:
            clean_df = unmatched.drop(columns=cols_to_drop, errors="ignore").reset_index(drop=True)
            result["df__unmatched"] = clean_df
        logger.warning(
            f"{len(unmatched)} rows could not be matched to any CBS dimension key. "
            f"Sample values: {unmatched[sbi_column].head(5).tolist()}"
        )

    return result


# --- Main execution ---
if __name__ == "__main__":
    # Quick demo: load Silver data and split by SBI category
    from src.utils.m_query_database import f_query_database
    from config import DIR_DB_SILVER

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    df = f_query_database(DIR_DB_SILVER, 'SELECT * FROM "80072ned_silver"', "pandas")
    splits = f_split_by_sbi(df)

    print(f"\nSplit into {len(splits)} DataFrames:")
    for name, sub_df in splits.items():
        print(f"  {name}: {len(sub_df)} rows")

    total_rows = sum(len(v) for v in splits.values())
    print(f"\nTotal rows across all splits: {total_rows} (input: {len(df)})")
