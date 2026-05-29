from pathlib import Path

# config.py is in the src directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data Directories ---
DIR_DATA_RAW  = PROJECT_ROOT / "data" / "0_raw"
DIR_DB_BRONZE = PROJECT_ROOT / "data" / "1_bronze" / "bronze_data.db"
DIR_DB_SILVER = PROJECT_ROOT / "data" / "2_silver" / "silver_data.db"
DIR_DB_GOLD   = PROJECT_ROOT / "data" / "3_gold"   / "gold_data.db"
DIR_DB_EVAL   = PROJECT_ROOT / "data" / "4_eval" / "eval_data.db"
DIR_FEATURE_SELECTION = PROJECT_ROOT / "data" / "feature_selection"


# --- Toggle Flags ---
START_MLFLOW_UI = True # Set to True/False to auto-start MLflow UI in background

# --- Temporal Filter ---
# Structural break: the WIA law (2003) caused a significant regime shift in
# Dutch absenteeism data.  All data before this year is excluded from the
# gold layer to avoid training on a fundamentally different regime.
DATA_START_YEAR = 2003

# ═══════════════════════════════════════════════════════════════════════════
# CBS TABLE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════
# Central definition of every CBS table in the pipeline.
# Each entry maps a table ID to its metadata:
#   category  — driver category from the literature review
#   frequency — "quarterly" or "yearly"
#   lag       — publication lag in years (yearly tables only; 1 = standard)
#
# The gold loader processes ALL tables in the registry.
# Presets (below) select subsets for feature selection and model training.
# ═══════════════════════════════════════════════════════════════════════════

CBS_TABLE_REGISTRY: dict[str, dict] = {
    # ── Target ────────────────────────────────────────────────────────────
    "80072ned": {
        "category": "target",
        "frequency": "quarterly",
        "description": "Ziekteverzuimpercentage; bedrijfstakken (SBI 2008)",
    },

    # ── Labor Volume  ─────────────────────────────────────
    "85920NED": {
        "category": "labor_volume",
        "frequency": "quarterly",
        "description": "Arbeidsvolume; bedrijfstak, kwartalen, nationale rekeningen",
    },
    "85916NED": {
        "category": "labor_volume",
        "frequency": "quarterly",
        "description": "Beloning en arbeidsvolume van werknemers; kwartalen, nationale rekeningen",
    },

    # ── Working Conditions (WC)  ──────────────────────────────────
    "86009NED": {
        "category": "working_conditions",
        "frequency": "yearly",
        "lag": 1,
        "description": "Sick leave by industry and branch size (cause of absence)",
    },
    "85917NED": {
         "category": "working_conditions",
         "frequency": "quarterly",
         "lag": 1,
         "description": "Werkgelegenheid; geslacht, dienstverband, kenmerken baan, SBI2008",
    },

    # ── Wellbeing (WB) — yearly ───────────────────────────────────────────
    "85542NED": {
        "category": "wellbeing",
        "frequency": "yearly",
        "lag": 1,
        "description": "Welzijn; kerncijfers, persoonskenmerken",
    },

    # ── Labor Structure (LS) — quarterly + yearly ─────────────────────────
    "85278NED": {
         "category": "labor_structure",
         "frequency": "quarterly",
         "description": "Werkzame beroepsbevolking; positie in de werkkring (fixed vs flex)",
    },
    "80590ned": {
         "category": "labor_structure",
         "frequency": "quarterly",
         "description": "Arbeidsdeelname en werkloosheid per maand",
    },
    "81433ned": {
         "category": "labor_structure",
         "frequency": "yearly",
         "lag": 1,
         "description": "Werkgelegenheid; geslacht, dienstverband, kenmerken baan, SBI2008",
    },

    # ── Socio-Economic (SE) — yearly ──────────────────────────────────────
    # "85266NED": {
    #     "category": "socioeconomic",
    #     "frequency": "yearly",
    #     "lag": 1,
    #     "description": "Arbeidsdeelname; onderwijsniveau",
    # },

    # ── Health & Lifestyle (HL) — yearly ──────────────────────────────────
    # "81628NED": {
    #     "category": "health",
    #     "frequency": "yearly",
    #     "lag": 1,
    #     "description": "Gezondheid, leefstijl, zorggebruik; kerncijfers",
    # },
}

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT PRESETS
# ═══════════════════════════════════════════════════════════════════════════
# Each preset defines which driver categories to include.
# "basic" and "all" are always present.  Add incremental presets for the
# thesis comparison experiment (baseline → +conditions → +wellbeing → ...).
# ═══════════════════════════════════════════════════════════════════════════

CBS_PRESETS: dict[str, list[str]] = {
    "basic":              ["labor_volume"],
    "basic_conditions":   ["labor_volume", "working_conditions"],
    "basic_wellbeing":    ["labor_volume", "working_conditions", "wellbeing"],
    "basic_labor_struct": ["labor_volume", "working_conditions", "wellbeing", "labor_structure"],
    "basic_socio":        ["labor_volume", "working_conditions", "wellbeing", "labor_structure", "socioeconomic"],
    "all":                ["labor_volume", "working_conditions", "wellbeing", "labor_structure", "socioeconomic", "health"],
}

# ═══════════════════════════════════════════════════════════════════════════
# DERIVED CONSTANTS (backward-compatible with existing loaders)
# ═══════════════════════════════════════════════════════════════════════════

# All quarterly tables to process (including target)
CBS_TABLES_TO_LOAD = [
    tid for tid, meta in CBS_TABLE_REGISTRY.items()
    if meta["frequency"] == "quarterly"
]

# All yearly feature tables with their publication lag
CBS_TABLES_YEARLY: dict[str, int] = {
    tid: meta.get("lag", 1)
    for tid, meta in CBS_TABLE_REGISTRY.items()
    if meta["frequency"] == "yearly"
}

# Target table ID
CBS_TARGET_TABLE = next(
    tid for tid, meta in CBS_TABLE_REGISTRY.items()
    if meta["category"] == "target"
)


def get_tables_for_preset(preset_name: str) -> list[str]:
    """Return the list of CBS table IDs (excluding target) for a preset."""
    if preset_name not in CBS_PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(CBS_PRESETS.keys())}")
    categories = set(CBS_PRESETS[preset_name])
    return [
        tid for tid, meta in CBS_TABLE_REGISTRY.items()
        if meta["category"] in categories
    ]


def get_category_for_table(table_id: str) -> str | None:
    """Return the driver category for a CBS table ID."""
    meta = CBS_TABLE_REGISTRY.get(table_id)
    return meta["category"] if meta else None

# --- Visualization Palette ---
C_GREY    = "#6B7280"   # Dropped features
C_BLUE    = "#3B82F6"   # Retained features
C_ORANGE  = "#F59E0B"   # Threshold
C_BAND_50 = "#BFDBFE"   # darkest blue (50% interval)
C_BAND_80 = "#DBEAFE"   # mid blue
C_BAND_95 = "#EFF6FF"   # lightest blue
C_GRID    = "#E5E7EB"   # gridlines
C_TEXT    = "#111827"   # text

# --- Model Storage ---
DIR_MODELS = PROJECT_ROOT / "models"

# --- ML Target Column ---
ML_TARGET_COLUMN = "Ziekteverzuimpercentage_1"

# --- Logging ---
LOG_PROFILE = "PRD"          # "PRD" | "TEST" | "DEBUG"
LOG_SEPARATOR_WIDTH = 80
LOG_LINE_WIDTH = 120
DIR_LOG = PROJECT_ROOT / "log"
