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
# Central definition of every CBS table in the pipeline and the SINGLE SOURCE
# OF TRUTH for which tables are processed.  Every listed table is active — to
# disable a table, remove its entry, do NOT comment it out.  Frequency drives
# how each table is reduced to one row per quarter.
# Each entry maps a table ID to its metadata:
#   category  — driver category from the literature review
#   frequency — "quarterly" | "yearly" | "monthly"
#   lag       — publication lag in years (yearly tables only; 1 = standard)
#   agg       — quarterly aggregation method (monthly tables only; default "mean")
#
# The gold loader processes ALL tables in the registry; the frequency-derived
# lists below route each table.  Presets select category subsets for feature
# selection and model training, and are validated/derived from this registry.
# ═══════════════════════════════════════════════════════════════════════════

CBS_TABLE_REGISTRY: dict[str, dict] = {
    # ── Target ────────────────────────────────────────────────────────────
    # Ziekteverzuimpercentage; bedrijfstakken (SBI 2008) en bedrijfsgrootte: data available from 1996 > onwards
    "80072ned": {
        "category": "target",
        "frequency": "quarterly",
        "description": "Ziekteverzuimpercentage; bedrijfstakken (SBI 2008)",
    },

    # ── Labor Volume (LV) — quarterly ─────────────────────────────────────
    # Arbeidsvolume; bedrijfstak, kwartalen, nationale rekeningen : data available from 1995 > onwards
    "85920NED": {
        "category": "labor_volume",
        "frequency": "quarterly",
        "description": "Arbeidsvolume; bedrijfstak, kwartalen, nationale rekeningen",
    },

    # ── Wages & Compensation (WG) ─────────────────────────────────────────
    # Beloning en arbeidsvolume van werknemers; kwartalen, nationale rekeningen: data available from 1995 > onwards
    "85917NED": {
        "category": "wages",
        "frequency": "quarterly",
        "description": "Beloning en arbeidsvolume van werknemers; kwartalen",
    },

    # ── Working Conditions (WC) — yearly ──────────────────────────────────
    # Ziekteverzuim volgens werknemers; bedrijfstak en vestigingsgrootte: data available from 2014 onwards
    "86009NED": {
        "category": "working_conditions",
        "frequency": "yearly",
        "lag": 1,
        "description": "Sick leave by industry and branch size (cause of absence)",
    },

    # ── Wellbeing (WB) — yearly ───────────────────────────────────────────
    # Welzijn; kerncijfers, persoonskenmerken: data available from 2013 onwards
    "85542NED": {
        "category": "wellbeing",
        "frequency": "yearly",
        "lag": 1,
        "description": "Welzijn; kerncijfers, persoonskenmerken",
    },

    # ── Labor Structure (LS) — quarterly ──────────────────────────────────
    # Werkzame beroepsbevolking; positie in de werkkring (fixed vs flex): data available from 2013 onwards
    "85278NED": {
        "category": "labor_structure",
        "frequency": "quarterly",
        "description": "Werkzame beroepsbevolking; positie in de werkkring (fixed vs flex)",
    },
    # -- Arbeidsdeelname en werkloosheid per maand: data available from 2003 > onwards
    "80590ned": {
        "category": "labor_structure",
        "frequency": "quarterly",
        "description": "Arbeidsdeelname en werkloosheid per maand",
    },
    # Vacatures; SBI 2008; naar economische activiteit en bedrijfsgrootte: data available from 1997 > onwards
    "80472ned": {
        "category": "labor_structure",
        "frequency": "quarterly",
        "description": "Arbeidsdeelname en werkloosheid per maand",
    },
    # ── Socio-Economic (SE) — monthly ─────────────────────────────────────
    # Consumentenvertrouwen, economisch klimaat en koopbereidheid; gecorrigeerd, data available from 1986 > onwards.
    # Aggregation: "mean" — CCI is a level/index variable, average over the 3 months in each quarter.
    "83693NED": {
         "category": "socioeconomic",
         "frequency": "monthly",
         "agg": "mean",
         "description": "Consumentenvertrouwen",
     },  
    
    # ── Socio-Economic (SE) — yearly ──────────────────────────────────────
    # Arbeidsdeelname; onderwijsniveau: data available from 2003 onwards
    "85266NED": {
        "category": "socioeconomic",
        "frequency": "yearly",
        "lag": 1,
        "description": "Arbeidsdeelname; onderwijsniveau",
    },

    # ── Health & Lifestyle (HL) — yearly ──────────────────────────────────
    # Gezondheid, leefstijl, zorggebruik; kerncijfers: data available from 2014 onwards
    "81628NED": {
        "category": "health",
        "frequency": "yearly",
        "lag": 1,
        "description": "Gezondheid, leefstijl, zorggebruik; kerncijfers",
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT PRESETS
# ═══════════════════════════════════════════════════════════════════════════
# Each preset defines which driver categories to include.
# "basic" and "all" are always present.  Add incremental presets for the
# thesis comparison experiment (baseline → +conditions → +wellbeing → ...).
# The "all" preset is DERIVED from the registry so it can never drift; named
# presets are validated against the registry via ``validate_presets()``.
# ═══════════════════════════════════════════════════════════════════════════

# Feature driver categories present in the registry (everything except the
# target), derived so presets and the "all" set stay in sync with the registry.
FEATURE_CATEGORIES: list[str] = sorted({
    meta["category"] for meta in CBS_TABLE_REGISTRY.values()
    if meta["category"] != "target"
})

CBS_PRESETS: dict[str, list[str]] = {
    "basic":              ["labor_volume"],
    "basic_wages":        ["labor_volume", "wages"],
    "basic_conditions":   ["labor_volume", "wages", "working_conditions"],
    "basic_wellbeing":    ["labor_volume", "wages", "working_conditions", "wellbeing"],
    "basic_macro":        ["labor_volume", "wages", "socioeconomic"],
    # Registry-derived: always covers every feature category present.
    "all":                list(FEATURE_CATEGORIES),
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

# All monthly feature tables with their quarterly aggregation method.
# Default "mean" is appropriate for level/index/rate variables (CCI, CPI,
# unemployment rate); override per-table with `"agg": "sum"` for flows
# (new vacancies posted), or `"agg": "last"` for end-of-period stocks.
CBS_TABLES_MONTHLY: dict[str, str] = {
    tid: meta.get("agg", "mean")
    for tid, meta in CBS_TABLE_REGISTRY.items()
    if meta["frequency"] == "monthly"
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


def get_feature_categories() -> list[str]:
    """All driver categories in the registry except the target (registry-derived)."""
    return list(FEATURE_CATEGORIES)


def validate_presets() -> dict[str, list[str]]:
    """Return ``{preset: [categories not present in the registry]}``.

    Empty dict means every preset references only categories that actually exist
    in ``CBS_TABLE_REGISTRY`` — i.e. presets and registry are in sync.
    """
    known = set(FEATURE_CATEGORIES)
    return {
        name: sorted(set(cats) - known)
        for name, cats in CBS_PRESETS.items()
        if set(cats) - known
    }

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════
# Configuration for the feature-selection flow (``main.py --select-features``
# → ``ml_orchestrator.run_feature_selection``).  Candidate features are
# restricted to columns originating from CBS tables whose registry frequency
# is listed below.  Yearly tables are excluded by default: their quarterly
# representation is a disaggregation artifact (replication/interpolation)
# whose transformation choices would otherwise leak into model training.
# Monthly tables are true intra-quarter observations aggregated to quarters
# and therefore eligible.
FEATURE_SELECTION_FREQUENCIES: tuple[str, ...] = ("quarterly", "monthly")

# Sequential funnel parameters (validated in the notebook feature study).
# Each filter receives the survivors of the previous one; the full chain is
# persisted in the generated feature_catalog.json for lineage.
FEATURE_SELECTION_FUNNEL: dict[str, dict] = {
    "near_constant":      {"max_fraction": 0.95},
    "correlation":        {"threshold": 0.21, "method": "within_sector_differenced"},
    "lagged_correlation": {"threshold": 0.12, "horizons": [1, 2, 3]},
    "granger":            {"max_lag": 4, "p_threshold": 0.05,
                           "min_sector_fraction": 0.20, "difference": True},
    "lasso_stability":    {"n_bootstraps": 100, "threshold": 0.60,
                           "random_state": 42, "horizons": [1, 2, 3]},
    "redundancy":         {"threshold": 0.90},
}

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
