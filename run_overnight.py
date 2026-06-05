"""
Overnight experiment runner.

Iterates over all (preset × model × sector) combinations, logging each
run to MLflow with a ``preset`` tag for filtering.

Usage:
    uv run python run_overnight.py

Prerequisites:
    1. Presets generated in data/feature_selection/preset_*.json
       (from the feature selection notebook)
    2. Model entries ("ridge", "elasticnet", "hist_gbr") registered
       in ModelConfiguration._CATALOG (model_configs.py)
    3. The orchestrator's run_pipeline must read os.environ["PRESET_NAME"]
       and add it as an MLflow tag (one-line change, see below)
"""
import os
import time
from pathlib import Path

from src.config import DIR_FEATURE_SELECTION
from src.ml_engineering.ml_orchestrator import run_sector_sweep
from src.utils.m_log import f_log


# ── Configuration ─────────────────────────────────────────────────────────
MODELS = ["ridge", "elasticnet", "hist_gbr"]
PRESET_DIR = Path(DIR_FEATURE_SELECTION)
N_TEST_POINTS = 20
THRESHOLD_R2 = 0.0      # log everything; filter quality in MLflow UI
RUN_BASELINE = False 


def _load_preset_inplace(preset_path: Path) -> int:
    """Load a preset JSON and update FEATURE_CATALOG in-place.

    CRITICAL: must use .clear() + .update() — NOT rebinding.
    ml_1_data_extraction.py imports FEATURE_CATALOG at module level with
    ``from model_configs import FEATURE_CATALOG``.  Rebinding the module
    attribute (``mc.FEATURE_CATALOG = new_dict``) creates a new dict that
    DataExtractor never sees.  In-place mutation ensures all references
    across the codebase point to the same (updated) dict.

    Returns the number of features in the loaded catalog.
    """
    import src.ml_engineering.model_configs as mc
    new_catalog = mc.load_feature_catalog(preset_path)
    mc.FEATURE_CATALOG.clear()
    mc.FEATURE_CATALOG.update(new_catalog)

    n_features = sum(
        len(g.columns) for g in new_catalog.values() if hasattr(g, "columns")
    )
    return n_features


def main():
    # ── Discover presets ──────────────────────────────────────────────────
    preset_files = sorted(PRESET_DIR.glob("preset_*.json"))
    if not preset_files:
        f_log(f"No preset files in {PRESET_DIR}", c_type="error")
        return

    n_presets = len(preset_files)
    # 39 sectors is approximate; actual count discovered per sweep
    est_runs = (1 + len(MODELS) * n_presets) * 39
    f_log(
        f"Overnight run: 1 baseline + {len(MODELS)} models × "
        f"{n_presets} presets × ~39 sectors ≈ {est_runs} runs",
        c_type="start",
    )
    t_start = time.time()

    # ── Baseline (runs once — feature-independent) ────────────────────────
    if RUN_BASELINE:
        f_log("Running baseline (SectorQuarterRollingMean)...", c_type="process")
        os.environ["PRESET_NAME"] = "baseline"
        try:
            run_sector_sweep(
                experiment_key="baseline",
                gold_table="master_data_ml_preprocessed",
                feature_groups=None,
                n_test_points=N_TEST_POINTS,
                threshold_r2=THRESHOLD_R2,
            )
        except Exception as exc:
            f_log(f"Baseline failed: {exc}", c_type="error")
    else:
        f_log("Baseline skipped (RUN_BASELINE = False)", c_type="process")

    # ── Model × Preset sweeps ────────────────────────────────────────────
    for i_preset, preset_path in enumerate(preset_files, 1):
        preset_name = preset_path.stem.replace("preset_", "")

        n_features = _load_preset_inplace(preset_path)
        f_log(
            f"\n{'='*60}\n"
            f"PRESET {i_preset}/{n_presets}: {preset_name} ({n_features} features)\n"
            f"{'='*60}",
            c_type="start",
        )
        os.environ["PRESET_NAME"] = preset_name

        for model_key in MODELS:
            f_log(f"  Model: {model_key}", c_type="process")
            t_model = time.time()

            try:
                run_sector_sweep(
                    experiment_key=model_key,
                    gold_table="master_data_ml_preprocessed",
                    feature_groups=["all_survivors"],
                    n_test_points=N_TEST_POINTS,
                    threshold_r2=THRESHOLD_R2,
                )
                elapsed = time.time() - t_model
                f_log(
                    f"  ✅ {model_key} complete ({elapsed / 60:.1f} min)",
                    c_type="success",
                )
            except Exception as exc:
                f_log(f"  ❌ {preset_name}/{model_key}: {exc}", c_type="error")
                continue

    total_min = (time.time() - t_start) / 60
    f_log(f"\nOvernight run complete in {total_min:.1f} minutes.", c_type="complete")


if __name__ == "__main__":
    main()
