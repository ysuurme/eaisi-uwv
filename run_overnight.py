"""
Overnight experiment runner.

Iterates over all (preset × model × sector) combinations, logging each
run to MLflow with a ``preset`` tag for filtering.

Usage:
    uv run python run_overnight.py                 # full sweep
    uv run python run_overnight.py --only baseline    # only the rolling-mean baseline
    uv run python run_overnight.py --only structural  # only the structural Ridge floor
    uv run python run_overnight.py --only presets     # only the preset × model sweep
    uv run python run_overnight.py --only all         # explicit full sweep (default)

The --only flag is intersected with the RUN_BASELINE / RUN_STRUCTURAL_BASELINE
constants: e.g. `--only baseline` still requires RUN_BASELINE=True to actually
fire the baseline block.

Prerequisites:
    1. Presets generated in data/feature_selection/preset_*.json
       (from the feature selection notebook)
    2. Model entries ("ridge", "elasticnet", "hist_gbr") registered
       in ModelConfiguration._CATALOG (model_configs.py)
    3. The orchestrator's run_pipeline must read os.environ["PRESET_NAME"]
       and add it as an MLflow tag (one-line change, see below)
"""
import argparse
import os
import time
from pathlib import Path

from src.config import DIR_FEATURE_SELECTION
from src.ml_engineering.ml_orchestrator import run_sector_sweep
from src.utils.m_log import f_log


# ── Configuration ─────────────────────────────────────────────────────────
MODELS = ["ridge", "elasticnet", "hist_gbr", "pls"]
PRESET_DIR = Path(DIR_FEATURE_SELECTION)
N_TEST_POINTS = 20
THRESHOLD_R2 = 0.0001      # log everything; filter quality in MLflow UI
RUN_BASELINE = False
# Structural-only Ridge: a diagnostic floor that uses ONLY the regime / temporal /
# trend features injected by ml_1's _KEEP_STRUCTURAL.  It ignores the preset's
# CBS feature set, so running it once (outside the preset loop) is sufficient.
# Tells you how much R² lives in the structural features alone, vs. what the
# preset-driven CBS feature engineering buys on top.
RUN_STRUCTURAL_BASELINE = False


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


def main(only: str = "all"):
    # ── Resolve which sections to run ──────────────────────────────────────
    do_baseline   = (only in ("baseline",   "all")) and RUN_BASELINE
    do_structural = (only in ("structural", "all")) and RUN_STRUCTURAL_BASELINE
    do_presets    = (only in ("presets",    "all"))

    # ── Discover presets (skip entirely if not needed) ────────────────────
    if do_presets:
        preset_files = sorted(PRESET_DIR.glob("preset_*.json"))
        if not preset_files:
            f_log(f"No preset files in {PRESET_DIR}", c_type="error")
            return
    else:
        preset_files = []

    n_presets = len(preset_files)
    # 39 sectors is approximate; actual count discovered per sweep
    n_outside_loop = int(do_baseline) + int(do_structural)
    est_runs = (n_outside_loop + (len(MODELS) * n_presets if do_presets else 0)) * 39
    f_log(
        f"Overnight run [only={only}]: {n_outside_loop} preset-independent + "
        f"{len(MODELS) if do_presets else 0} models × {n_presets} presets × "
        f"~39 sectors ≈ {est_runs} runs",
        c_type="start",
    )
    t_start = time.time()

    # ── Baseline (runs once — feature-independent) ────────────────────────
    if do_baseline:
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
        f_log(f"Baseline skipped (do_baseline=False)", c_type="process")

    # ── Structural-only diagnostic (runs once — preset-independent) ───────
    # Ridge on the structural feature set (trend_index, covid_period,
    # post_covid, year, quarter).  These features come from ml_1's
    # _KEEP_STRUCTURAL injection, not from any preset.  Running once is
    # sufficient; running per-preset would log identical predictions and
    # pollute the experiment.  Provides the diagnostic floor against which
    # the preset-driven models should improve.
    if do_structural:
        f_log("Running structural_linear diagnostic baseline...", c_type="process")
        os.environ["PRESET_NAME"] = "structural_only"
        try:
            run_sector_sweep(
                experiment_key="structural_linear",
                gold_table="master_data_ml_preprocessed",
                feature_groups=None,  # config.feature_groups=["structural_only"] wins
                n_test_points=N_TEST_POINTS,
                threshold_r2=THRESHOLD_R2,
            )
        except Exception as exc:
            f_log(f"Structural baseline failed: {exc}", c_type="error")
    else:
        f_log("Structural baseline skipped (do_structural=False)", c_type="process")

    # ── Model × Preset sweeps ────────────────────────────────────────────
    if do_presets:
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
    else:
        f_log("Preset × model sweeps skipped (do_presets=False)", c_type="process")

    total_min = (time.time() - t_start) / 60
    f_log(f"\nOvernight run complete in {total_min:.1f} minutes.", c_type="complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overnight experiment runner.")
    parser.add_argument(
        "--only",
        choices=["baseline", "structural", "presets", "all"],
        default="all",
        help=(
            "Which section(s) to run. "
            "'baseline' = SectorQuarterRollingMean sweep only "
            "(requires RUN_BASELINE=True); "
            "'structural' = structural_linear diagnostic only "
            "(requires RUN_STRUCTURAL_BASELINE=True); "
            "'presets' = preset × model loop only; "
            "'all' (default) = everything."
        ),
    )
    args = parser.parse_args()
    main(only=args.only)
