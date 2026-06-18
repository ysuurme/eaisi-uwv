"""
evaluation_method.py — DEPRECATED shim.
=======================================
The evaluation/comparison logic moved to ``src/utils/m_evaluation.py`` (the
``m_`` naming convention; metric primitives now share ``ml_5``'s sklearn calls).

This module is retained ONLY so the standalone ``model_comparison.ipynb`` keeps
importing from ``evaluation_method`` without changes.  New code should import
from ``src.utils.m_evaluation`` directly.

In-framework comparison is now driven by ``ml_orchestrator.run_comparison``
(``python main.py --compare``), which reads the eval DB via
``m_pipeline_loader.load_families_from_eval_db`` instead of the parquet loaders.

Note: the old ``load_pipeline_predictions`` (which read a now-removed
``model_evaluation_records`` table) has been dropped — use
``m_pipeline_loader.load_pipeline_honest`` / ``load_families_from_eval_db``.
"""

from src.utils.m_evaluation import (  # noqa: F401  (re-exported for back-compat)
    CANONICAL_COLS,
    AlignmentReport,
    Scorecard,
    # metric primitives
    mae,
    rmse,
    mape,
    r2,
    bias,
    directional_accuracy,
    seasonal_naive_mae,
    mase,
    point_metrics,
    # per-method parquet loaders (legacy notebook)
    load_autoets_predictions,
    load_stl_ets_predictions,
    load_chronos_predictions,
    load_external_predictions,
    # alignment + metrics
    align_predictions,
    compute_point_metrics,
    compute_regime_metrics,
    compute_per_sector_metrics,
    compute_per_horizon_metrics,
    compute_skill_score,
    compute_pi_coverage,
    compute_crps_quantile_approx,
    # statistical tests + scorecard
    diebold_mariano_test,
    pairwise_dm_matrix,
    friedman_nemenyi,
    make_decision_matrix,
    compare_all_models,
)
