"""
Step 6 — ML Model Validation (MAPE champion/challenger gate).

Decides promotion to the MLflow Model Registry and manages governance.
Gate policy (ADR-001 / ADR-002):

- There is ONE registered model per sector; its ``@prod`` alias is that
  sector's champion.
- A challenger is promoted to ``@prod`` if and only if its MAPE is finite AND
  strictly lower than the incumbent champion's MAPE.  If no champion exists
  yet, the challenger is registered unconditionally (seed).
- MAPE is the primary gate; R² is recorded but only gates when an optional
  ``r2_floor`` is supplied (disabled by default).
- Losers are NOT registered.  The run is tagged ``passed_gate=false`` and its
  ``ModelEvaluationRecord`` row is updated, so failed runs stay diagnosable.
- The promoted version carries ``mape`` / ``r2`` / ``model_family`` tags and a
  human-readable description, so the team can read each registered model's
  accuracy straight from the MLflow UI.
"""
import math
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Optional

from sqlalchemy.orm import Session

from src.ml_engineering.model_configs import ModelEvaluationRecord
from src.utils.m_log import f_log

_PROD_ALIAS = "prod"


class ModelValidator:
    """Champion/challenger gate + MLflow model lifecycle management."""

    def __init__(self):
        self.client = MlflowClient()

    def validate_and_register(
        self,
        run_id: str,
        registered_model_name: str,
        metrics: Dict[str, float],
        model_family: str,
        model_type: str = "",
        feature_groups: str = "",
        sector_code: str = "",
        r2_floor: Optional[float] = None,
        tags: Optional[dict] = None,
        description: str = "",
        session: Optional[Session] = None,
    ) -> bool:
        """Run the MAPE champion/challenger gate; register + promote if it wins.

        Args:
            run_id: MLflow run that produced the candidate.
            registered_model_name: Sector-keyed registry name (ADR-002).
            metrics: Must contain ``mape`` and ``r2`` (outer-fold honest values).
            model_family: Estimator family (e.g. ``RandomForest_Reduced``);
                stored as a version tag so the family survives the sector-only
                registry name.
            sector_code: Sector label, for logging.
            r2_floor: Optional secondary R² floor.  None disables it (default).
            tags: Extra version tags to attach on promotion.
            description: Optional version description; a MAPE/R²/family summary
                is generated when omitted.
            session: When supplied, the matching ``ModelEvaluationRecord``'s
                ``passed_gate`` field is updated.

        Returns:
            True if the candidate was promoted to ``@prod``, else False.
        """
        candidate_mape = metrics.get("mape", float("nan"))
        candidate_r2 = metrics.get("r2", float("nan"))
        champion_mape = self._champion_mape(registered_model_name)

        promote = self._is_winner(candidate_mape, candidate_r2, champion_mape, r2_floor)

        # Tag the run + persist the gate result regardless of outcome, so failed
        # runs remain filterable in the UI and queryable in SQL.
        self._safe_set_run_tag(run_id, "passed_gate", "true" if promote else "false")
        self._update_sql_gate(session, run_id, promote)

        if not promote:
            champ = "none" if champion_mape is None else f"{champion_mape:.4f}"
            f_log(
                f"Gate FAIL | {sector_code or registered_model_name} | "
                f"candidate MAPE={candidate_mape:.4f} vs champion {champ}",
                c_type="gate_fail",
            )
            return False

        version = self._register_and_promote(
            run_id, registered_model_name, model_family, model_type, feature_groups,
            candidate_mape, candidate_r2, tags, description,
        )
        f_log(
            f"Promoted | {registered_model_name} v{version} -> @{_PROD_ALIAS} | "
            f"MAPE={candidate_mape:.2%} R²={candidate_r2:.4f} family={model_family}",
            c_type="register",
        )
        return True

    # ── gate decision ────────────────────────────────────────────────────
    @staticmethod
    def _is_winner(
        candidate_mape: float,
        candidate_r2: float,
        champion_mape: Optional[float],
        r2_floor: Optional[float],
    ) -> bool:
        """Pure decision: finite MAPE, optional R² floor, then beat the champion."""
        if candidate_mape is None or not math.isfinite(candidate_mape):
            return False
        if r2_floor is not None and (
            candidate_r2 is None
            or not math.isfinite(candidate_r2)
            or candidate_r2 < r2_floor
        ):
            return False
        if champion_mape is None:
            return True  # seed: no incumbent to beat
        return candidate_mape < champion_mape

    def _champion_mape(self, registered_model_name: str) -> Optional[float]:
        """Read the incumbent ``@prod`` champion's MAPE tag, or None if absent."""
        try:
            mv = self.client.get_model_version_by_alias(registered_model_name, _PROD_ALIAS)
        except Exception:
            return None  # no registered model / no @prod alias yet
        raw = (getattr(mv, "tags", None) or {}).get("mape")
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    # ── registration / promotion ──────────────────────────────────────────
    def _register_and_promote(
        self,
        run_id: str,
        registered_model_name: str,
        model_family: str,
        model_type: str,
        feature_groups: str,
        mape: float,
        r2: float,
        tags: Optional[dict],
        description: str,
    ) -> str:
        """Register the run's model, stamp provenance tags, move ``@prod``.

        The version is made fully self-describing — model family, the underlying
        algorithm (model_type), the config feature groups used, and the metrics —
        so the registry is the single source of truth for downstream views.
        """
        model_version = mlflow.register_model(f"runs:/{run_id}/model", registered_model_name)
        version = model_version.version

        # Store MAPE/R² at full precision (repr round-trips exactly) so the
        # champion lookup compares the same value that produced the tag.
        # Rounding here would let an identical re-run falsely "beat" itself.
        version_tags = {
            "mape": repr(float(mape)),
            "r2": repr(float(r2)),
            "model_family": model_family,
            "model_type": model_type,
            "feature_groups": feature_groups,
            **(tags or {}),
        }
        for key, val in version_tags.items():
            self._safe_set_version_tag(registered_model_name, version, key, val)

        summary = description or f"MAPE={mape:.2%} R²={r2:.4f} family={model_family}"
        try:
            self.client.update_model_version(
                name=registered_model_name, version=version, description=summary,
            )
        except Exception as exc:
            f_log(f"update_model_version failed for {registered_model_name} v{version}: {exc}",
                  c_type="warning")

        self.client.set_registered_model_alias(
            name=registered_model_name, alias=_PROD_ALIAS, version=version,
        )
        return version

    def get_model_uri(self, registered_model_name: str, alias: str = _PROD_ALIAS) -> str:
        """Returns the URI for loading a registered model by alias."""
        return f"models:/{registered_model_name}/@{alias}"

    # ── side-effect helpers (best-effort; never abort the gate) ────────────
    def _safe_set_run_tag(self, run_id: str, key: str, value: str) -> None:
        try:
            self.client.set_tag(run_id, key, value)
        except Exception as exc:
            f_log(f"MLflow set_tag({key}) failed for {run_id}: {exc}", c_type="warning")

    def _safe_set_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        try:
            self.client.set_model_version_tag(name=name, version=version, key=key, value=value)
        except Exception as exc:
            f_log(f"set_model_version_tag({key}) failed for {name} v{version}: {exc}",
                  c_type="warning")

    def _update_sql_gate(self, session: Optional[Session], run_id: str, promote: bool) -> None:
        if session is None:
            return
        try:
            rec = session.get(ModelEvaluationRecord, run_id)
            if rec is not None:
                rec.passed_gate = 1 if promote else 0
                session.commit()
            else:
                f_log(f"No ModelEvaluationRecord for run_id={run_id} — "
                      f"passed_gate SQL update skipped.", c_type="warning")
        except Exception as exc:
            session.rollback()
            f_log(f"Failed to update passed_gate SQL field: {exc}", c_type="warning")
