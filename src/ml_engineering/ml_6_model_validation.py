"""
Step 6 — ML Model Validation.

Validates model quality against thresholds and manages MLflow registration.
Combines the quality gate decision with model governance:
- Tags every run with passed_gate (so filtering — not deletion — is the
  UI filter strategy; failed runs remain queryable for diagnostics)
- Persists the gate result to the ModelEvaluationRecord SQL row
- Checks metrics against the acceptance threshold
- Registers the model in the MLflow Model Registry
- Promotes to an alias (@prod, @staging)
"""
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Optional

from sqlalchemy.orm import Session

from src.ml_engineering.model_configs import ModelEvaluationRecord
from src.utils.m_log import f_log


class ModelValidator:
    """Validates model quality and manages MLflow model lifecycle."""

    def __init__(self):
        self.client = MlflowClient()

    def validate_and_register(
        self,
        run_id: str,
        model_name: str,
        metrics: Dict[str, float],
        threshold_r2: float = 0.5,
        tags: Optional[dict] = None,
        description: str = "",
        session: Optional[Session] = None,
    ) -> bool:
        """Checks quality gate. If passed, registers and promotes the model.

        Side effects regardless of pass/fail (so failed runs remain
        diagnosable rather than deleted):
        - Sets MLflow tag ``passed_gate=true|false`` on the run.
        - When a ``session`` is supplied, updates the corresponding
          ``ModelEvaluationRecord.passed_gate`` SQL field.

        Returns:
            True if the model passed the gate and was registered, False otherwise.
        """
        passed = metrics["r2"] >= threshold_r2

        # ── MLflow tag (always, regardless of pass/fail) ──────────────────
        # Use MlflowClient because the Step 4 mlflow.start_run() context has
        # already closed by the time Step 6 runs.  This keeps failed runs in
        # the database, filterable in the UI via tags.passed_gate = "true".
        try:
            self.client.set_tag(run_id, "passed_gate", "true" if passed else "false")
        except Exception as exc:
            f_log(f"MLflow set_tag(passed_gate) failed for {run_id}: {exc}", c_type="warning")

        # ── SQL update (when session is provided) ─────────────────────────
        # Overwrites the placeholder 0 written by Step 5.  Failures here are
        # non-fatal — the MLflow tag remains the authoritative gate marker.
        if session is not None:
            try:
                rec = session.get(ModelEvaluationRecord, run_id)
                if rec is not None:
                    rec.passed_gate = 1 if passed else 0
                    session.commit()
                else:
                    f_log(
                        f"No ModelEvaluationRecord for run_id={run_id} — "
                        f"passed_gate SQL update skipped.",
                        c_type="warning",
                    )
            except Exception as exc:
                session.rollback()
                f_log(f"Failed to update passed_gate SQL field: {exc}", c_type="warning")

        if not passed:
            f_log(
                f"Gate FAIL | R2: {metrics['r2']:.4f} < threshold {threshold_r2}",
                c_type="gate_fail",
            )
            return False

        f_log(f"Gate PASS | R2: {metrics['r2']:.4f} >= threshold {threshold_r2}", c_type="success")

        version = self._register_model(run_id, model_name, tags, description)
        self._promote_to_alias(model_name, version, alias="prod")

        f_log(f"Model registered | {model_name} v{version} -> @prod", c_type="register")
        return True

    def _register_model(
        self, run_id: str, model_name: str,
        tags: Optional[dict], description: str,
    ) -> str:
        """Registers a model version from a completed run."""
        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(model_uri, model_name)

        if tags:
            for key, val in tags.items():
                self.client.set_model_version_tag(
                    name=model_name, version=model_version.version, key=key, value=val,
                )
        if description:
            self.client.update_model_version(
                name=model_name, version=model_version.version, description=description,
            )
        return model_version.version

    def _promote_to_alias(self, model_name: str, version: str, alias: str = "prod") -> None:
        """Promotes a specific model version to an alias (e.g., @prod)."""
        self.client.set_registered_model_alias(name=model_name, alias=alias, version=version)

    def get_model_uri(self, model_name: str, alias: str = "prod") -> str:
        """Returns the URI for loading a registered model by alias."""
        return f"models:/{model_name}/@{alias}"
