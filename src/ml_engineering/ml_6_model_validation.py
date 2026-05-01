"""
Step 6 — ML Model Validation.

Validates model quality against thresholds and manages MLflow registration.
Combines the quality gate decision with model governance:
- Checks metrics against the acceptance threshold
- Registers the model in the MLflow Model Registry
- Promotes to an alias (@prod, @staging)
"""
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Optional

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
    ) -> bool:
        """Checks quality gate. If passed, registers and promotes the model.

        Returns:
            True if model passed the gate and was registered, False otherwise.
        """
        passed = metrics["r2"] >= threshold_r2

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
