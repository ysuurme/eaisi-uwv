"""
Model Registry for the ML Layer.
Responsible for:
- Programmatic model registration via MlflowClient.
- Managing Aliases (@prod, @staging).
- Metadata enrichment (Tags, Descriptions).
- Retrieving models using the 2026 URI standard.
"""
import mlflow
from mlflow.tracking import MlflowClient

# --- Logging ---
from src.utils.m_log import f_log


class ModelRegistry:
    """Handles model lifecycle and governance in MLflow."""

    def __init__(self):
        self.client = MlflowClient()

    def register_candidate(
        self,
        run_id: str,
        model_name: str,
        tags: dict = None,
        description: str = ""
    ) -> str:
        """Registers a model version from a run."""
        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(model_uri, model_name)
        
        if tags:
            for key, val in tags.items():
                self.client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=key,
                    value=val
                )
        
        if description:
            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
            
        return model_version.version

    def promote_to_alias(self, model_name: str, version: str, alias: str = "prod"):
        """Promotes a specific model version to an alias (e.g., @prod)."""
        self.client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=version
        )


    def get_model_uri(self, model_name: str, alias: str = "prod") -> str:
        """Returns the centralized URI for a given model and alias."""
        return f"models:/{model_name}/@{alias}"

    def get_latest_version(self, model_name: str) -> str:
        """Fetches the latest version number for a model."""
        versions = self.client.get_latest_versions(model_name)
        return versions[0].version if versions else None

if __name__ == "__main__":
    from src.utils.m_log import setup_logging
    setup_logging()
