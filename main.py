"""
Main Entry Point for EAISI UWV ML Pipeline.
Usage: python main.py <gold_table> <model_key>
"""
import logging
import sys
from src.ml_engineering.model_configs import ModelRegistry
from src.ml_engineering.model_orchestrator import ModelOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_ml_pipeline(gold_table: str, model_key: str):
    """
    Triggers the full ML lifecycle for a specific Gold table and estimator.
    """
    try:
        # Derive identifiers (e.g. "80072ned_gold" -> "80072ned")
        dataset_id = gold_table.replace("_gold", "")
        
        # Fetch configuration and initialise orchestrator
        config = ModelRegistry.get(model_key)
        
        orchestrator = ModelOrchestrator(
            experiment_name=f"{dataset_id}_SickLeave",
            model_name=f"{config.name}_{dataset_id}"
        )
        
        # Execute pipeline
        orchestrator.run_experiment(
            gold_table=gold_table,
            experiment_config=config,
            threshold_r2=0.0
        )
    except Exception as e:
        logger.error(f"❌ Pipeline failed for table '{gold_table}' with model '{model_key}': {e}")
        raise e

def main():
    # CLI Handling: python main.py <gold_table> <model_key>
    gold_table = sys.argv[1] if len(sys.argv) > 1 else "80072ned_gold"
    model_key = sys.argv[2] if len(sys.argv) > 2 else "random_forest"
    
    logger.info(f"🎯 Starting Pipeline | Table: {gold_table} | Model: {model_key}")
    
    try:
        run_ml_pipeline(gold_table, model_key)
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()
