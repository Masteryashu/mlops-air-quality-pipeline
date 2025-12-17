"""
Script to promote Champion Model to Production in MLflow Registry
"""
import mlflow
from mlflow.tracking import MlflowClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config.mlflow_config import setup_mlflow, MODEL_NAME_RANDOM_FOREST

def promote_to_production():
    print("=" * 80)
    print("PROMOTING CHAMPION MODEL")
    print("=" * 80)
    
    # Setup MLflow
    try:
        setup_mlflow()
        client = MlflowClient()
    except Exception as e:
        print(f"✗ Failed to connect to MLflow: {e}")
        return

    model_name = MODEL_NAME_RANDOM_FOREST
    print(f"Target Model: {model_name}")
    
    # Get latest version
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            print(f"✗ No versions found for {model_name}")
            return
            
        # Sort by version number desc
        versions.sort(key=lambda x: int(x.version), reverse=True)
        latest_version = versions[0]
        
        print(f"Latest Version: {latest_version.version} (Current Stage: {latest_version.current_stage})")
        
        if latest_version.current_stage == "Production":
            print("✓ Already in Production")
            return
            
        # Transition to Production
        print(f"Promoting version {latest_version.version} to Production...")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"✓ Successfully transitioned {model_name} version {latest_version.version} to Production")
        
    except Exception as e:
        print(f"✗ Error during promotion: {e}")

if __name__ == "__main__":
    promote_to_production()
