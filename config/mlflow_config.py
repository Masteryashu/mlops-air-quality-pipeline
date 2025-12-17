"""
MLflow Configuration Module
Centralized configuration for MLflow tracking
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "air-quality-prediction"

# AWS S3 Configuration (for remote artifact store)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "")

# Model Registry Configuration
MODEL_NAME_XGBOOST = "air-quality-xgboost"
MODEL_NAME_RANDOM_FOREST = "air-quality-random-forest"
MODEL_NAME_NEURAL_NETWORK = "air-quality-neural-network"

def get_mlflow_config():
    """
    Get MLflow configuration dictionary
    """
    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "experiment_name": MLFLOW_EXPERIMENT_NAME,
        "s3_endpoint_url": MLFLOW_S3_ENDPOINT_URL,
    }

def print_mlflow_config():
    """
    Print current MLflow configuration
    """
    print("\n" + "=" * 80)
    print("MLFLOW CONFIGURATION")
    print("=" * 80)
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experiment Name: {MLFLOW_EXPERIMENT_NAME}")
    
    if MLFLOW_S3_ENDPOINT_URL:
        print(f"S3 Endpoint: {MLFLOW_S3_ENDPOINT_URL}")
        print(f"AWS Access Key: {'*' * 10 if AWS_ACCESS_KEY_ID else 'Not set'}")
    else:
        print("S3 Endpoint: Not configured (using local storage)")
    
    print("=" * 80 + "\n")

def setup_mlflow():
    """
    Setup MLflow with current configuration
    Returns configured mlflow module
    """
    import mlflow
    
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Set experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Configure AWS credentials if provided
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
        os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    
    if MLFLOW_S3_ENDPOINT_URL:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
    
    return mlflow
