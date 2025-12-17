"""
FastAPI Application for Air Quality Prediction
Serves three models: XGBoost, Random Forest, and Neural Network
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from api.schemas import PredictionInput, PredictionOutput, ModelInfo
from config.mlflow_config import (
    setup_mlflow, 
    MODEL_NAME_XGBOOST, 
    MODEL_NAME_RANDOM_FOREST, 
    MODEL_NAME_NEURAL_NETWORK
)

# Initialize FastAPI app
app = FastAPI(
    title="Air Quality Prediction API",
    description="MLOps Final Project - Predict CO(GT) concentration using three ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup MLflow
mlflow_client = setup_mlflow()

# Global variables for models
models = {}
scaler = None

def load_models():
    """Load all three models from MLflow Model Registry"""
    global models, scaler
    
    try:
        # Load XGBoost model
        models['xgboost'] = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME_XGBOOST}/latest")
        print(f"✓ Loaded {MODEL_NAME_XGBOOST}")
        
        # Load Random Forest model
        models['random_forest'] = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME_RANDOM_FOREST}/latest")
        print(f"✓ Loaded {MODEL_NAME_RANDOM_FOREST}")
        
        # Load Neural Network model
        models['neural_network'] = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME_NEURAL_NETWORK}/latest")
        print(f"✓ Loaded {MODEL_NAME_NEURAL_NETWORK}")
        
        # Load scaler for neural network
        models_dir = Path(__file__).parent.parent / "models"
        scaler_path = models_dir / "neural_network_scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            print(f"✓ Loaded scaler for neural network")
        
        print("\n✓ All models loaded successfully!")
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        print("\nNote: Make sure to train all models first by running:")
        print("  uv run python scripts/05_train_xgboost.py")
        print("  uv run python scripts/06_train_random_forest.py")
        print("  uv run python scripts/07_train_neural_network.py")

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()

@app.get("/")
async def root():
    """Health check and API information"""
    return {
        "message": "Air Quality Prediction API",
        "version": "1.0.0",
        "status": "running",
        "models": {
            "xgboost": "available" if 'xgboost' in models else "not loaded",
            "random_forest": "available" if 'random_forest' in models else "not loaded",
            "neural_network": "available" if 'neural_network' in models else "not loaded"
        },
        "endpoints": [
            "/predict_model1 (XGBoost)",
            "/predict_model2 (Random Forest)",
            "/predict_model3 (Neural Network)",
            "/models/info"
        ]
    }

def prepare_input_data(input_data: PredictionInput) -> pd.DataFrame:
    """Convert input data to DataFrame with correct column order"""
    data = {
        'PT08_S1CO': [input_data.PT08_S1CO],
        'NMHCGT': [input_data.NMHCGT],
        'C6H6GT': [input_data.C6H6GT],
        'PT08_S2NMHC': [input_data.PT08_S2NMHC],
        'NOxGT': [input_data.NOxGT],
        'PT08_S3NOx': [input_data.PT08_S3NOx],
        'NO2GT': [input_data.NO2GT],
        'PT08_S4NO2': [input_data.PT08_S4NO2],
        'PT08_S5O3': [input_data.PT08_S5O3],
        'T': [input_data.T],
        'RH': [input_data.RH],
        'AH': [input_data.AH],
        'hour': [input_data.hour],
        'day_of_week': [input_data.day_of_week],
        'month': [input_data.month],
        'is_weekend': [input_data.is_weekend]
    }
    return pd.DataFrame(data)

@app.post("/predict_model1", response_model=PredictionOutput)
async def predict_xgboost(input_data: PredictionInput):
    """
    Predict CO(GT) concentration using XGBoost model
    """
    if 'xgboost' not in models:
        raise HTTPException(status_code=503, detail="XGBoost model not loaded")
    
    try:
        # Prepare input
        df = prepare_input_data(input_data)
        
        # Make prediction
        prediction = models['xgboost'].predict(df)[0]
        
        return PredictionOutput(
            prediction=float(prediction),
            model_name="XGBoost",
            model_version="latest",
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_model2", response_model=PredictionOutput)
async def predict_random_forest(input_data: PredictionInput):
    """
    Predict CO(GT) concentration using Random Forest model
    """
    if 'random_forest' not in models:
        raise HTTPException(status_code=503, detail="Random Forest model not loaded")
    
    try:
        # Prepare input
        df = prepare_input_data(input_data)
        
        # Make prediction
        prediction = models['random_forest'].predict(df)[0]
        
        return PredictionOutput(
            prediction=float(prediction),
            model_name="Random Forest",
            model_version="latest",
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_model3", response_model=PredictionOutput)
async def predict_neural_network(input_data: PredictionInput):
    """
    Predict CO(GT) concentration using Neural Network model
    """
    if 'neural_network' not in models:
        raise HTTPException(status_code=503, detail="Neural Network model not loaded")
    
    try:
        # Prepare input
        df = prepare_input_data(input_data)
        
        # Scale features for neural network
        if scaler is not None:
            df_scaled = scaler.transform(df)
            df = pd.DataFrame(df_scaled, columns=df.columns)
        
        # Make prediction
        prediction = models['neural_network'].predict(df)[0]
        
        return PredictionOutput(
            prediction=float(prediction),
            model_name="Neural Network",
            model_version="latest",
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/models/info")
async def get_models_info():
    """
    Get information about all loaded models
    """
    models_info = []
    
    if 'xgboost' in models:
        models_info.append(ModelInfo(
            model_name=MODEL_NAME_XGBOOST,
            model_type="XGBoost",
            version="latest",
            status="loaded"
        ))
    
    if 'random_forest' in models:
        models_info.append(ModelInfo(
            model_name=MODEL_NAME_RANDOM_FOREST,
            model_type="Random Forest",
            version="latest",
            status="loaded"
        ))
    
    if 'neural_network' in models:
        models_info.append(ModelInfo(
            model_name=MODEL_NAME_NEURAL_NETWORK,
            model_type="Neural Network (MLP)",
            version="latest",
            status="loaded"
        ))
    
    return {
        "total_models": len(models_info),
        "models": models_info
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
