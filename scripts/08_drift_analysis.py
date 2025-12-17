"""
Model Drift Analysis Script using Evidently AI (Legacy API)
Detects data drift and performance drift
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Evidently Imports (Legacy API to support column_mapping)
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, RegressionPreset
from evidently.legacy.base_metric import ColumnMapping

import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config.mlflow_config import setup_mlflow, MODEL_NAME_RANDOM_FOREST, MODEL_NAME_XGBOOST

def run_evidently_analysis():
    print("=" * 80)
    print("EVIDENTLY DRIFT ANALYSIS")
    print("=" * 80)
    
    # Setup MLflow (Wrap to avoid connection crash)
    mlflow_client = None
    try:
        mlflow_client = setup_mlflow()
    except Exception as e:
        print(f"   ⚠ MLflow setup failed (Offline mode?): {e}")

    # Define paths
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    reports_dir = Path(__file__).parent.parent / "reports" / "drift_analysis"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading datasets...")
    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    
    train_df['datetime'] = pd.to_datetime(train_df['datetime'])
    test_df['datetime'] = pd.to_datetime(test_df['datetime'])
    
    # Prepare features
    exclude_cols = ['datetime', 'COGT']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    target_col = 'COGT'
    
    # Load Champion Model
    print("\n[2/5] Loading Champion Model...")
    model = None
    model_name = "Unknown"
    
    if mlflow_client:
        try:
            try:
                model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME_RANDOM_FOREST}/latest")
                model_name = "Random Forest"
            except:
                print("   ⚠ Random Forest load failed, trying XGBoost...")
                model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME_XGBOOST}/latest")
                model_name = "XGBoost"
            print(f"   ✓ Loaded {model_name}")
        except Exception as e:
            print(f"   ⚠ Could not load any model from MLflow. Performance drift will be skipped.\n   Error: {e}")
    else:
        print("   ⚠ Skipping model load (MLflow offline).")

    # Make predictions
    if model is not None:
        print(f"   Making predictions using {model_name}...")
        train_df['prediction'] = model.predict(train_df[feature_cols])
        test_df['prediction'] = model.predict(test_df[feature_cols])
    else:
        print("   ⚠ Using dummy predictions (0) for report generation.")
        train_df['prediction'] = 0
        test_df['prediction'] = 0

    # Define Column Mapping (Using Object for Legacy API)
    column_mapping = ColumnMapping()
    column_mapping.target = target_col
    column_mapping.prediction = 'prediction'
    column_mapping.numerical_features = feature_cols
    column_mapping.datetime = 'datetime'

    # Generate Reports
    print("\n[3/5] Generating Data Drift Report...")
    
    drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    drift_report.run(
        reference_data=train_df,
        current_data=test_df,
        column_mapping=column_mapping
    )
    
    drift_path = reports_dir / "evidently_data_drift.html"
    drift_report.save_html(str(drift_path))
    print(f"   ✓ Saved: {drift_path}")
    
    # Performance Report
    if model is not None:
        print("\n[4/5] Generating Performance Report...")
        perf_report = Report(metrics=[
            RegressionPreset(),
        ])
        
        perf_report.run(
            reference_data=train_df,
            current_data=test_df,
            column_mapping=column_mapping
        )
        
        perf_path = reports_dir / "evidently_performance_drift.html"
        perf_report.save_html(str(perf_path))
        print(f"   ✓ Saved: {perf_path}")
    else:
        print("\n[4/5] Skipping Performance Report (No Model).")
        
    print("\n" + "=" * 80)
    print("EVIDENTLY ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_evidently_analysis()
