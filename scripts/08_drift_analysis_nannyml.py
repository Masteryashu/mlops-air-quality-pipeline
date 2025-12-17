"""
Model Drift Analysis Script using NannyML
Estimates performance (CBPE) and detects data drift
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime
import nannyml as nml
import mlflow

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config.mlflow_config import setup_mlflow, MODEL_NAME_RANDOM_FOREST, MODEL_NAME_XGBOOST

def run_nannyml_analysis():
    """
    Run NannyML analysis on production data
    """
    print("=" * 80)
    print("NANNYML DRIFT ANALYSIS")
    print("=" * 80)
    
    # Setup MLflow
    mlflow_client = setup_mlflow()
    
    # Define paths
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    reports_dir = Path(__file__).parent.parent / "reports" / "nannyml"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading datasets...")
    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    
    # Parse datetime
    train_df['datetime'] = pd.to_datetime(train_df['datetime'])
    test_df['datetime'] = pd.to_datetime(test_df['datetime'])
    
    print(f"   Reference (Train): {train_df.shape}")
    print(f"   Analysis (Test): {test_df.shape}")
    
    # Prepare features
    exclude_cols = ['datetime', 'COGT']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    target_col = 'COGT'
    
    # Load Champion Model (Random Forest)
    print("\n[2/5] Loading Champion Model (Random Forest)...")
    try:
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME_RANDOM_FOREST}/latest")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        # Try XGBoost as fallback
        print("   ⚠ Fallback to XGBoost...")
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME_XGBOOST}/latest")
        
    # Make predictions
    print("   Making predictions...")
    
    # NannyML needs a single DataFrame with predictions for reference and analysis typically,
    # or separate. We'll add predictions to the DataFrames.
    
    train_df['y_pred'] = model.predict(train_df[feature_cols])
    test_df['y_pred'] = model.predict(test_df[feature_cols])
    
    # Initialize NannyML estimators
    print("\n[3/5] Initializing NannyML estimators...")
    
    # 1. Performance Estimation (DLE) - Regression
    # Direct Loss Estimation for Regression
    
    estimator = nml.DLE(
        y_pred='y_pred',
        y_true=target_col,
        timestamp_column_name='datetime',
        metrics=['rmse', 'mae'],
        chunk_size=500
    )
    
    # 2. Univariate Drift
    drift_calc = nml.UnivariateDriftCalculator(
        column_names=feature_cols,
        timestamp_column_name='datetime',
        chunk_size=500,
        continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
        categorical_methods=['chi2', 'jensen_shannon'],
    )
    
    # Fit on Reference (Train)
    print("   Fitting estimators on Reference data (Train)...")
    estimator.fit(train_df)
    drift_calc.fit(train_df)
    
    # Estimate/Calculate on Analysis (Test)
    print("\n[4/5] Calculating drift on Analysis data (Test)...")
    
    # For CBPE, we don't strictly need y_true in analysis, but we provide it for realized performance comparison if supported.
    # NannyML separates 'estimation' from 'performance calculation'.
    # estimate() produces ESTIMATED metrics.
    
    estimated_perf = estimator.estimate(test_df)
    drift_results = drift_calc.calculate(test_df)
    
    # Calculate Realized Performance (since we have ground truth)
    calculator = nml.PerformanceCalculator(
        y_pred='y_pred',
        y_true=target_col,
        timestamp_column_name='datetime',
        metrics=['rmse', 'mae'],
        chunk_size=500,
        problem_type='regression'
    )
    calculator.fit(train_df)
    realized_perf = calculator.calculate(test_df)
    
    print("   ✓ Calculations complete")
    
    # Generate and Save Reports
    print("\n[5/5] Generating reports...")
    
    # 1. Performance Estimation Plot
    print(f"   Saving Performance Estimation Plot...")
    fig_est = estimated_perf.plot()
    fig_est.write_html(str(reports_dir / "performance_estimation.html"))
    
    # 2. Realized Performance Plot
    print(f"   Saving Realized Performance Plot...")
    fig_real = realized_perf.plot()
    fig_real.write_html(str(reports_dir / "realized_performance.html"))
    
    # 3. Comparison Plot (Estimated vs Realized)
    # NannyML allows creating comparison if metrics match
    print(f"   Saving Comparison Plot...")
    try:
        fig_comp = estimated_perf.compare(realized_perf).plot()
        fig_comp.write_html(str(reports_dir / "performance_comparison_est_vs_real.html"))
    except Exception as e:
        print(f"   ⚠ Could not create comparison plot: {e}")
        
    # 4. Univariate Drift Plot
    print(f"   Saving Univariate Drift Plot...")
    fig_drift = drift_results.plot()
    fig_drift.write_html(str(reports_dir / "univariate_drift.html"))
    
    # 5. Summary JSON
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'reference_period': {
            'start': str(train_df['datetime'].min()),
            'end': str(train_df['datetime'].max()),
            'rows': len(train_df)
        },
        'analysis_period': {
            'start': str(test_df['datetime'].min()),
            'end': str(test_df['datetime'].max()),
            'rows': len(test_df)
        },
        'drift_detected_features': [], # To be populated if we parse results
        'reports': {
            'performance_estimation': str(reports_dir / "performance_estimation.html"),
            'realized_performance': str(reports_dir / "realized_performance.html"),
            'univariate_drift': str(reports_dir / "univariate_drift.html")
        }
    }
    
    # Simple check for drift in results dataframe
    # drift_results.to_df()
    drift_df = drift_results.filter(period='analysis').to_df()
    
    # Loop to find drifted features (simplified logic)
    # The structure is complex MultiIndex, skipping detailed parsing for JSON now
    
    summary_path = reports_dir / "nannyml_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"   ✓ Summary saved: {summary_path}")
    
    print("\n" + "=" * 80)
    print("NANNYML ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nReports generated in: {reports_dir}")
    print("Open the HTML files to view interactive plots.")
    
if __name__ == "__main__":
    run_nannyml_analysis()
