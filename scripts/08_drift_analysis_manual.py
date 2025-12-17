"""
Manual Model Drift Analysis Script
Detects data drift using Kolmogorov-Smirnov test
Calculates performance drift using RMSE/MAE
Generates a standalone HTML report
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime
import mlflow
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config.mlflow_config import setup_mlflow, MODEL_NAME_RANDOM_FOREST, MODEL_NAME_XGBOOST

def run_manual_drift_analysis():
    print("=" * 80)
    print("MANUAL DRIFT ANALYSIS")
    print("=" * 80)
    
    # Setup MLflow
    try:
        mlflow_client = setup_mlflow()
    except Exception as e:
        print(f"   ⚠ MLflow setup failed (Offline mode?): {e}")
        mlflow_client = None
    
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
    
    # Load Champion Model (Random Forest)
    print("\n[2/5] Loading Champion Model...")
    model = None
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
        print(f"   ⚠ Could not load any model from MLflow (Connection Error?). Performance drift will be skipped.\n   Error: {e}")
        
    # Make predictions
    if model is not None:
        print(f"   Making predictions using {model_name}...")
        train_df['y_pred'] = model.predict(train_df[feature_cols])
        test_df['y_pred'] = model.predict(test_df[feature_cols])
    else:
        # Create dummy predictions to prevent crash, or handle skipping
        train_df['y_pred'] = 0
        test_df['y_pred'] = 0
    
    # Calculate Data Drift (KS Test)
    print("\n[3/5] Calculating Data Drift (KS Test)...")
    drift_results = []
    
    for feature in feature_cols:
        # KS Test
        stat, p_value = ks_2samp(train_df[feature], test_df[feature])
        drifted = p_value < 0.05
        
        drift_results.append({
            'feature': feature,
            'p_value': p_value,
            'drifted': drifted,
            'stat': stat
        })
        
    drift_df = pd.DataFrame(drift_results)
    n_drifted = drift_df['drifted'].sum()
    print(f"   Drift detected in {n_drifted}/{len(feature_cols)} features")
    
    # Calculate Performance Drift
    print("\n[4/5] Calculating Performance Drift...")
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    if model is not None:
        perf = {
            'reference': {
                'rmse': np.sqrt(mean_squared_error(train_df[target_col], train_df['y_pred'])),
                'mae': mean_absolute_error(train_df[target_col], train_df['y_pred'])
            },
            'analysis': {
                'rmse': np.sqrt(mean_squared_error(test_df[target_col], test_df['y_pred'])),
                'mae': mean_absolute_error(test_df[target_col], test_df['y_pred'])
            }
        }
        print(f"   Ref RMSE: {perf['reference']['rmse']:.4f}, Analysis RMSE: {perf['analysis']['rmse']:.4f}")
    else:
        perf = {
            'reference': {'rmse': 0, 'mae': 0},
            'analysis': {'rmse': 0, 'mae': 0}
        }
        print("   ⚠ Skipped due to missing model.")
    
    # Generate Plots
    print("\n[5/5] Generating Plots and Report...")
    plots = {}
    
    # Drift Plot (Bar chart of P-values)
    plt.figure(figsize=(10, 6))
    colors = ['red' if d else 'green' for d in drift_df['drifted']]
    plt.barh(drift_df['feature'], drift_df['p_value'], color=colors)
    plt.axvline(0.05, color='black', linestyle='--', label='Threshold (0.05)')
    plt.title('Feature Drift (KS Test P-values)')
    plt.xlabel('P-Value (Lower = More Drift)')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plots['drift_chart'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Generate HTML Report
    html_content = f"""
    <html>
    <head>
        <title>Drift Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            .metric-box {{ background: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .drifted {{ color: red; font-weight: bold; }}
            .safe {{ color: green; }}
        </style>
    </head>
    <body>
        <h1>ML Model Drift Analysis Report</h1>
        <p>Generated on: {datetime.now()}</p>
        
        <div class="metric-box">
            <h2>Model Performance (RMSE)</h2>
            <p>Reference (Train): <b>{perf['reference']['rmse']:.4f}</b></p>
            <p>Analysis (Test): <b>{perf['analysis']['rmse']:.4f}</b></p>
            <p>Change: <span class="{'drifted' if perf['analysis']['rmse'] > perf['reference']['rmse'] * 1.1 else 'safe'}">
                {((perf['analysis']['rmse'] - perf['reference']['rmse']) / perf['reference']['rmse']) * 100:.1f}%
            </span></p>
        </div>
        
        <div class="metric-box">
            <h2>Data Drift Summary</h2>
            <p>Features Drifted: <b>{n_drifted} / {len(feature_cols)}</b></p>
            <img src="data:image/png;base64,{plots['drift_chart']}" width="800" />
            
            <h3>Detailed Feature Drift (KS Test)</h3>
            <table>
                <tr><th>Feature</th><th>P-Value</th><th>Status</th></tr>
                {''.join([f"<tr><td>{r['feature']}</td><td>{r['p_value']:.4f}</td><td class='{('drifted' if r['drifted'] else 'safe')}'>{('DRIFT DETECTED' if r['drifted'] else 'Stable')}</td></tr>" for r in drift_results])}
            </table>
        </div>
    </body>
    </html>
    """
    
    report_path = reports_dir / "drift_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
        
    print(f"   ✓ Report saved: {report_path}")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
if __name__ == "__main__":
    run_manual_drift_analysis()
