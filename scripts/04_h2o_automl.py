"""
H2O AutoML Script
Runs H2O AutoML to identify top 3 model types for regression
"""
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def run_h2o_automl():
    """
    Run H2O AutoML on training data to identify top model types
    """
    
    print("=" * 80)
    print("H2O AUTOML ANALYSIS")
    print("=" * 80)
    
    # Define paths
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "validate.csv"
    
    # Initialize H2O
    print("\n[1/6] Initializing H2O cluster...")
    try:
        h2o.init(max_mem_size="4G")
        print("   ✓ H2O cluster initialized")
    except Exception as e:
        print(f"   ✗ Error initializing H2O: {e}")
        print("\n   H2O requires Java Runtime Environment (JRE)")
        print("   Please install Java 8 or later from: https://www.java.com/download/")
        return None
    
    # Load data
    print("\n[2/6] Loading training and validation data...")
    train_h2o = h2o.import_file(str(train_path))
    val_h2o = h2o.import_file(str(val_path))
    print(f"   Training set: {train_h2o.shape}")
    print(f"   Validation set: {val_h2o.shape}")
    
    # Define features and target
    print("\n[3/6] Defining features and target...")
    
    # Exclude datetime and target from features
    exclude_cols = ['datetime', 'COGT']
    x = [col for col in train_h2o.columns if col not in exclude_cols]
    y = 'COGT'
    
    print(f"   Target: {y}")
    print(f"   Features ({len(x)}): {x}")
    
    # Configure and run AutoML
    print("\n[4/6] Running H2O AutoML...")
    print("   Configuration:")
    print("   - Max runtime: 600 seconds (10 minutes)")
    print("   - Max models: 20")
    print("   - Sort metric: RMSE")
    print("   - Seed: 42 (for reproducibility)")
    
    aml = H2OAutoML(
        max_runtime_secs=600,
        max_models=20,
        seed=42,
        sort_metric="RMSE",
        exclude_algos=None,  # Include all algorithms
        verbosity="info"
    )
    
    aml.train(x=x, y=y, training_frame=train_h2o, validation_frame=val_h2o)
    
    print("\n   ✓ AutoML training complete!")
    
    # Get leaderboard
    print("\n[5/6] Analyzing results...")
    lb = aml.leaderboard
    lb_df = lb.as_data_frame()
    
    print("\n   Top 10 Models:")
    print("   " + "-" * 76)
    print(f"   {'Rank':<6} {'Model ID':<40} {'RMSE':<15} {'MAE':<15}")
    print("   " + "-" * 76)
    
    for idx, row in lb_df.head(10).iterrows():
        model_id = row['model_id']
        rmse = row.get('rmse', row.get('RMSE', 'N/A'))
        mae = row.get('mae', row.get('MAE', 'N/A'))
        print(f"   {idx+1:<6} {model_id:<40} {rmse:<15.4f} {mae:<15.4f}")
    
    # Extract model types from top models
    print("\n[6/6] Identifying top 3 model types...")
    
    model_types = []
    model_details = []
    
    for idx, row in lb_df.iterrows():
        model_id = row['model_id']
        
        # Extract model type from model ID
        if 'GBM' in model_id:
            model_type = 'GBM'
        elif 'XGBoost' in model_id:
            model_type = 'XGBoost'
        elif 'DeepLearning' in model_id:
            model_type = 'DeepLearning'
        elif 'DRF' in model_id:
            model_type = 'RandomForest'
        elif 'GLM' in model_id:
            model_type = 'GLM'
        elif 'StackedEnsemble' in model_id:
            model_type = 'StackedEnsemble'
        else:
            model_type = 'Other'
        
        # Add to list if not already present and not StackedEnsemble
        if model_type not in model_types and model_type != 'StackedEnsemble':
            model_types.append(model_type)
            model_details.append({
                'model_type': model_type,
                'model_id': model_id,
                'rmse': float(row.get('rmse', row.get('RMSE', 0))),
                'mae': float(row.get('mae', row.get('MAE', 0))),
                'r2': float(row.get('r2', row.get('R2', 0)))
            })
        
        # Stop after finding 3 unique model types
        if len(model_types) >= 3:
            break
    
    print(f"\n   Top 3 Model Types:")
    for i, detail in enumerate(model_details[:3], 1):
        print(f"   {i}. {detail['model_type']}")
        print(f"      Model ID: {detail['model_id']}")
        print(f"      RMSE: {detail['rmse']:.4f}")
        print(f"      MAE: {detail['mae']:.4f}")
        print(f"      R²: {detail['r2']:.4f}")
        print()
    
    # Save results
    results = {
        'top_3_models': model_details[:3],
        'leaderboard': lb_df.head(10).to_dict('records'),
        'automl_config': {
            'max_runtime_secs': 600,
            'max_models': 20,
            'sort_metric': 'RMSE',
            'seed': 42
        }
    }
    
    results_path = models_dir / "h2o_automl_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✓ Results saved to: {results_path}")
    
    # Shutdown H2O
    print("\n[CLEANUP] Shutting down H2O cluster...")
    h2o.cluster().shutdown()
    
    print("\n" + "=" * 80)
    print("H2O AUTOML COMPLETE")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    results = run_h2o_automl()
    if results:
        print("\n✓ H2O AutoML analysis complete!")
        print(f"\nTop 3 model types to implement manually:")
        for i, model in enumerate(results['top_3_models'], 1):
            print(f"  {i}. {model['model_type']}")
