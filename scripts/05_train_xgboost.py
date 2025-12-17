"""
XGBoost Model Training Script with MLflow Tracking
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config.mlflow_config import setup_mlflow, print_mlflow_config, MODEL_NAME_XGBOOST

def train_xgboost_model():
    """
    Train XGBoost model with MLflow tracking
    """
    
    print("=" * 80)
    print("XGBOOST MODEL TRAINING")
    print("=" * 80)
    
    # Setup MLflow
    mlflow = setup_mlflow()
    print_mlflow_config()
    
    # Define paths
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("[1/6] Loading datasets...")
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "validate.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    
    print(f"   Training: {train_df.shape}")
    print(f"   Validation: {val_df.shape}")
    print(f"   Test: {test_df.shape}")
    
    # Prepare features and target
    print("\n[2/6] Preparing features and target...")
    exclude_cols = ['datetime', 'COGT']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    target_col = 'COGT'
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"   Features ({len(feature_cols)}): {feature_cols}")
    print(f"   Target: {target_col}")
    
    # Start MLflow run
    print("\n[3/6] Training XGBoost model...")
    
    with mlflow.start_run(run_name="xgboost_model"):
        
        # Hyperparameters
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 200,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        print(f"   Hyperparameters: {params}")
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        print("   ✓ Model training complete")
        
        # Make predictions
        print("\n[4/6] Making predictions...")
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        print("\n[5/6] Calculating metrics...")
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        print("\n   Metrics Summary:")
        print("   " + "-" * 60)
        print(f"   {'Dataset':<12} {'RMSE':<15} {'MAE':<15} {'R²':<15}")
        print("   " + "-" * 60)
        print(f"   {'Training':<12} {metrics['train_rmse']:<15.4f} {metrics['train_mae']:<15.4f} {metrics['train_r2']:<15.4f}")
        print(f"   {'Validation':<12} {metrics['val_rmse']:<15.4f} {metrics['val_mae']:<15.4f} {metrics['val_r2']:<15.4f}")
        print(f"   {'Test':<12} {metrics['test_rmse']:<15.4f} {metrics['test_mae']:<15.4f} {metrics['test_r2']:<15.4f}")
        print("   " + "-" * 60)
        
        # Log to MLflow
        print("\n[6/6] Logging to MLflow...")
        
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Create and log feature importance plot
        fig, ax = plt.subplots(figsize=(10, 8))
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
        ax.set_title('Top 15 Feature Importances - XGBoost')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        
        importance_path = models_dir / "xgboost_feature_importance.png"
        plt.savefig(importance_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(str(importance_path))
        plt.close()
        
        # Create and log residual plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Validation residuals
        val_residuals = y_val - y_val_pred
        axes[0].scatter(y_val_pred, val_residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residual Plot - Validation Set')
        
        # Test residuals
        test_residuals = y_test - y_test_pred
        axes[1].scatter(y_test_pred, test_residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot - Test Set')
        
        plt.tight_layout()
        residual_path = models_dir / "xgboost_residuals.png"
        plt.savefig(residual_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(str(residual_path))
        plt.close()
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=MODEL_NAME_XGBOOST
        )
        
        print("   ✓ Logged parameters, metrics, and artifacts to MLflow")
        print(f"   ✓ Model registered as: {MODEL_NAME_XGBOOST}")
        
        # Get run ID
        run_id = mlflow.active_run().info.run_id
        print(f"\n   MLflow Run ID: {run_id}")
    
    print("\n" + "=" * 80)
    print("XGBOOST TRAINING COMPLETE")
    print("=" * 80)
    
    return model, metrics

if __name__ == "__main__":
    model, metrics = train_xgboost_model()
    print("\n✓ XGBoost model training complete!")
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {metrics['test_rmse']:.4f}")
    print(f"  MAE:  {metrics['test_mae']:.4f}")
    print(f"  R²:   {metrics['test_r2']:.4f}")
