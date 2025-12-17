"""
Script to explore the Air Quality UCI dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path

def explore_dataset():
    """Load and explore the Air Quality dataset"""
    
    # Define paths
    data_path = Path(__file__).parent.parent / "data" / "raw" / "AirQualityUCI.xlsx"
    
    print("=" * 80)
    print("AIR QUALITY UCI DATASET EXPLORATION")
    print("=" * 80)
    
    # Load dataset
    print(f"\nLoading dataset from: {data_path}")
    df = pd.read_excel(data_path)
    
    # Basic info
    print(f"\n{'Dataset Shape':-<50}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    
    # Column names
    print(f"\n{'Column Names':-<50}")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    # Data types
    print(f"\n{'Data Types':-<50}")
    print(df.dtypes)
    
    # Missing values
    print(f"\n{'Missing Values':-<50}")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    # First few rows
    print(f"\n{'First 5 Rows':-<50}")
    print(df.head())
    
    # Statistical summary
    print(f"\n{'Statistical Summary':-<50}")
    print(df.describe())
    
    # Check for timestamp columns
    print(f"\n{'Potential Timestamp Columns':-<50}")
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            print(f"  - {col}: {df[col].dtype}")
            print(f"    Sample values: {df[col].head(3).tolist()}")
    
    # Identify potential target columns (numeric columns)
    print(f"\n{'Potential Regression Targets (Numeric Columns)':-<50}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        print(f"  - {col}")
        print(f"    Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
        print(f"    Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
        print()
    
    print("=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)
    
    return df

if __name__ == "__main__":
    df = explore_dataset()
