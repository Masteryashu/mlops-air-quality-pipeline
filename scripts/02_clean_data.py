"""
Data Cleaning and Preprocessing Script
Cleans the Air Quality UCI dataset and prepares it for modeling
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def clean_air_quality_data():
    """
    Clean the Air Quality UCI dataset
    
    Steps:
    1. Load raw data
    2. Combine Date and Time columns
    3. Replace -200.00 with NaN (missing value indicator)
    4. Handle missing values
    5. Remove outliers
    6. Create time-based features
    7. Save cleaned data
    """
    
    print("=" * 80)
    print("DATA CLEANING AND PREPROCESSING")
    print("=" * 80)
    
    # Define paths
    raw_data_path = Path(__file__).parent.parent / "data" / "raw" / "AirQualityUCI.xlsx"
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "cleaned_data.csv"
    
    # Load raw data
    print(f"\n[1/7] Loading raw data from: {raw_data_path}")
    df = pd.read_excel(raw_data_path)
    print(f"   Initial shape: {df.shape}")
    
    # Combine Date and Time columns
    print("\n[2/7] Combining Date and Time columns into datetime")
    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df = df.drop(['Date', 'Time'], axis=1)
    
    # Move datetime to first column
    cols = ['datetime'] + [col for col in df.columns if col != 'datetime']
    df = df[cols]
    
    print(f"   Datetime range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Replace -200.00 with NaN (missing value indicator)
    print("\n[3/7] Replacing -200.00 with NaN (missing value indicator)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        df.loc[df[col] == -200.0, col] = np.nan
    
    missing_before = df.isnull().sum().sum()
    print(f"   Total missing values: {missing_before:,}")
    
    # Handle missing values
    print("\n[4/7] Handling missing values")
    
    # Drop rows with more than 50% missing values
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=threshold)
    print(f"   After dropping rows with >50% missing: {df.shape}")
    
    # For remaining missing values, use forward fill then median imputation
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            # Forward fill (time series data)
            df[col] = df[col].fillna(method='ffill')
            # Backward fill for any remaining at the start
            df[col] = df[col].fillna(method='bfill')
            # Median for any still remaining
            df[col] = df[col].fillna(df[col].median())
    
    missing_after = df.isnull().sum().sum()
    print(f"   Remaining missing values: {missing_after}")
    
    # Remove outliers using IQR method
    print("\n[5/7] Removing outliers using IQR method (1.5 * IQR)")
    
    initial_rows = len(df)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    rows_removed = initial_rows - len(df)
    print(f"   Rows removed: {rows_removed:,} ({rows_removed/initial_rows*100:.1f}%)")
    print(f"   Final shape: {df.shape}")
    
    # Normalize column names (remove special characters, use underscores)
    print("\n[6/7] Normalizing column names")
    df.columns = df.columns.str.replace('(', '', regex=False)
    df.columns = df.columns.str.replace(')', '', regex=False)
    df.columns = df.columns.str.replace('.', '_', regex=False)
    df.columns = df.columns.str.replace(' ', '_', regex=False)
    
    print(f"   Columns: {list(df.columns)}")
    
    # Create time-based features
    print("\n[7/7] Creating time-based features")
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
    
    print(f"   Added features: hour, day_of_week, month, is_weekend")
    
    # Save cleaned data
    print(f"\n[SAVE] Saving cleaned data to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"   Final shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)
    print(f"Original rows: {9357:,}")
    print(f"Final rows: {len(df):,}")
    print(f"Rows retained: {len(df)/9357*100:.1f}%")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print("=" * 80)
    
    return df

if __name__ == "__main__":
    df = clean_air_quality_data()
    print("\n✓ Data cleaning complete!")
