"""
Time-Based Data Splitting Script
Creates train/validation/test splits preserving time order (35/35/30)
"""
import pandas as pd
import numpy as np
from pathlib import Path

def split_data_by_time():
    """
    Split cleaned data into train/validation/test sets based on time
    
    Split proportions:
    - First 35% → Training set
    - Next 35% → Validation set
    - Final 30% → Test set
    """
    
    print("=" * 80)
    print("TIME-BASED DATA SPLITTING")
    print("=" * 80)
    
    # Define paths
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    input_path = processed_dir / "cleaned_data.csv"
    
    # Load cleaned data
    print(f"\n[1/4] Loading cleaned data from: {input_path}")
    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"   Shape: {df.shape}")
    
    # Sort by datetime (ascending order)
    print("\n[2/4] Sorting by datetime in ascending order")
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Calculate split indices
    print("\n[3/4] Calculating split indices (35% / 35% / 30%)")
    n_total = len(df)
    
    # Calculate split points
    train_end = int(n_total * 0.35)
    val_end = int(n_total * 0.70)  # 35% + 35% = 70%
    
    # Create splits
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Verify proportions
    train_pct = len(train_df) / n_total * 100
    val_pct = len(val_df) / n_total * 100
    test_pct = len(test_df) / n_total * 100
    
    print(f"\n   Split Summary:")
    print(f"   {'Set':<12} {'Rows':<10} {'Percentage':<12} {'Date Range'}")
    print(f"   {'-'*70}")
    print(f"   {'Training':<12} {len(train_df):<10,} {train_pct:>6.1f}%      {train_df['datetime'].min()} to {train_df['datetime'].max()}")
    print(f"   {'Validation':<12} {len(val_df):<10,} {val_pct:>6.1f}%      {val_df['datetime'].min()} to {val_df['datetime'].max()}")
    print(f"   {'Test':<12} {len(test_df):<10,} {test_pct:>6.1f}%      {test_df['datetime'].min()} to {test_df['datetime'].max()}")
    print(f"   {'-'*70}")
    print(f"   {'Total':<12} {n_total:<10,} {100.0:>6.1f}%")
    
    # Save splits
    print("\n[4/4] Saving splits to CSV files")
    
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "validate.csv"
    test_path = processed_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    print(f"   ✓ Saved training set: {train_path}")
    
    val_df.to_csv(val_path, index=False)
    print(f"   ✓ Saved validation set: {val_path}")
    
    test_df.to_csv(test_path, index=False)
    print(f"   ✓ Saved test set: {test_path}")
    
    # Create summary report
    summary_path = processed_dir / "split_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("TIME-BASED DATA SPLIT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total rows: {n_total:,}\n\n")
        f.write(f"Training Set:\n")
        f.write(f"  Rows: {len(train_df):,} ({train_pct:.1f}%)\n")
        f.write(f"  Date range: {train_df['datetime'].min()} to {train_df['datetime'].max()}\n\n")
        f.write(f"Validation Set:\n")
        f.write(f"  Rows: {len(val_df):,} ({val_pct:.1f}%)\n")
        f.write(f"  Date range: {val_df['datetime'].min()} to {val_df['datetime'].max()}\n\n")
        f.write(f"Test Set:\n")
        f.write(f"  Rows: {len(test_df):,} ({test_pct:.1f}%)\n")
        f.write(f"  Date range: {test_df['datetime'].min()} to {test_df['datetime'].max()}\n\n")
        f.write("=" * 80 + "\n")
    
    print(f"   ✓ Saved summary report: {summary_path}")
    
    print("\n" + "=" * 80)
    print("SPLITTING COMPLETE")
    print("=" * 80)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = split_data_by_time()
    print("\n✓ Data splitting complete!")
