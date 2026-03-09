import os
import sys
import time
from pathlib import Path
import pandas as pd

# Add the root directory to path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.feature_engineering import prepare_data
from src.train import train_model

def run_pipeline():
    print("="*60)
    print("🚀 GNSS Anti-Spoofing Pipeline Started 🚀")
    print("="*60)
    
    start_time_total = time.time()
    
    # Paths
    raw_train_path = project_root / 'train.csv'
    engineered_train_path = project_root / 'train_engineered.parquet'
    
    # ---------------------------------------------------------
    # STEP 1: Feature Engineering
    # ---------------------------------------------------------
    print("\n[STEP 1/2] Feature Engineering & Data Preparation")
    if not raw_train_path.exists():
        print(f"❌ ERROR: Could not find raw dataset at {raw_train_path}")
        sys.exit(1)
        
    print(f"Loading raw data from: {raw_train_path.name}")
    start_t = time.time()
    train_df = pd.read_csv(raw_train_path)
    print(f"Loaded {train_df.shape[0]:,} rows in {time.time() - start_t:.2f} seconds.")
    
    # Run the comprehensive feature engineering pipeline
    # (This internally calls features.build_features)
    print("\nRunning Feature Engineering Pipeline...")
    train_eng = prepare_data(train_df, is_train=True, verbose=True)
    
    # Save the processed data 
    print(f"\nSaving processed dataset to Parquet... ({train_eng.shape[1]} features)")
    train_eng.to_parquet(engineered_train_path, index=False)
    print(f"✅ Feature Engineering Complete! Processed data saved to: {engineered_train_path.name}")
    
    # ---------------------------------------------------------
    # STEP 2: Model Training
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("\n[STEP 2/2] XGBoost Model Training & Evaluation")
    
    # We just call the train_model function. 
    # It handles loading the parquet and saving the JSON model.
    train_model()
    
    print("\n" + "="*60)
    total_time = (time.time() - start_time_total) / 60
    print(f"🎉 Pipeline Complete in {total_time:.2f} minutes! 🎉")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
