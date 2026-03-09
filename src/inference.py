import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_engineering import prepare_data

def generate_submission():
    base_dir = Path(__file__).parent.parent
    test_path = base_dir / 'test.csv'
    model_path = base_dir / 'models' / 'xgb_model.json'
    submission_path = base_dir / 'submission.csv'
    
    if not test_path.exists():
        print(f"Error: Could not find {test_path}")
        sys.exit(1)
        
    if not model_path.exists():
        print(f"Error: Could not find trained model at {model_path}. Run main.py first.")
        sys.exit(1)
        
    print("Loading XGBoost model...")
    model = XGBClassifier()
    model.load_model(model_path)
    
    print("Loading raw test data...")
    test_df = pd.read_csv(test_path)
    
    # Keep the original absolute time stamps for the submission file
    submission_times = test_df['time'].unique()
    
    print("Preprocessing Test features (this will take a few minutes)...")
    # pass is_train=False so it doesn't try to look for the 'spoofed' answer key column
    test_eng = prepare_data(test_df, is_train=False, verbose=True)
    
    # Exclude time column just like we did during training
    X_test = test_eng.drop(columns=['time'])
    
    print("Running Inference...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # Get the probability of class 1 (Spoofed)
    
    print("Formatting Submission...")
    sub_df = pd.DataFrame({
        'time': test_eng['time'],
        'Spoofed': predictions,
        'Confidence': probabilities
    })
    
    # Ensure every single exact unique time step from the original raw file exists
    # (just in case the preprocessing dropped corrupted garbage rows)
    final_sub = pd.DataFrame({'time': submission_times})
    final_sub = final_sub.merge(sub_df, on='time', how='left')
    final_sub['Spoofed'] = final_sub['Spoofed'].fillna(0).astype(int) # default to not-spoofed if corrupted
    final_sub['Confidence'] = final_sub['Confidence'].fillna(0.0) # default to 0 confidence if corrupted
    
    final_sub.to_csv(submission_path, index=False)
    print(f"✅ Success! Submission file saved as: {submission_path}")

if __name__ == '__main__':
    generate_submission()
