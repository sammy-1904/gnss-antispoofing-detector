import pandas as pd
import numpy as np
import sys
import os
import json
from pathlib import Path
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_engineering import prepare_data

def generate_submission():
    base_dir = Path(__file__).parent.parent
    test_path = base_dir / 'test.csv'
    model_dir = base_dir / 'models'
    submission_path = base_dir / 'submission.csv'

    if not test_path.exists():
        print(f"Error: Could not find {test_path}")
        sys.exit(1)

    # Load all fold models for ensemble inference
    fold_models = []
    for i in range(5):
        fold_path = model_dir / f'xgb_model_fold{i}.json'
        if not fold_path.exists():
            print(f"Warning: fold model {i} not found at {fold_path}. Stopping ensemble load.")
            break
        m = XGBClassifier()
        m.load_model(fold_path)
        fold_models.append(m)
        print(f"Loaded fold {i+1} model.")

    if not fold_models:
        print("Error: No fold models found. Run main.py first.")
        sys.exit(1)

    # Load the tuned decision threshold
    threshold_path = model_dir / 'threshold.json'
    if threshold_path.exists():
        with open(threshold_path) as f:
            threshold = json.load(f)['threshold']
        print(f"Using tuned threshold: {threshold:.2f}")
    else:
        threshold = 0.5
        print("threshold.json not found — falling back to threshold=0.5")

    print("Loading raw test data...")
    test_df = pd.read_csv(test_path)

    # Keep the original timestamps for the submission file
    submission_times = test_df['time'].unique()

    print("Preprocessing test features (this will take a few minutes)...")
    test_eng = prepare_data(test_df, is_train=False, verbose=True)

    X_test = test_eng.drop(columns=['time'])

    # Align test columns to exactly what the models were trained on.
    # XGBoost silently fills unrecognised/missing features with 0, which can
    # collapse all probabilities to ~0 if there is any column mismatch.
    expected_features = fold_models[0].get_booster().feature_names
    missing = set(expected_features) - set(X_test.columns)
    extra   = set(X_test.columns) - set(expected_features)
    if missing:
        print(f"Warning: {len(missing)} features missing from test set — filling with 0: {sorted(missing)[:5]}...")
    if extra:
        print(f"Info: {len(extra)} extra test columns not seen during training — dropping them.")
    X_test = X_test.reindex(columns=expected_features, fill_value=0)
    print(f"Feature alignment: {X_test.shape[1]} features matched to model.")

    print(f"Running ensemble inference across {len(fold_models)} fold models...")
    all_probs = np.mean(
        [m.predict_proba(X_test)[:, 1] for m in fold_models],
        axis=0
    )
    print("\n--- Probability Distribution Diagnostics ---")
    print(f"  min   : {all_probs.min():.4f}")
    print(f"  mean  : {all_probs.mean():.4f}")
    print(f"  median: {np.median(all_probs):.4f}")
    print(f"  90th  : {np.percentile(all_probs, 90):.4f}")
    print(f"  99th  : {np.percentile(all_probs, 99):.4f}")
    print(f"  max   : {all_probs.max():.4f}")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, threshold]:
        n = int((all_probs > t).sum())
        print(f"  > {t:.2f} : {n} samples ({100*n/len(all_probs):.2f}%)")
    print("---------------------------------------------\n")

    # If the OOF-tuned threshold exceeds the test probability range, the model's
    # probabilities are compressed on test data (common with XGBoost + class imbalance).
    # Fall back to a gap-based threshold: find the natural valley between the two
    # probability clusters by scanning for the lowest-density bin between p=0.05
    # and the 95th percentile of probabilities.
    effective_threshold = threshold
    if all_probs.max() < threshold:
        hist, bin_edges = np.histogram(all_probs, bins=50)
        # Find the bin with minimum density between 0.05 and max_prob*0.8
        search_mask = (bin_edges[:-1] >= 0.05) & (bin_edges[1:] <= all_probs.max() * 0.8)
        if search_mask.any():
            valley_idx = np.argmin(hist[search_mask])
            valley_idxs = np.where(search_mask)[0]
            effective_threshold = float(bin_edges[valley_idxs[valley_idx] + 1])
        else:
            effective_threshold = all_probs.max() * 0.5
        print(f"OOF threshold ({threshold:.2f}) exceeds test max ({all_probs.max():.4f}).")
        print(f"Using gap-based threshold: {effective_threshold:.4f}")

    predictions = (all_probs > effective_threshold).astype(int)

    print("Formatting submission...")
    sub_df = pd.DataFrame({
        'time': test_eng['time'],
        'Spoofed': predictions,
        'Confidence': all_probs
    })

    # Ensure every unique timestamp from the original raw file is present
    # (in case preprocessing dropped corrupted rows)
    final_sub = pd.DataFrame({'time': submission_times})
    final_sub = final_sub.merge(sub_df, on='time', how='left')
    n_missing = final_sub['Spoofed'].isna().sum()
    if n_missing > 0:
        print(f"Warning: {n_missing} timestamps had no prediction (corrupted rows) — defaulting to Spoofed=0")
    final_sub['Spoofed'] = final_sub['Spoofed'].fillna(0).astype(int)
    final_sub['Confidence'] = final_sub['Confidence'].fillna(0.0)

    final_sub.to_csv(submission_path, index=False)
    print(f"✅ Success! Submission saved to: {submission_path}")
    print(f"   Threshold used  : {effective_threshold:.4f}")
    print(f"   Spoofed flagged : {final_sub['Spoofed'].sum()} / {len(final_sub)} ({100*final_sub['Spoofed'].mean():.2f}%)")

if __name__ == '__main__':
    generate_submission()
