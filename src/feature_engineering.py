import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import pywt
import warnings
warnings.filterwarnings('ignore')

import time

# Add parent directory to path to import features.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features import build_features

def spectral_entropy(signal):
    if len(signal) < 8 or np.all(signal == 0):
        return 0
    try:
        # Use discrete wavelet transform db8
        coeffs = pywt.dwt(signal, 'db8')
        cd = coeffs[1] # Detail coefficients
        p = np.abs(cd) + 1e-9
        p = p / np.sum(p)
        se = -np.sum(p * np.log2(p))
        return se
    except Exception:
        return 0

def preprocess_wavelet(df):
    # Group by channel and compute rolling wavelet spectral entropy
    df = df.sort_values(['channel', 'time']).copy()
    groups = df.groupby('channel', sort=False)
    
    # We apply it to correlation outputs as indicated in problem description
    for col in ['PC', 'EC', 'LC']:
        if col in df.columns:
            # We use a window size of 16 for rolling computation to be able to use db8
            df[f'{col}_spectral_entropy'] = groups[col].transform(
                lambda x: x.rolling(16, min_periods=16).apply(spectral_entropy, raw=True).fillna(0)
            )
    return df

def aggregate_to_time_level(df, is_train=True):
    """
    Given a channel-level dataframe, aggregate it such that there is only 1 row per time.
    """
    col_exclude = ['time', 'channel', 'PRN', 'spoofed', 'Spoofed']
    features = [c for c in df.columns if c not in col_exclude]
    
    # Aggregate features by taking mean, std, max, min across all channels at a given time
    agg_funcs = ['mean', 'std', 'max', 'min']
    agg_dict = {f: agg_funcs for f in features}
    
    print(f"Aggregating {len(features)} features across channels per timestamp...")
    time_df = df.groupby('time').agg(agg_dict)
    
    # Flatten columns
    time_df.columns = [f"{col[0]}_{col[1]}" for col in time_df.columns]
    time_df = time_df.reset_index()
    
    # Drop absolute timestamp metrics so the XGBoost model doesn't overfit to them
    cols_to_drop = [c for c in time_df.columns if c.startswith('TOW_') or c.startswith('RX_time_')]
    time_df = time_df.drop(columns=cols_to_drop)
    
    if is_train and 'spoofed' in df.columns:
        # Since spoofed status is same across all channels per time step, we just take max
        target_df = df.groupby('time')['spoofed'].max().reset_index()
        time_df = time_df.merge(target_df, on='time', how='left')
        
    return time_df

def prepare_data(df, is_train=True, verbose=True):
    # Clean data: force numeric types on non-categorical columns
    exclude_cols = {'time', 'channel', 'PRN', 'spoofed', 'Spoofed'}
    if verbose: print("Coercing numeric columns...")
    
    # Drop rows that have strings in columns that should be pure floats (like the first 8 rows of train.csv)
    # By using errors='coerce', the strings become NaN. We then drop those rows.
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Also coerce PRN to numeric if applicable, to avoid PRN="ch0"
    if 'PRN' in df.columns:
        df['PRN'] = pd.to_numeric(df['PRN'], errors='coerce')

    # Drop the rows that turned into NaNs because they contained garbage string data instead of float data
    initial_len = len(df)
    df = df.dropna(subset=['Pseudorange_m', 'Carrier_Doppler_hz'])
    if verbose: print(f"Dropped {initial_len - len(df)} corrupt rows with non-numeric data.")
        
    start_t = time.time()
    if verbose: print("Adding wavelet features...")
    df = preprocess_wavelet(df)
    if verbose: print(f"Wavelet features took {time.time() - start_t:.2f}s")
    
    start_t = time.time()
    if verbose: print("Applying features.py build_features...")
    df = build_features(df, verbose=verbose)
    if verbose: print(f"build_features took {time.time() - start_t:.2f}s")
    
    start_t = time.time()
    if verbose: print("Aggregating to time level...")
    time_df = aggregate_to_time_level(df, is_train=is_train)
    if verbose: print(f"Aggregating took {time.time() - start_t:.2f}s")
    
    # Fill any remaining NaNs after aggregation
    time_df = time_df.fillna(0)
    
    return time_df

if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent
    print('Loading train.csv...')
    train = pd.read_csv(base_dir / 'train.csv')
    print(f'Raw train shape: {train.shape}')
    
    train_eng = prepare_data(train, is_train=True)
    print(f'Engineered train shape: {train_eng.shape}')
    
    train_eng.to_parquet(base_dir / 'train_engineered.parquet', index=False)
    print("Saved engineered train dataset.")
