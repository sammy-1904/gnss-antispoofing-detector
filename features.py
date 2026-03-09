"""
features.py — Physics-Informed Feature Engineering for GNSS Anti-Spoofing
==========================================================================
Implements three layers of features as described in the ProblemDescription:
  1. SQM (Signal Quality Monitoring) metrics — per-row, physics-based
  2. Temporal rolling/lag features — per channel group
  3. Cross-channel consistency features — per timestep aggregation
  4. Clock / PIR (Physics-Informed Residual) features — per channel group
"""

import numpy as np
import pandas as pd
from typing import List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPS = 1e-9   # avoid division by zero

# Features to compute rolling/lag/diff statistics on
TEMPORAL_COLS = [
    'CN0', 'Carrier_Doppler_hz', 'Carrier_phase', 'PC',
    'delta_metric', 'ratio_metric', 'qsqm',
]

ROLL_WINDOWS = [5, 20]   # short-term and medium-term windows


# ---------------------------------------------------------------------------
# 1. SQM / Physics features (per row — no group context needed)
# ---------------------------------------------------------------------------
def add_sqm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Signal Quality Monitoring metrics derived from correlator outputs.

    delta_metric  : (EC - LC) / (2*PC)  — measures peak symmetry.
                    Near 0 for clean signals; distorted by overlapping spoofed signals.
    ratio_metric  : (EC + LC) / (2*PC)  — measures peak flatness.
                    Approaches 1.0 for clean; rises for spoofing-induced flat peaks.
    qsqm          : PQP / PIP           — Q-channel energy leakage ratio.
                    Authentic signals carry most energy in I-channel (PQP ≈ 0).
                    Spoofing induces phase misalignment → abnormal Q-energy.
    pip_pqp_ratio : PIP / PQP           — inverse view of energy leakage.
    tow_rx_diff   : TOW - RX_time       — timing mismatch between satellite
                    transmit time and receiver timestamp.
                    Replay / delay-and-forward attacks shift this offset.
    is_active     : 1 if channel is tracking a satellite (PC > 0), else 0.
    """
    df = df.copy()

    pc2 = 2.0 * df['PC'] + EPS

    df['delta_metric']  = (df['EC'] - df['LC']) / pc2
    df['ratio_metric']  = (df['EC'] + df['LC']) / pc2
    df['qsqm']          = df['PQP'] / (df['PIP'] + EPS)
    df['pip_pqp_ratio'] = df['PIP'] / (df['PQP'] + EPS)
    df['tow_rx_diff']   = df['TOW'] - df['RX_time']
    df['is_active']     = (df['PC'] > 0).astype(np.int8)

    # Prompt correlator cross-check: PC ≈ sqrt(PIP^2 + PQP^2)
    df['pc_reconstructed'] = np.sqrt(df['PIP'] ** 2 + df['PQP'] ** 2)
    df['pc_residual']      = df['PC'] - df['pc_reconstructed']   # should be ~0 for clean

    return df


# ---------------------------------------------------------------------------
# 2. Temporal rolling / lag features (per channel, sorted by time)
# ---------------------------------------------------------------------------
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling statistics, diffs, and lag features within each channel group.

    Spoofing attacks evolve over time (drag-off). Rolling features capture:
    - Short-term trend (window=5)
    - Medium-term trend (window=20)
    - First-difference (rate of change)
    - Lag values (what the signal looked like k steps ago)
    """
    df = df.sort_values(['channel', 'time']).copy()

    groups = df.groupby('channel', sort=False)

    roll_mean_parts = []
    roll_std_parts  = []
    diff1_parts     = []
    diff3_parts     = []
    lag1_parts      = []
    lag3_parts      = []

    for col in TEMPORAL_COLS:
        if col not in df.columns:
            continue

        for w in ROLL_WINDOWS:
            roll_mean_parts.append(
                groups[col].transform(lambda x: x.rolling(w, min_periods=1).mean())
                           .rename(f'{col}_rollmean{w}')
            )
            roll_std_parts.append(
                groups[col].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
                           .rename(f'{col}_rollstd{w}')
            )

        diff1_parts.append(
            groups[col].transform(lambda x: x.diff(1).fillna(0)).rename(f'{col}_diff1')
        )
        diff3_parts.append(
            groups[col].transform(lambda x: x.diff(3).fillna(0)).rename(f'{col}_diff3')
        )
        lag1_parts.append(
            groups[col].transform(lambda x: x.shift(1).bfill()).rename(f'{col}_lag1')
        )
        lag3_parts.append(
            groups[col].transform(lambda x: x.shift(3).bfill()).rename(f'{col}_lag3')
        )

    all_parts = roll_mean_parts + roll_std_parts + diff1_parts + diff3_parts + lag1_parts + lag3_parts
    new_cols = pd.concat(all_parts, axis=1)
    df = pd.concat([df.reset_index(drop=True), new_cols.reset_index(drop=True)], axis=1)

    return df


# ---------------------------------------------------------------------------
# 3. PIR / Clock residuals (per channel group)
# ---------------------------------------------------------------------------
def add_pir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Physics-Informed Residuals:

    doppler_rate     : first difference of Carrier_Doppler_hz (Doppler acceleration).
                       A spoofer cannot perfectly replicate the natural Doppler dynamics
                       of a receiver in motion → non-physical acceleration spikes.
    pseudorange_rate : first difference of Pseudorange_m.
                       Should be consistent with Doppler shift × GPS L1 wavelength (~0.19m).
    carrier_phase_acc: second difference of Carrier_phase (acceleration).
                       Phase discontinuities (cycle slips) are physically impossible under
                       normal motion but appear under spoofing.
    tcd_deviation    : TCD minus its rolling mean — detects abnormal clock corrections.
    doppler_pseudo_residual: physical consistency check between Doppler and pseudorange rate.
                       Authentic: pseudorange_rate ≈ -doppler_rate × λ (λ≈0.19m for L1).
    """
    df = df.sort_values(['channel', 'time']).copy()
    groups = df.groupby('channel', sort=False)

    GPS_L1_WAVELENGTH = 0.1903   # metres

    df['doppler_rate']     = groups['Carrier_Doppler_hz'].transform(lambda x: x.diff(1).fillna(0))
    df['pseudorange_rate'] = groups['Pseudorange_m'].transform(lambda x: x.diff(1).fillna(0))
    df['carrier_phase_acc']= groups['Carrier_phase'].transform(lambda x: x.diff(1).diff(1).fillna(0))
    df['tcd_deviation']    = (
        groups['TCD'].transform(lambda x: x - x.rolling(10, min_periods=1).mean())
    )

    # Geometric consistency: pseudorange_rate + Carrier_Doppler_hz*λ should be ~0
    # (range rate = -Doppler_freq × wavelength, so residual = pseudorange_rate + f_d × λ ≈ 0)
    df['doppler_pseudo_residual'] = (
        df['pseudorange_rate'] + df['Carrier_Doppler_hz'] * GPS_L1_WAVELENGTH
    )

    return df


# ---------------------------------------------------------------------------
# 4. Cross-channel consistency features (per timestep)
# ---------------------------------------------------------------------------
def add_cross_channel_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate channel-level features per timestep.

    A single spoofer emits from a fixed location, so ALL channels should be
    affected simultaneously. Cross-channel variance features detect this.

    active_channels   : how many channels are tracking satellites at this time.
    cn0_std/range     : CN0 variance — spoofer broadcasts uniform-power signals,
                        reducing the natural CN0 spread across satellites.
    doppler_std/range : Doppler variance — authentic satellites have different
                        Doppler shifts due to geometry; a spoofer is fixed.
    delta/ratio/qsqm  : SQM metric aggregates across all channels.
    """
    agg_dict = {
        'is_active':    ['sum'],
        'CN0':          ['std', 'mean', 'max', 'min', lambda x: x.max() - x.min()],
        'Carrier_Doppler_hz': ['std', 'mean', lambda x: x.max() - x.min()],
        'delta_metric': ['max', 'std', 'mean'],
        'ratio_metric': ['max', 'std'],
        'qsqm':         ['max', 'mean'],
        'tow_rx_diff':  ['std', 'mean'],
        'pc_residual':  ['std', 'mean'],
        'doppler_pseudo_residual': ['std', 'mean', lambda x: x.abs().mean()],
        'PC':           ['max', 'sum'],
    }

    agg = df.groupby('time').agg(agg_dict)
    # Flatten multi-level column names
    agg.columns = ['_'.join(str(c) for c in col).strip('_') for col in agg.columns]
    # Rename lambda columns
    rename_map = {}
    for c in agg.columns:
        if '<lambda' in c:
            base = c.split('_<lambda')[0]
            rename_map[c] = f'{base}_range' if 'range' not in c else c
    # Handle duplicate renames
    seen = {}
    clean_rename = {}
    for old, new in rename_map.items():
        if new in seen:
            seen[new] += 1
            new = f'{new}_{seen[new]}'
        else:
            seen[new] = 0
        clean_rename[old] = new
    agg = agg.rename(columns=clean_rename)
    agg = agg.reset_index()

    df = df.merge(agg, on='time', how='left')
    return df


# ---------------------------------------------------------------------------
# 5. Master pipeline
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline. Call this on train and test separately.

    Returns the dataframe with all engineered features added.
    The columns 'time', 'channel', 'spoofed' (if present) are preserved unchanged.
    """
    n0 = df.shape[1]

    if verbose:
        print('[1/4] SQM features...')
    df = add_sqm_features(df)

    if verbose:
        print('[2/4] PIR / clock residuals...')
    df = add_pir_features(df)

    if verbose:
        print('[3/4] Temporal rolling/lag features...')
    df = add_temporal_features(df)

    if verbose:
        print('[4/4] Cross-channel consistency features...')
    df = add_cross_channel_features(df)

    # Clean up infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    n1 = df.shape[1]
    if verbose:
        print(f'Done. Added {n1 - n0} features. Total columns: {n1}')

    return df


# ---------------------------------------------------------------------------
# Utility: get feature column names (everything except id/target cols)
# ---------------------------------------------------------------------------
def get_feature_cols(df: pd.DataFrame) -> List[str]:
    drop_cols = {'time', 'channel', 'spoofed', 'Spoofed'}
    return [c for c in df.columns if c not in drop_cols]


if __name__ == '__main__':
    import sys
    from pathlib import Path

    BASE = Path(r'c:\Users\Sameer Rawat\OneDrive\Desktop\Kaizen anti spoofing')
    print('Loading train.csv...')
    train = pd.read_csv(BASE / 'train.csv')
    print(f'Raw shape: {train.shape}')

    train_eng = build_features(train, verbose=True)
    print(train_eng.head(3))
    print(f'Final shape: {train_eng.shape}')
