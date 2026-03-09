import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import json

def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    # xgboost predicting values directly, compute weighted F1
    # since we use predict_proba or predict?
    # returning a metric name and value
    y_pred_bin = (y_pred > 0.5).astype(int)
    score = f1_score(y_true, y_pred_bin, average='weighted')
    return 'weighted_f1', score

def train_model():
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'train_engineered.parquet'
    
    if not data_path.exists():
        print("Engineered data not found. Please run feature_engineering.py first.")
        sys.exit(1)
        
    print("Loading engineered dataset...")
    df = pd.read_parquet(data_path)
    
    # Sort by time to ensure TimeSeriesSplit works correctly
    # (Assuming we have flattened time or preserved the sequential order.
    # If time isn't in columns, the Parquet order is assumed preserved).
    if 'time' in df.columns:
        df = df.sort_values('time')
        X = df.drop(columns=['time', 'spoofed'])
    else:
        X = df.drop(columns=['spoofed'])
        
    y = df['spoofed']
    
    print(f"Features dimension: {X.shape}")
    print(f"Target class distribution:\n{y.value_counts(normalize=True)}")
    
    # We use StratifiedKFold instead of TimeSeriesSplit because spoofing events
    # are extremely rare and chronologically clustered. TimeSeriesSplit creates 
    # early folds with zero spoofing examples, which mathematically breaks SMOTE. 
    n_splits = 5
    print(f"\nInitializing StratifiedKFold with {n_splits} folds...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\n--- Fold {fold+1} ---")
        X_tr_raw, y_tr_raw = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        # Calculate scale_pos_weight dynamically for each fold's training set
        # to handle the imbalance natively inside XGBoost instead of using SMOTE.
        scale_pos = float(np.sum(y_tr_raw == 0)) / np.sum(y_tr_raw == 1)
        print(f"Training Fold {fold+1} natively with scale_pos_weight={scale_pos:.2f}...")
    
        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            scale_pos_weight=scale_pos,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_tr_raw, y_tr_raw,
            eval_set=[(X_va, y_va)],
            verbose=50
        )
        
        models.append(model)
        
        val_preds = model.predict(X_va)
        oof_preds[val_idx] = val_preds
        f1 = f1_score(y_va, val_preds, average='weighted')
        print(f"Fold {fold+1} Weighted F1: {f1:.4f}")
        
    print("\n" + "="*40)
    print("📊 FULL OUT-OF-FOLD EVALUATION 📊")
    print("="*40)
    print(classification_report(y, oof_preds, target_names=['Authentic (0)', 'Spoofed (1)']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, oof_preds)
    print(f"True Authentic:  {cm[0][0]} | False Alarms: {cm[0][1]}")
    print(f"Missed Attacks:  {cm[1][0]} | Caught Attacks: {cm[1][1]}")
    
    # Feature Importance from the Best Model (Fold 1)
    print("\n" + "="*40)
    print("🧠 TOP 20 MOST IMPORTANT FEATURES 🧠")
    print("="*40)
    importance_dict = models[0].get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': importance_dict.keys(),
        'importance': importance_dict.values()
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head(20).to_string(index=False))
        
    model_dir = base_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    models[0].save_model(model_dir / 'xgb_model.json')
    print("Model saved to models/xgb_model.json")
    
if __name__ == '__main__':
    train_model()
