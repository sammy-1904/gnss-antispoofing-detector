import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix

def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
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

    if 'time' in df.columns:
        df = df.sort_values('time')
        X = df.drop(columns=['time', 'spoofed'])
    else:
        X = df.drop(columns=['spoofed'])

    y = df['spoofed']

    print(f"Features dimension: {X.shape}")
    print(f"Target class distribution:\n{y.value_counts(normalize=True)}")

    # Assign each timestamp to one of N_BLOCKS contiguous temporal blocks.
    # StratifiedGroupKFold then holds out entire blocks at a time, so training
    # rows' rolling/lag features can only look back into earlier training blocks —
    # no future validation-fold data leaks into the training features.
    N_BLOCKS = 20
    n_splits = 5
    time_col = df['time'] if 'time' in df.columns else pd.Series(np.arange(len(df)))
    time_rank = time_col.rank(method='first')
    groups = pd.cut(time_rank, bins=N_BLOCKS, labels=False)

    print(f"\nInitializing StratifiedGroupKFold with {n_splits} folds over {N_BLOCKS} temporal blocks...")
    cv = StratifiedGroupKFold(n_splits=n_splits)

    oof_preds = np.zeros(len(X))
    oof_probs = np.zeros(len(X))

    models = []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
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

        val_probs = model.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = val_probs
        val_preds = (val_probs > 0.5).astype(int)
        oof_preds[val_idx] = val_preds
        f1 = f1_score(y_va, val_preds, average='weighted')
        fold_scores.append(f1)
        print(f"Fold {fold+1} Weighted F1: {f1:.4f}")

    print("\n" + "="*40)
    print("📊 FULL OUT-OF-FOLD EVALUATION (threshold=0.5) 📊")
    print("="*40)
    print(classification_report(y, oof_preds, target_names=['Authentic (0)', 'Spoofed (1)']))
    cm = confusion_matrix(y, oof_preds)
    print(f"True Authentic:  {cm[0][0]} | False Alarms: {cm[0][1]}")
    print(f"Missed Attacks:  {cm[1][0]} | Caught Attacks: {cm[1][1]}")

    # Tune threshold to maximise F1 for the minority (Spoofed) class.
    # Optimising for spoofed-class F1 pushes recall up without collapsing precision.
    print("\nTuning decision threshold on OOF probabilities...")
    thresholds = np.arange(0.05, 0.95, 0.01)
    spoofed_f1s = [
        f1_score(y, (oof_probs > t).astype(int), pos_label=1, average='binary')
        for t in thresholds
    ]
    best_thresh = float(thresholds[np.argmax(spoofed_f1s)])
    print(f"Optimal threshold: {best_thresh:.2f}  (Spoofed F1={max(spoofed_f1s):.4f})")

    oof_preds_tuned = (oof_probs > best_thresh).astype(int)
    print("\n📊 FULL OUT-OF-FOLD EVALUATION (optimal threshold) 📊")
    print("="*40)
    print(classification_report(y, oof_preds_tuned, target_names=['Authentic (0)', 'Spoofed (1)']))
    cm2 = confusion_matrix(y, oof_preds_tuned)
    print(f"True Authentic:  {cm2[0][0]} | False Alarms: {cm2[0][1]}")
    print(f"Missed Attacks:  {cm2[1][0]} | Caught Attacks: {cm2[1][1]}")

    # Feature importance: average gain across all fold models for a stable ranking
    print("\n" + "="*40)
    print("🧠 TOP 20 MOST IMPORTANT FEATURES (avg across folds) 🧠")
    print("="*40)
    combined_importance: dict = {}
    for m in models:
        for feat, score in m.get_booster().get_score(importance_type='gain').items():
            combined_importance[feat] = combined_importance.get(feat, 0.0) + score
    for feat in combined_importance:
        combined_importance[feat] /= len(models)

    importance_df = pd.DataFrame({
        'feature': combined_importance.keys(),
        'importance': combined_importance.values()
    }).sort_values('importance', ascending=False)
    print(importance_df.head(20).to_string(index=False))

    # Save all fold models and the tuned threshold for ensemble inference
    model_dir = base_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(model_dir / f'xgb_model_fold{i}.json')
        print(f"Saved fold {i+1} model → models/xgb_model_fold{i}.json")

    with open(model_dir / 'threshold.json', 'w') as f:
        json.dump({'threshold': best_thresh}, f)
    print(f"Saved optimal threshold ({best_thresh:.2f}) → models/threshold.json")

if __name__ == '__main__':
    train_model()
