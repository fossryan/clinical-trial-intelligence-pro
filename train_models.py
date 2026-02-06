#!/usr/bin/env python3
"""
Model Training Script - Handles NaN values
Trains models only on trials with clear success/failure labels
"""

import sys
from pathlib import Path
import pandas as pd
import joblib
import json
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

print("\n" + "="*70)
print("TRAINING MODELS ON PROCESSED DATA")
print("="*70 + "\n")

# Find processed features
print(">>> Finding processed features...")
processed_dir = Path(__file__).parent / 'data' / 'processed'
feature_files = list(processed_dir.glob('clinical_trials_features*.csv'))

if not feature_files:
    print(f"❌ ERROR: No processed features found in {processed_dir}")
    sys.exit(1)

latest_features = max(feature_files, key=lambda p: p.stat().st_mtime)
print(f"✓ Found: {latest_features.name}")

# Load features
print(f"\n>>> Loading features...")
df = pd.read_csv(latest_features, low_memory=False)
print(f"✓ Loaded {len(df):,} trials with {df.shape[1]} columns")

# Check for trial_success column
if 'trial_success' not in df.columns:
    print(f"❌ ERROR: No 'trial_success' column found")
    print(f"Available columns: {df.columns.tolist()[:10]}...")
    sys.exit(1)

# Remove trials without success labels (active/recruiting trials)
print(f"\n>>> Filtering trials with labels...")
original_count = len(df)
df_labeled = df[df['trial_success'].notna()].copy()
removed_count = original_count - len(df_labeled)

print(f"✓ Original trials: {original_count:,}")
print(f"✓ Removed {removed_count:,} trials without labels (active/recruiting)")
print(f"✓ Training on: {len(df_labeled):,} labeled trials")

if len(df_labeled) < 100:
    print(f"❌ ERROR: Not enough labeled trials ({len(df_labeled)})")
    print("Need at least 100 trials with clear success/failure outcomes")
    sys.exit(1)

# Train models
print(f"\n>>> Training models...")
try:
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    import numpy as np
    
    # Prepare data
    X = df_labeled.drop(['trial_success', 'nct_id'], axis=1, errors='ignore')
    y = df_labeled['trial_success']
    
    # Remove text columns and keep only numeric
    X_numeric = X.select_dtypes(include=['number'])
    
    # Remove columns with too many NaN values
    nan_threshold = 0.5  # Remove columns with >50% NaN
    X_clean = X_numeric.loc[:, X_numeric.isna().mean() < nan_threshold]
    
    # Fill remaining NaN with median
    X_clean = X_clean.fillna(X_clean.median())
    
    print(f"✓ Features for training: {X_clean.shape[1]} numeric columns")
    print(f"✓ Success rate: {y.mean()*100:.1f}%")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Training set: {len(X_train):,} trials")
    print(f"✓ Test set: {len(X_test):,} trials")
    
    # Train XGBoost
    print(f"\n>>> Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train, verbose=False)
    xgb_acc = xgb.score(X_test, y_test)
    print(f"✓ XGBoost accuracy: {xgb_acc:.1%}")
    
    # Train LightGBM
    print(f"\n>>> Training LightGBM...")
    lgb = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    lgb.fit(X_train, y_train)
    lgb_acc = lgb.score(X_test, y_test)
    print(f"✓ LightGBM accuracy: {lgb_acc:.1%}")
    
    # Save models
    print(f"\n>>> Saving models...")
    model_dir = Path(__file__).parent / 'data' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(xgb, model_dir / 'xgboost_model.joblib')
    joblib.dump(lgb, model_dir / 'lightgbm_model.joblib')
    
    # Save feature names
    with open(model_dir / 'feature_names.json', 'w') as f:
        json.dump(list(X_clean.columns), f, indent=2)
    
    print(f"✓ Models saved to: {model_dir}")
    
    # Calculate additional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    xgb_pred = xgb.predict(X_test)
    lgb_pred = lgb.predict(X_test)
    
    xgb_precision = precision_score(y_test, xgb_pred)
    xgb_recall = recall_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred)
    
    lgb_precision = precision_score(y_test, lgb_pred)
    lgb_recall = recall_score(y_test, lgb_pred)
    lgb_f1 = f1_score(y_test, lgb_pred)
    
    # Success summary
    print("\n" + "="*70)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Dataset Summary:")
    print(f"  • Total collected: {original_count:,} trials")
    print(f"  • Labeled (completed/terminated): {len(df_labeled):,} trials")
    print(f"  • Unlabeled (active/recruiting): {removed_count:,} trials")
    print(f"  • Features used: {X_clean.shape[1]} columns")
    
    print(f"\n🎯 XGBoost Performance:")
    print(f"  • Accuracy:  {xgb_acc:.1%}")
    print(f"  • Precision: {xgb_precision:.1%}")
    print(f"  • Recall:    {xgb_recall:.1%}")
    print(f"  • F1-Score:  {xgb_f1:.1%}")
    
    print(f"\n🎯 LightGBM Performance:")
    print(f"  • Accuracy:  {lgb_acc:.1%}")
    print(f"  • Precision: {lgb_precision:.1%}")
    print(f"  • Recall:    {lgb_recall:.1%}")
    print(f"  • F1-Score:  {lgb_f1:.1%}")
    
    print(f"\n✅ Files Created:")
    print(f"  • {model_dir / 'xgboost_model.joblib'}")
    print(f"  • {model_dir / 'lightgbm_model.joblib'}")
    print(f"  • {model_dir / 'feature_names.json'}")
    
    print(f"\n🚀 Your app is ready with {len(df_labeled):,} training examples!")
    
    print(f"\n📈 Next Steps:")
    print(f"  1. Test locally:")
    print(f"     streamlit run src/app/streamlit_app.py")
    print(f"\n  2. Verify it works:")
    print(f"     - Overview page should show {original_count:,} trials")
    print(f"     - Predictions use {len(df_labeled):,} labeled trials")
    print(f"     - Model accuracy: ~{max(xgb_acc, lgb_acc):.1%}")
    print(f"\n  3. Deploy to production:")
    print(f"     git add data/")
    print(f"     git commit -m 'Expand to 8,471 trials (4X growth)'")
    print(f"     git push")
    
    print(f"\n💡 Marketing Update:")
    print(f"  • Change '2,000 trials' → '8,500+ trials'")
    print(f"  • Trained on {len(df_labeled):,} completed/terminated trials")
    print(f"  • Model accuracy: {max(xgb_acc, lgb_acc):.1%}")
    print(f"  • 4X data expansion for better predictions")
    
    print("="*70 + "\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
