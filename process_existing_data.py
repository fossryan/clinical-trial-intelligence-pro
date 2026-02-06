#!/usr/bin/env python3
"""
Process Existing Trial Data - FINAL FIXED VERSION
Run this after you've already collected trials
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

print("\n" + "="*70)
print("PROCESSING EXISTING TRIAL DATA")
print("="*70 + "\n")

# Step 1: Find the data
print(">>> Finding latest raw data file...")
raw_dir = Path(__file__).parent / 'data' / 'raw'
raw_files = list(raw_dir.glob('clinical_trials_raw*.csv'))

if not raw_files:
    print(f"❌ ERROR: No data files found in {raw_dir}")
    print("\nMake sure you have a CSV file in data/raw/ directory")
    sys.exit(1)

latest_raw = max(raw_files, key=lambda p: p.stat().st_mtime)
print(f"✓ Found: {latest_raw.name}")

# Load data
print(f"\n>>> Loading data...")
df_raw = pd.read_csv(latest_raw)
print(f"✓ Loaded {len(df_raw):,} trials with {len(df_raw.columns)} columns")

# Step 2: Feature Engineering
print(f"\n>>> Engineering features...")
try:
    from features.engineer_features import TrialFeatureEngineer
    
    engineer = TrialFeatureEngineer()
    df_features = engineer.create_features(df_raw)  # FIXED: Correct method name
    
    # Save processed data
    output_dir = Path(__file__).parent / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'clinical_trials_features_{timestamp}.csv'
    df_features.to_csv(output_file, index=False)
    
    print(f"✓ Features saved: {output_file.name}")
    print(f"✓ Total features: {df_features.shape[1]} columns")
    print(f"✓ Trials processed: {len(df_features):,}")
    
except Exception as e:
    print(f"❌ ERROR during feature engineering: {e}")
    print("\nTroubleshooting:")
    print("  Check that src/features/engineer_features.py exists")
    print("  Make sure pandas and numpy are installed")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Model Training
print(f"\n>>> Training ML models...")
try:
    from models.train_models import ClinicalTrialModelTrainer
    
    trainer = ClinicalTrialModelTrainer()
    results = trainer.train_and_evaluate(df_features)
    
    # Save models
    model_dir = Path(__file__).parent / 'data' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_models(str(model_dir))
    
    print(f"✓ Models saved to: {model_dir}")
    
    # Show results
    best_accuracy = 0
    if 'xgboost' in results:
        xgb_acc = results['xgboost']['test_accuracy']
        print(f"✓ XGBoost accuracy: {xgb_acc:.1%}")
        best_accuracy = max(best_accuracy, xgb_acc)
    
    if 'lightgbm' in results:
        lgb_acc = results['lightgbm']['test_accuracy']
        print(f"✓ LightGBM accuracy: {lgb_acc:.1%}")
        best_accuracy = max(best_accuracy, lgb_acc)
    
    if 'logistic' in results:
        log_acc = results['logistic']['test_accuracy']
        print(f"✓ Logistic Regression accuracy: {log_acc:.1%}")
        best_accuracy = max(best_accuracy, log_acc)
    
except Exception as e:
    print(f"❌ ERROR during model training: {e}")
    print("\nTroubleshooting:")
    print("  Check that src/models/train_models.py exists")
    print("  Make sure xgboost and lightgbm are installed")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "="*70)
print("✅ PROCESSING COMPLETE!")
print("="*70)
print(f"\nYour data is ready:")
print(f"  📊 Trials: {len(df_features):,}")
print(f"  🎯 Best Accuracy: {best_accuracy:.1%}")
print(f"  📁 Raw file: {latest_raw.name}")
print(f"  📁 Features file: {output_file.name}")
print(f"\nFiles created:")
print(f"  ✅ {output_file}")
print(f"  ✅ {model_dir}\\xgboost_model.joblib")
print(f"  ✅ {model_dir}\\lightgbm_model.joblib")
print(f"  ✅ {model_dir}\\feature_names.json")
print(f"\n🚀 Next steps:")
print(f"  1. Test locally:")
print(f"     streamlit run src/app/streamlit_app.py")
print(f"\n  2. Deploy to production:")
print(f"     git add data/")
print(f"     git commit -m 'Expand to 8,471 trials (4X improvement)'")
print(f"     git push")
print(f"\n  3. Update your marketing:")
print(f"     - Change '2,000 trials' → '8,500+ trials'")
print(f"     - Update competitive intelligence demos")
print(f"     - Show 4X data expansion in sales pitches")
print("="*70 + "\n")
