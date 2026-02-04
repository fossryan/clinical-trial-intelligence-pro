#!/usr/bin/env python3
"""
Clinical Trial Intelligence - Complete Data Pipeline Runner
Executes: Data Collection -> Feature Engineering -> Model Training

Usage:
    python run_pipeline.py                    # Full pipeline
    python run_pipeline.py --skip-collection  # Skip data collection (use existing)
    python run_pipeline.py --max-trials 500   # Collect fewer trials for testing
"""

import sys
import argparse
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))


def run_pipeline(skip_collection=False, max_trials=2000):
    """Run the complete data pipeline"""
    
    print("\n" + "="*70)
    print("CLINICAL TRIAL INTELLIGENCE - DATA PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Data Collection
    if not skip_collection:
        print("\n>>> STEP 1: Collecting Clinical Trial Data")
        print("-" * 70)
        from data_collection.collect_trials import ClinicalTrialsCollector
        from datetime import datetime
        
        collector = ClinicalTrialsCollector(rate_limit=0.5)
        df_raw = collector.fetch_trials(
            phases=['PHASE2', 'PHASE3'],
            statuses=['COMPLETED', 'TERMINATED', 'ACTIVE_NOT_RECRUITING', 'WITHDRAWN'],
            max_studies=max_trials,
            study_type='INTERVENTIONAL'
        )
        
        # Save raw data
        output_dir = Path(__file__).parent / 'data' / 'raw'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'clinical_trials_raw_{timestamp}.csv'
        df_raw.to_csv(output_file, index=False)
        
        print(f"\n✓ Data saved: {output_file}")
        print(f"✓ Collected {len(df_raw)} trials")
    else:
        print("\n>>> STEP 1: Skipping data collection (using existing data)")
        print("-" * 70)
    
    # Step 2: Feature Engineering
    print("\n>>> STEP 2: Engineering Features")
    print("-" * 70)
    from features.engineer_features import TrialFeatureEngineer
    import pandas as pd
    from datetime import datetime
    
    # Load most recent raw data
    data_dir = Path(__file__).parent / 'data' / 'raw'
    raw_files = list(data_dir.glob('clinical_trials_raw_*.csv'))
    
    if not raw_files:
        print("ERROR: No raw data found. Please run data collection first.")
        return False
    
    latest_file = max(raw_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest_file.name}")
    df_raw = pd.read_csv(latest_file)
    
    engineer = TrialFeatureEngineer()
    df_features = engineer.create_features(df_raw)
    
    # Save processed data
    output_dir = Path(__file__).parent / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'clinical_trials_features_{timestamp}.csv'
    df_features.to_csv(output_file, index=False)
    
    print(f"\n✓ Features saved: {output_file}")
    print(f"✓ Total features: {df_features.shape[1]}")
    print(f"✓ Target distribution:")
    print(df_features['trial_success'].value_counts())
    
    # Step 3: Model Training
    print("\n>>> STEP 3: Training Prediction Models")
    print("-" * 70)
    from models.train_models import TrialRiskPredictor
    import joblib
    import json
    
    # Load most recent feature data
    data_dir = Path(__file__).parent / 'data' / 'processed'
    feature_files = list(data_dir.glob('clinical_trials_features_*.csv'))
    
    if not feature_files:
        print("ERROR: No feature data found.")
        return False
    
    latest_file = max(feature_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest_file.name}")
    df = pd.read_csv(latest_file)
    
    predictor = TrialRiskPredictor(random_state=42)
    
    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(df, balance_classes=True)
    
    # Train models
    print("\nTraining XGBoost...")
    predictor.train_xgboost(X_train, y_train, X_test, y_test)
    
    print("\nTraining LightGBM...")
    predictor.train_lightgbm(X_train, y_train, X_test, y_test)
    
    print("\nTraining Logistic Regression...")
    predictor.train_logistic_regression(X_train, y_train, X_test, y_test)
    
    # Save models
    model_dir = Path(__file__).parent / 'data' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save XGBoost
    xgb_path = model_dir / f'xgboost_{timestamp}.joblib'
    joblib.dump(predictor.models['xgboost'], xgb_path)
    print(f"\n✓ XGBoost saved: {xgb_path}")
    
    # Save LightGBM
    lgb_path = model_dir / f'lightgbm_{timestamp}.joblib'
    joblib.dump(predictor.models['lightgbm'], lgb_path)
    print(f"✓ LightGBM saved: {lgb_path}")
    
    # Save feature names
    features_path = model_dir / f'feature_names_{timestamp}.json'
    with open(features_path, 'w') as f:
        json.dump(predictor.feature_names, f)
    print(f"✓ Features saved: {features_path}")
    
    # Save metrics
    metrics_path = model_dir / f'metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(predictor.metrics, f, indent=2)
    print(f"✓ Metrics saved: {metrics_path}")
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\n📊 Model Performance Summary:")
    for model_name, metrics in predictor.metrics.items():
        print(f"\n{model_name.upper()}:")
        print(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
    
    print("\n✅ Ready to launch Streamlit app!")
    print("   Run: streamlit run src/app/streamlit_app.py")
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Clinical Trial Intelligence pipeline')
    parser.add_argument('--skip-collection', action='store_true',
                        help='Skip data collection, use existing raw data')
    parser.add_argument('--max-trials', type=int, default=2000,
                        help='Maximum number of trials to collect (default: 2000)')
    
    args = parser.parse_args()
    
    success = run_pipeline(
        skip_collection=args.skip_collection,
        max_trials=args.max_trials
    )
    
    sys.exit(0 if success else 1)
