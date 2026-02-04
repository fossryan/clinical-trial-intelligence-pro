#!/usr/bin/env python3
"""
Clinical Trial Intelligence - Complete Data Pipeline Runner
Executes: Data Collection -> Feature Engineering -> Model Training

Usage:
    python run_pipeline.py                    # Full pipeline (10K trials)
    python run_pipeline.py --skip-collection  # Skip data collection (use existing)
    python run_pipeline.py --max-trials 2000  # Collect fewer trials for testing
    python run_pipeline.py --mode 10k         # Use enhanced 10K collector
"""

import sys
import argparse
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))


def run_pipeline(skip_collection=False, max_trials=10000, mode='10k'):
    """Run the complete data pipeline"""
    
    print("\n" + "="*70)
    print("CLINICAL TRIAL INTELLIGENCE - DATA PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Data Collection
    if not skip_collection:
        print("\n>>> STEP 1: Collecting Clinical Trial Data")
        print("-" * 70)
        
        if mode == '10k':
            # Use enhanced 10K collector
            print(f"Mode: Enhanced 10K Collection")
            print(f"Target: {max_trials:,} trials")
            print(f"Strategy: Multi-phase comprehensive collection")
            
            try:
                from data_collection.collect_trials_10k import EnhancedTrialCollector
                from datetime import datetime
                
                collector = EnhancedTrialCollector(rate_limit=0.5)
                df_raw = collector.collect_comprehensive_dataset(target_total=max_trials)
                
                # Save raw data
                output_dir = Path(__file__).parent / 'data' / 'raw'
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = output_dir / f'clinical_trials_raw_10k_{timestamp}.csv'
                df_raw.to_csv(output_file, index=False)
                
                print(f"\n✓ Data saved: {output_file}")
                print(f"✓ Trials collected: {len(df_raw):,}")
                
            except ImportError as e:
                print(f"\n⚠ Enhanced collector not found: {e}")
                print("Falling back to standard collector...")
                mode = 'standard'
        
        if mode == 'standard':
            # Use standard collector (2K trials, Phase 2-3 only)
            print(f"Mode: Standard Collection")
            print(f"Target: {max_trials:,} trials (Phase 2-3 only)")
            
            try:
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
                print(f"✓ Trials collected: {len(df_raw):,}")
                
            except ImportError as e:
                print(f"\n❌ ERROR: Could not import collector: {e}")
                print("\nTroubleshooting:")
                print("1. Make sure you're in the project root directory")
                print("2. Check that src/data_collection/collect_trials.py exists")
                print("3. Or use --skip-collection to skip this step")
                return False
    else:
        print("\n>>> STEP 1: Skipping data collection (using existing data)")
        print("-" * 70)
    
    # Step 2: Feature Engineering
    print("\n>>> STEP 2: Engineering Features")
    print("-" * 70)
    
    try:
        from features.engineer_features import FeatureEngineer
        from datetime import datetime
        import pandas as pd
        
        # Find most recent raw data file
        raw_dir = Path(__file__).parent / 'data' / 'raw'
        raw_files = list(raw_dir.glob('clinical_trials_raw*.csv'))
        
        if not raw_files:
            print("❌ ERROR: No raw data files found!")
            print(f"   Location: {raw_dir}")
            print("   Run without --skip-collection to collect data first")
            return False
        
        latest_raw = max(raw_files, key=lambda p: p.stat().st_mtime)
        print(f"Using data: {latest_raw.name}")
        
        df_raw = pd.read_csv(latest_raw)
        print(f"Loaded: {len(df_raw):,} trials")
        
        # Engineer features
        engineer = FeatureEngineer()
        df_features = engineer.create_all_features(df_raw)
        
        # Save processed data
        output_dir = Path(__file__).parent / 'data' / 'processed'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'clinical_trials_features_{timestamp}.csv'
        df_features.to_csv(output_file, index=False)
        
        print(f"\n✓ Features saved: {output_file}")
        print(f"✓ Features created: {df_features.shape[1]} columns")
        print(f"✓ Trials with features: {len(df_features):,}")
        
    except ImportError as e:
        print(f"\n❌ ERROR: Could not import feature engineer: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR during feature engineering: {e}")
        return False
    
    # Step 3: Model Training
    print("\n>>> STEP 3: Training ML Models")
    print("-" * 70)
    
    try:
        from models.train_models import ModelTrainer
        
        # Find most recent features file
        processed_dir = Path(__file__).parent / 'data' / 'processed'
        feature_files = list(processed_dir.glob('clinical_trials_features*.csv'))
        
        if not feature_files:
            print("❌ ERROR: No processed feature files found!")
            return False
        
        latest_features = max(feature_files, key=lambda p: p.stat().st_mtime)
        print(f"Using features: {latest_features.name}")
        
        df_features = pd.read_csv(latest_features)
        print(f"Loaded: {len(df_features):,} trials with {df_features.shape[1]} features")
        
        # Train models
        trainer = ModelTrainer()
        results = trainer.train_all_models(df_features)
        
        # Save models
        model_dir = Path(__file__).parent / 'data' / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        trainer.save_models(str(model_dir))
        
        print(f"\n✓ Models saved to: {model_dir}")
        print(f"✓ XGBoost accuracy: {results['xgboost']['test_accuracy']:.1%}")
        print(f"✓ LightGBM accuracy: {results['lightgbm']['test_accuracy']:.1%}")
        
    except ImportError as e:
        print(f"\n❌ ERROR: Could not import model trainer: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR during model training: {e}")
        return False
    
    # Success!
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print("\nYour app is ready with fresh data and models!")
    print(f"Total trials: {len(df_features):,}")
    print(f"Model accuracy: {results['xgboost']['test_accuracy']:.1%}")
    print("\nNext step: Deploy to Streamlit Cloud")
    print("  git add data/")
    print("  git commit -m 'Update with 10K+ trials'")
    print("  git push")
    print("="*70 + "\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run Clinical Trial Intelligence data pipeline'
    )
    parser.add_argument(
        '--skip-collection',
        action='store_true',
        help='Skip data collection step (use existing raw data)'
    )
    parser.add_argument(
        '--max-trials',
        type=int,
        default=10000,
        help='Maximum number of trials to collect (default: 10000)'
    )
    parser.add_argument(
        '--mode',
        choices=['10k', 'standard'],
        default='10k',
        help='Collection mode: 10k (enhanced) or standard (Phase 2-3 only)'
    )
    
    args = parser.parse_args()
    
    try:
        success = run_pipeline(
            skip_collection=args.skip_collection,
            max_trials=args.max_trials,
            mode=args.mode
        )
        
        if success:
            print("\n✅ SUCCESS: Pipeline completed successfully")
            sys.exit(0)
        else:
            print("\n❌ ERROR: Pipeline failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
