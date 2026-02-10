"""
Automated Model Retraining Pipeline
Monitors model performance and triggers retraining when degradation is detected
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import json
import sys
from typing import Dict, List, Tuple, Optional
import shutil

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from models.train_models import TrialRiskPredictor


class PerformanceMonitor:
    """
    Monitor model performance and detect degradation
    """
    
    def __init__(self, metrics_file: str = "data/models/performance_history.json"):
        self.metrics_file = Path(metrics_file)
        self.performance_history = self._load_history()
        
        # Performance thresholds for retraining
        self.thresholds = {
            'accuracy_drop': 0.05,      # Retrain if accuracy drops by 5%
            'roc_auc_drop': 0.05,       # Retrain if ROC-AUC drops by 5%
            'f1_drop': 0.05,            # Retrain if F1 drops by 5%
            'min_accuracy': 0.75,       # Retrain if accuracy falls below 75%
            'min_roc_auc': 0.70,        # Retrain if ROC-AUC falls below 70%
            'days_since_training': 90   # Retrain if model is older than 90 days
        }
    
    def _load_history(self) -> Dict:
        """Load performance history from file"""
        if not self.metrics_file.exists():
            return {
                'models': {},
                'retraining_events': []
            }
        
        with open(self.metrics_file, 'r') as f:
            return json.load(f)
    
    def _save_history(self):
        """Save performance history to file"""
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
    
    def record_performance(
        self,
        model_name: str,
        metrics: Dict,
        model_path: str,
        dataset_size: int
    ):
        """Record model performance metrics"""
        
        timestamp = datetime.now().isoformat()
        
        performance_record = {
            'timestamp': timestamp,
            'metrics': metrics,
            'model_path': model_path,
            'dataset_size': dataset_size
        }
        
        if model_name not in self.performance_history['models']:
            self.performance_history['models'][model_name] = []
        
        self.performance_history['models'][model_name].append(performance_record)
        
        # Keep only last 50 records per model
        if len(self.performance_history['models'][model_name]) > 50:
            self.performance_history['models'][model_name] = \
                self.performance_history['models'][model_name][-50:]
        
        self._save_history()
        
        print(f"✅ Recorded performance for {model_name}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"   F1 Score: {metrics['f1_score']:.4f}")
    
    def check_performance_degradation(self, model_name: str) -> Tuple[bool, List[str]]:
        """
        Check if model performance has degraded
        
        Returns:
            (should_retrain, reasons)
        """
        
        if model_name not in self.performance_history['models']:
            return False, ["No performance history available"]
        
        history = self.performance_history['models'][model_name]
        
        if len(history) < 2:
            return False, ["Insufficient history for comparison"]
        
        latest = history[-1]
        baseline = history[0]  # First recorded performance (best baseline)
        
        reasons = []
        should_retrain = False
        
        # Check 1: Absolute performance thresholds
        if latest['metrics']['accuracy'] < self.thresholds['min_accuracy']:
            reasons.append(f"Accuracy below threshold: {latest['metrics']['accuracy']:.4f} < {self.thresholds['min_accuracy']}")
            should_retrain = True
        
        if latest['metrics']['roc_auc'] < self.thresholds['min_roc_auc']:
            reasons.append(f"ROC-AUC below threshold: {latest['metrics']['roc_auc']:.4f} < {self.thresholds['min_roc_auc']}")
            should_retrain = True
        
        # Check 2: Performance degradation from baseline
        accuracy_drop = baseline['metrics']['accuracy'] - latest['metrics']['accuracy']
        if accuracy_drop > self.thresholds['accuracy_drop']:
            reasons.append(f"Accuracy dropped by {accuracy_drop:.4f} (threshold: {self.thresholds['accuracy_drop']})")
            should_retrain = True
        
        roc_auc_drop = baseline['metrics']['roc_auc'] - latest['metrics']['roc_auc']
        if roc_auc_drop > self.thresholds['roc_auc_drop']:
            reasons.append(f"ROC-AUC dropped by {roc_auc_drop:.4f} (threshold: {self.thresholds['roc_auc_drop']})")
            should_retrain = True
        
        f1_drop = baseline['metrics']['f1_score'] - latest['metrics']['f1_score']
        if f1_drop > self.thresholds['f1_drop']:
            reasons.append(f"F1 score dropped by {f1_drop:.4f} (threshold: {self.thresholds['f1_drop']})")
            should_retrain = True
        
        # Check 3: Time since last training
        latest_timestamp = datetime.fromisoformat(latest['timestamp'])
        days_old = (datetime.now() - latest_timestamp).days
        
        if days_old > self.thresholds['days_since_training']:
            reasons.append(f"Model is {days_old} days old (threshold: {self.thresholds['days_since_training']} days)")
            should_retrain = True
        
        return should_retrain, reasons
    
    def record_retraining_event(
        self,
        model_name: str,
        reason: str,
        old_metrics: Dict,
        new_metrics: Dict
    ):
        """Record a model retraining event"""
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'reason': reason,
            'old_metrics': old_metrics,
            'new_metrics': new_metrics,
            'improvement': {
                'accuracy': new_metrics['accuracy'] - old_metrics['accuracy'],
                'roc_auc': new_metrics['roc_auc'] - old_metrics['roc_auc'],
                'f1_score': new_metrics['f1_score'] - old_metrics['f1_score']
            }
        }
        
        self.performance_history['retraining_events'].append(event)
        self._save_history()
        
        print("\n" + "="*80)
        print("RETRAINING EVENT RECORDED")
        print("="*80)
        print(f"Model: {model_name}")
        print(f"Reason: {reason}")
        print(f"\nPerformance Improvement:")
        print(f"  Accuracy: {old_metrics['accuracy']:.4f} → {new_metrics['accuracy']:.4f} ({event['improvement']['accuracy']:+.4f})")
        print(f"  ROC-AUC:  {old_metrics['roc_auc']:.4f} → {new_metrics['roc_auc']:.4f} ({event['improvement']['roc_auc']:+.4f})")
        print(f"  F1 Score: {old_metrics['f1_score']:.4f} → {new_metrics['f1_score']:.4f} ({event['improvement']['f1_score']:+.4f})")
        print("="*80 + "\n")
    
    def get_latest_metrics(self, model_name: str) -> Optional[Dict]:
        """Get latest performance metrics for a model"""
        if model_name not in self.performance_history['models']:
            return None
        
        if not self.performance_history['models'][model_name]:
            return None
        
        return self.performance_history['models'][model_name][-1]['metrics']


class ModelRetrainingPipeline:
    """
    Automated pipeline for model retraining
    """
    
    def __init__(
        self,
        data_dir: str = "data/processed",
        model_dir: str = "data/models",
        backup_dir: str = "data/models/backups"
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.backup_dir = Path(backup_dir)
        
        self.monitor = PerformanceMonitor()
        
        # Create directories
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def backup_current_models(self):
        """Backup current models before retraining"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Backup all model files
        model_files = list(self.model_dir.glob("*.joblib")) + \
                     list(self.model_dir.glob("*.json"))
        
        for model_file in model_files:
            if model_file.is_file() and 'backup' not in str(model_file):
                shutil.copy2(model_file, backup_path / model_file.name)
        
        print(f"✅ Backed up {len(model_files)} model files to {backup_path}")
        
        return backup_path
    
    def load_latest_data(self) -> pd.DataFrame:
        """Load the most recent processed dataset"""
        
        feature_files = list(self.data_dir.glob("clinical_trials_features_*.csv"))
        
        if not feature_files:
            raise FileNotFoundError(f"No feature files found in {self.data_dir}")
        
        # Get most recent file
        latest_file = max(feature_files, key=lambda p: p.stat().st_mtime)
        
        print(f"Loading data from: {latest_file.name}")
        df = pd.read_csv(latest_file)
        print(f"Loaded {len(df):,} trials")
        
        return df
    
    def evaluate_model(self, model, X_test, y_test) -> Dict:
        """Evaluate model and return metrics"""
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred))
        }
        
        return metrics
    
    def train_new_models(self, df: pd.DataFrame) -> Dict:
        """Train new models and return metrics"""
        
        print("\n" + "="*80)
        print("TRAINING NEW MODELS")
        print("="*80 + "\n")
        
        # Initialize trainer
        trainer = TrialRiskPredictor(random_state=42)
        
        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(df, balance_classes=True)
        
        # Train models
        print("\nTraining XGBoost...")
        trainer.train_xgboost(X_train, y_train)
        
        print("\nTraining LightGBM...")
        trainer.train_lightgbm(X_train, y_train)
        
        print("\nTraining Logistic Regression...")
        trainer.train_logistic_regression(X_train, y_train)
        
        # Evaluate models
        print("\nEvaluating models...")
        xgb_metrics = self.evaluate_model(trainer.models['xgboost'], X_test, y_test)
        lgb_metrics = self.evaluate_model(trainer.models['lightgbm'], X_test, y_test)
        lr_metrics = self.evaluate_model(trainer.models['logistic_regression'], X_test, y_test)
        
        # Save models
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        trainer.save_models(self.model_dir, timestamp=timestamp)
        
        metrics = {
            'xgboost': xgb_metrics,
            'lightgbm': lgb_metrics,
            'logistic_regression': lr_metrics
        }
        
        return metrics, len(df)
    
    def should_deploy_new_model(
        self,
        old_metrics: Dict,
        new_metrics: Dict,
        min_improvement: float = 0.01
    ) -> Tuple[bool, str]:
        """
        Decide whether to deploy new model
        
        Args:
            old_metrics: Performance of current model
            new_metrics: Performance of newly trained model
            min_improvement: Minimum improvement required to deploy
        
        Returns:
            (should_deploy, reason)
        """
        
        if old_metrics is None:
            return True, "No existing model found - deploying new model"
        
        # Compare key metrics
        accuracy_improvement = new_metrics['accuracy'] - old_metrics['accuracy']
        roc_auc_improvement = new_metrics['roc_auc'] - old_metrics['roc_auc']
        f1_improvement = new_metrics['f1_score'] - old_metrics['f1_score']
        
        # Deploy if any metric improved significantly
        if accuracy_improvement >= min_improvement:
            return True, f"Accuracy improved by {accuracy_improvement:.4f}"
        
        if roc_auc_improvement >= min_improvement:
            return True, f"ROC-AUC improved by {roc_auc_improvement:.4f}"
        
        if f1_improvement >= min_improvement:
            return True, f"F1 score improved by {f1_improvement:.4f}"
        
        # Don't deploy if performance degraded
        if accuracy_improvement < -min_improvement/2:
            return False, f"Accuracy degraded by {-accuracy_improvement:.4f}"
        
        if roc_auc_improvement < -min_improvement/2:
            return False, f"ROC-AUC degraded by {-roc_auc_improvement:.4f}"
        
        # Deploy if roughly same performance (data drift accommodation)
        if abs(accuracy_improvement) < min_improvement and \
           abs(roc_auc_improvement) < min_improvement:
            return True, "Similar performance - updating for data freshness"
        
        return False, "Insufficient improvement to justify deployment"
    
    def run_retraining_pipeline(
        self,
        force_retrain: bool = False
    ) -> Dict:
        """
        Run complete retraining pipeline
        
        Args:
            force_retrain: Force retraining even if performance is good
        
        Returns:
            Dictionary with pipeline results
        """
        
        print("\n" + "="*80)
        print("MODEL RETRAINING PIPELINE")
        print("="*80 + "\n")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'retrained': False,
            'deployed': False,
            'reason': None,
            'metrics': {}
        }
        
        try:
            # Step 1: Load latest data
            print("\n[Step 1/5] Loading latest data...")
            df = self.load_latest_data()
            
            # Step 2: Check if retraining needed
            print("\n[Step 2/5] Checking if retraining needed...")
            
            old_metrics = self.monitor.get_latest_metrics('lightgbm')
            
            if not force_retrain:
                should_retrain, reasons = self.monitor.check_performance_degradation('lightgbm')
                
                if not should_retrain:
                    print("✅ Model performance is good - no retraining needed")
                    print(f"   Current accuracy: {old_metrics['accuracy']:.4f}")
                    print(f"   Current ROC-AUC: {old_metrics['roc_auc']:.4f}")
                    results['reason'] = "Performance is acceptable"
                    return results
                
                print("⚠️  Retraining triggered!")
                for reason in reasons:
                    print(f"   - {reason}")
                results['reason'] = '; '.join(reasons)
            else:
                print("🔄 Force retraining enabled")
                results['reason'] = "Force retrain requested"
            
            # Step 3: Backup current models
            print("\n[Step 3/5] Backing up current models...")
            backup_path = self.backup_current_models()
            results['backup_path'] = str(backup_path)
            
            # Step 4: Train new models
            print("\n[Step 4/5] Training new models...")
            new_metrics, dataset_size = self.train_new_models(df)
            results['retrained'] = True
            results['metrics']['new'] = new_metrics
            results['metrics']['old'] = old_metrics
            
            # Step 5: Decide deployment
            print("\n[Step 5/5] Evaluating new models...")
            
            should_deploy, deploy_reason = self.should_deploy_new_model(
                old_metrics,
                new_metrics['lightgbm']
            )
            
            if should_deploy:
                print(f"✅ Deploying new model: {deploy_reason}")
                
                # Record performance
                for model_name, metrics in new_metrics.items():
                    self.monitor.record_performance(
                        model_name,
                        metrics,
                        str(self.model_dir / f"{model_name}_model.joblib"),
                        dataset_size
                    )
                
                # Record retraining event
                if old_metrics:
                    self.monitor.record_retraining_event(
                        'lightgbm',
                        results['reason'],
                        old_metrics,
                        new_metrics['lightgbm']
                    )
                
                results['deployed'] = True
                results['deploy_reason'] = deploy_reason
            else:
                print(f"❌ Not deploying: {deploy_reason}")
                print("   Restoring backup...")
                
                # Restore backup
                for backup_file in backup_path.glob("*"):
                    shutil.copy2(backup_file, self.model_dir / backup_file.name)
                
                results['deployed'] = False
                results['deploy_reason'] = deploy_reason
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETE")
            print("="*80)
            print(f"Retrained: {results['retrained']}")
            print(f"Deployed: {results['deployed']}")
            
            return results
        
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            results['error'] = str(e)
            raise


def main():
    """Main function for command-line execution"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Retraining Pipeline')
    parser.add_argument('--force', action='store_true', help='Force retraining')
    parser.add_argument('--check-only', action='store_true', help='Only check if retraining needed')
    
    args = parser.parse_args()
    
    pipeline = ModelRetrainingPipeline()
    
    if args.check_only:
        print("Checking if retraining needed...")
        should_retrain, reasons = pipeline.monitor.check_performance_degradation('lightgbm')
        
        if should_retrain:
            print("\n⚠️  Retraining recommended:")
            for reason in reasons:
                print(f"   - {reason}")
            sys.exit(1)  # Exit with error code to trigger retraining
        else:
            print("\n✅ No retraining needed")
            sys.exit(0)
    else:
        results = pipeline.run_retraining_pipeline(force_retrain=args.force)
        
        # Save results
        results_file = Path("data/models/last_retraining_result.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
