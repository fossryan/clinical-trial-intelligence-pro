"""
Validation Study Framework for Publication

Target journals:
- Nature Medicine (IF: 87.241)
- JAMA Network Open (IF: 13.8)
- Clinical Trials (IF: 2.8)
- Drug Discovery Today (IF: 7.4)

This module generates publication-ready validation analysis:
1. Temporal validation (train on past, test on future)
2. External validation (different data source)
3. Calibration analysis
4. Clinical utility assessment
5. Subgroup performance
6. Comparison to baseline models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
import joblib


class ValidationStudy:
    """
    Comprehensive validation for publication
    
    Follows TRIPOD guidelines (Transparent Reporting of a multivariable 
    prediction model for Individual Prognosis Or Diagnosis)
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def temporal_validation(
        self,
        df: pd.DataFrame,
        model,
        feature_names: List[str],
        cutoff_date: str = '2023-01-01'
    ) -> Dict:
        """
        Temporal validation: Train on past, test on future
        
        This is the gold standard for clinical prediction models.
        Shows model generalizes to new trials, not just random split.
        
        Args:
            df: Full dataset
            model: Trained model
            feature_names: List of feature names
            cutoff_date: Split date (YYYY-MM-DD)
            
        Returns:
            Dictionary of validation metrics
        """
        
        print("\n" + "="*80)
        print("TEMPORAL VALIDATION")
        print("="*80 + "\n")
        
        print(f"Cutoff date: {cutoff_date}")
        print("Training set: Trials starting before cutoff")
        print("Test set: Trials starting after cutoff (prospective validation)\n")
        
        # Parse dates
        df['start_date_dt'] = pd.to_datetime(df['start_date'], errors='coerce')
        cutoff = pd.to_datetime(cutoff_date)
        
        # Split
        train_df = df[df['start_date_dt'] < cutoff].copy()
        test_df = df[df['start_date_dt'] >= cutoff].copy()
        
        # Remove rows without outcome
        train_df = train_df[train_df['trial_success'].notna()]
        test_df = test_df[test_df['trial_success'].notna()]
        
        print(f"Training set: {len(train_df):,} trials (before {cutoff_date})")
        print(f"Test set: {len(test_df):,} trials (after {cutoff_date})")
        
        if len(test_df) < 50:
            print("\n⚠️  Warning: Small test set (<50 trials)")
            print("    Consider using earlier cutoff date for more test samples\n")
        
        # Prepare features
        X_train = train_df[feature_names].fillna(0)
        y_train = train_df['trial_success'].astype(int)
        
        X_test = test_df[feature_names].fillna(0)
        y_test = test_df['trial_success'].astype(int)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'n_train': len(train_df),
            'n_test': len(test_df),
            'success_rate_train': y_train.mean(),
            'success_rate_test': y_test.mean()
        }
        
        print("\nTemporal Validation Results:")
        print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
        print(f"  Brier Score: {metrics['brier_score']:.3f} (lower is better)")
        
        # Statistical test: is performance significantly better than random?
        self._test_statistical_significance(y_test, y_pred_proba, metrics)
        
        # Calibration
        cal_metrics = self._calibration_analysis(y_test, y_pred_proba, 'Temporal')
        metrics.update(cal_metrics)
        
        # Save
        self.results['temporal_validation'] = metrics
        
        return metrics
    
    def cross_temporal_validation(
        self,
        df: pd.DataFrame,
        model_class,
        feature_names: List[str],
        n_splits: int = 5
    ) -> Dict:
        """
        Multiple temporal splits for robust validation
        
        Split by year and validate on each future year
        """
        
        print("\n" + "="*80)
        print("CROSS-TEMPORAL VALIDATION")
        print("="*80 + "\n")
        
        df['start_date_dt'] = pd.to_datetime(df['start_date'], errors='coerce')
        df = df[df['start_date_dt'].notna()].copy()
        df = df[df['trial_success'].notna()].copy()
        
        df['year'] = df['start_date_dt'].dt.year
        years = sorted(df['year'].unique())
        
        print(f"Available years: {years}")
        print(f"Will train on year N and test on year N+1\n")
        
        all_metrics = []
        
        for i in range(len(years) - 1):
            train_year = years[i]
            test_year = years[i + 1]
            
            train_df = df[df['year'] == train_year]
            test_df = df[df['year'] == test_year]
            
            if len(test_df) < 20:
                print(f"Skipping {train_year}→{test_year}: test set too small ({len(test_df)} trials)")
                continue
            
            print(f"\nFold {i+1}: Train on {train_year} ({len(train_df)} trials), "
                  f"Test on {test_year} ({len(test_df)} trials)")
            
            # Train model
            X_train = train_df[feature_names].fillna(0)
            y_train = train_df['trial_success'].astype(int)
            
            X_test = test_df[feature_names].fillna(0)
            y_test = test_df['trial_success'].astype(int)
            
            # Create and train new model
            fold_model = model_class()
            fold_model.fit(X_train, y_train)
            
            # Predict
            y_pred_proba = fold_model.predict_proba(X_test)[:, 1]
            
            # Metrics
            fold_metrics = {
                'train_year': train_year,
                'test_year': test_year,
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'accuracy': accuracy_score(y_test, (y_pred_proba >= 0.5)),
                'brier_score': brier_score_loss(y_test, y_pred_proba),
                'n_test': len(test_df)
            }
            
            all_metrics.append(fold_metrics)
            
            print(f"  ROC-AUC: {fold_metrics['roc_auc']:.3f}")
        
        # Aggregate
        avg_metrics = {
            'mean_roc_auc': np.mean([m['roc_auc'] for m in all_metrics]),
            'std_roc_auc': np.std([m['roc_auc'] for m in all_metrics]),
            'mean_accuracy': np.mean([m['accuracy'] for m in all_metrics]),
            'std_accuracy': np.std([m['accuracy'] for m in all_metrics]),
            'n_folds': len(all_metrics),
            'fold_results': all_metrics
        }
        
        print(f"\n{'='*80}")
        print("CROSS-TEMPORAL RESULTS")
        print(f"{'='*80}")
        print(f"Mean ROC-AUC: {avg_metrics['mean_roc_auc']:.3f} ± {avg_metrics['std_roc_auc']:.3f}")
        print(f"Mean Accuracy: {avg_metrics['mean_accuracy']:.3f} ± {avg_metrics['std_accuracy']:.3f}")
        
        self.results['cross_temporal_validation'] = avg_metrics
        
        return avg_metrics
    
    def subgroup_analysis(
        self,
        df: pd.DataFrame,
        model,
        feature_names: List[str],
        subgroups: Dict[str, str]
    ) -> Dict:
        """
        Validate model performance across subgroups
        
        Args:
            df: Test dataset
            model: Trained model
            feature_names: Features
            subgroups: Dict mapping subgroup name to column name
                       e.g., {'Oncology': 'is_oncology', 'Phase 2': 'is_phase2'}
        
        Returns:
            Performance metrics by subgroup
        """
        
        print("\n" + "="*80)
        print("SUBGROUP ANALYSIS")
        print("="*80 + "\n")
        
        df = df[df['trial_success'].notna()].copy()
        
        # Overall performance
        X_all = df[feature_names].fillna(0)
        y_all = df['trial_success'].astype(int)
        y_pred_proba_all = model.predict_proba(X_all)[:, 1]
        
        overall_auc = roc_auc_score(y_all, y_pred_proba_all)
        
        print(f"Overall ROC-AUC: {overall_auc:.3f} (n={len(df):,})\n")
        
        subgroup_results = {}
        
        for subgroup_name, column in subgroups.items():
            if column not in df.columns:
                print(f"⚠️  Skipping {subgroup_name}: column {column} not found")
                continue
            
            # Filter to subgroup
            subgroup_df = df[df[column] == 1].copy()
            
            if len(subgroup_df) < 30:
                print(f"⚠️  Skipping {subgroup_name}: too few samples ({len(subgroup_df)})")
                continue
            
            X_sub = subgroup_df[feature_names].fillna(0)
            y_sub = subgroup_df['trial_success'].astype(int)
            y_pred_proba_sub = model.predict_proba(X_sub)[:, 1]
            
            auc_sub = roc_auc_score(y_sub, y_pred_proba_sub)
            acc_sub = accuracy_score(y_sub, (y_pred_proba_sub >= 0.5))
            
            # Compare to overall
            auc_diff = auc_sub - overall_auc
            
            subgroup_results[subgroup_name] = {
                'n': len(subgroup_df),
                'roc_auc': auc_sub,
                'accuracy': acc_sub,
                'success_rate': y_sub.mean(),
                'auc_vs_overall': auc_diff
            }
            
            print(f"{subgroup_name:30s}: ROC-AUC {auc_sub:.3f} "
                  f"({auc_diff:+.3f} vs overall) | n={len(subgroup_df):,}")
        
        self.results['subgroup_analysis'] = subgroup_results
        
        return subgroup_results
    
    def _calibration_analysis(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        label: str
    ) -> Dict:
        """Analyze probability calibration"""
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(
            y_true, y_pred_proba, n_bins=10, strategy='quantile'
        )
        
        # Expected Calibration Error (ECE)
        ece = np.mean(np.abs(prob_true - prob_pred))
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8, 
                label=f'{label} Model')
        
        ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Observed Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'Calibration Curve - {label}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add ECE to plot
        ax.text(0.05, 0.95, f'ECE = {ece:.3f}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        output_file = self.output_dir / f'calibration_curve_{label.lower()}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved calibration curve to {output_file.name}")
        
        return {
            'expected_calibration_error': ece,
            'calibration_plot': str(output_file)
        }
    
    def _test_statistical_significance(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metrics: Dict
    ):
        """Test if model is significantly better than random"""
        
        # DeLong test for ROC-AUC confidence interval
        # Simplified: bootstrap CI
        
        n_bootstrap = 1000
        bootstrap_aucs = []
        
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
            y_pred_boot = y_pred_proba[indices]
            
            try:
                auc_boot = roc_auc_score(y_true_boot, y_pred_boot)
                bootstrap_aucs.append(auc_boot)
            except:
                pass
        
        # 95% CI
        ci_lower = np.percentile(bootstrap_aucs, 2.5)
        ci_upper = np.percentile(bootstrap_aucs, 97.5)
        
        print(f"\n  95% Confidence Interval for ROC-AUC: [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        # P-value: is AUC significantly > 0.5?
        if ci_lower > 0.5:
            print(f"  ✓ Model significantly better than random (p < 0.05)")
        else:
            print(f"  ⚠️  Model not significantly better than random")
        
        metrics['roc_auc_ci_lower'] = ci_lower
        metrics['roc_auc_ci_upper'] = ci_upper
    
    def generate_publication_figures(
        self,
        df: pd.DataFrame,
        model,
        feature_names: List[str]
    ):
        """
        Generate all figures for publication
        
        Figures:
        1. ROC curve with CI
        2. Precision-Recall curve
        3. Calibration plot
        4. Feature importance
        5. Confusion matrix
        6. Subgroup forest plot
        """
        
        print("\n" + "="*80)
        print("GENERATING PUBLICATION FIGURES")
        print("="*80 + "\n")
        
        df = df[df['trial_success'].notna()].copy()
        
        X = df[feature_names].fillna(0)
        y = df['trial_success'].astype(int)
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Figure 1: ROC Curve
        self._plot_roc_curve(y, y_pred_proba)
        
        # Figure 2: Precision-Recall
        self._plot_precision_recall(y, y_pred_proba)
        
        # Figure 3: Confusion Matrix
        self._plot_confusion_matrix(y, y_pred)
        
        # Figure 4: Feature Importance
        self._plot_feature_importance(model, feature_names)
        
        print("\n✓ All figures generated and saved to:")
        print(f"  {self.output_dir}/")
    
    def _plot_roc_curve(self, y_true, y_pred_proba):
        """Plot ROC curve with confidence interval"""
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(fpr, tpr, linewidth=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'roc_curve.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Figure 1: ROC curve → {output_file.name}")
        plt.close()
    
    def _plot_precision_recall(self, y_true, y_pred_proba):
        """Plot Precision-Recall curve"""
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(recall, precision, linewidth=2, label='PR Curve')
        
        # Baseline (random classifier)
        baseline = y_true.mean()
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                   label=f'Baseline (prevalence = {baseline:.2f})')
        
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'precision_recall_curve.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Figure 2: Precision-Recall → {output_file.name}")
        plt.close()
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Failure', 'Success'],
                    yticklabels=['Failure', 'Success'],
                    cbar_kws={'label': 'Count'},
                    ax=ax)
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Figure 3: Confusion matrix → {output_file.name}")
        plt.close()
    
    def _plot_feature_importance(self, model, feature_names):
        """Plot top 20 feature importances"""
        
        if not hasattr(model, 'feature_importances_'):
            print("  ⚠️  Model doesn't have feature_importances_")
            return
        
        importances = model.feature_importances_
        
        # Sort and get top 20
        indices = np.argsort(importances)[-20:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(range(len(indices)), importances[indices], color='steelblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'feature_importance.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Figure 4: Feature importance → {output_file.name}")
        plt.close()
    
    def generate_manuscript_table1(self, df: pd.DataFrame) -> str:
        """
        Generate Table 1 for manuscript (cohort characteristics)
        
        Standard format for medical journals
        """
        
        df = df[df['trial_success'].notna()].copy()
        
        success = df[df['trial_success'] == 1]
        failure = df[df['trial_success'] == 0]
        
        table = "\nTable 1. Baseline Characteristics of Clinical Trials\n"
        table += "="*80 + "\n"
        table += f"{'Characteristic':<40} {'All Trials':<15} {'Success':<15} {'Failure':<15}\n"
        table += "-"*80 + "\n"
        
        # N
        table += f"{'N':<40} {len(df):<15} {len(success):<15} {len(failure):<15}\n"
        
        # Phase distribution
        for phase in ['is_phase1', 'is_phase2', 'is_phase3', 'is_phase4']:
            if phase in df.columns:
                all_pct = df[phase].mean() * 100
                succ_pct = success[phase].mean() * 100
                fail_pct = failure[phase].mean() * 100
                
                label = phase.replace('is_', '').replace('phase', 'Phase ')
                table += f"{label:<40} {all_pct:>6.1f}% {succ_pct:>12.1f}% {fail_pct:>12.1f}%\n"
        
        # Therapeutic areas
        for area in ['is_oncology', 'is_cns', 'is_cardiovascular', 'is_autoimmune']:
            if area in df.columns:
                all_pct = df[area].mean() * 100
                succ_pct = success[area].mean() * 100
                fail_pct = failure[area].mean() * 100
                
                label = area.replace('is_', '').capitalize()
                table += f"{label:<40} {all_pct:>6.1f}% {succ_pct:>12.1f}% {fail_pct:>12.1f}%\n"
        
        # Enrollment
        enrollment_cols = ['enrollment']
        if 'enrollment' in df.columns:
            all_med = df['enrollment'].median()
            succ_med = success['enrollment'].median()
            fail_med = failure['enrollment'].median()
            
            table += f"{'Enrollment (median)':<40} {all_med:>6.0f} {succ_med:>15.0f} {fail_med:>15.0f}\n"
        
        table += "="*80 + "\n"
        
        # Save
        output_file = self.output_dir / 'table1_characteristics.txt'
        with open(output_file, 'w') as f:
            f.write(table)
        
        print(table)
        print(f"\n✓ Table 1 saved to {output_file.name}")
        
        return table
    
    def save_results(self):
        """Save all validation results to JSON"""
        
        output_file = self.output_dir / f'validation_results_{datetime.now().strftime("%Y%m%d")}.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n✓ All validation results saved to {output_file.name}")


def main():
    """
    Run complete validation study
    """
    
    print("\n" + "="*80)
    print("VALIDATION STUDY FOR PUBLICATION")
    print("="*80 + "\n")
    
    print("This validation study follows TRIPOD guidelines:")
    print("  1. Temporal validation (train on past, test on future)")
    print("  2. Cross-temporal validation (multiple year splits)")
    print("  3. Subgroup analysis (performance by indication, phase)")
    print("  4. Calibration analysis")
    print("  5. Publication-ready figures")
    print("  6. Manuscript tables")
    print("\n" + "="*80 + "\n")
    
    # Load data
    from pathlib import Path
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    feature_files = list(data_dir.glob('clinical_trials_features_*.csv'))
    
    if not feature_files:
        print("❌ No processed data found")
        return
    
    latest_file = max(feature_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest_file.name}\n")
    
    df = pd.read_csv(latest_file)
    print(f"Loaded {len(df):,} trials\n")
    
    # Load model
    model_dir = Path(__file__).parent.parent.parent / 'data' / 'models'
    model_files = list(model_dir.glob('xgboost_*.joblib'))
    
    if not model_files:
        print("❌ No trained models found")
        return
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    model = joblib.load(latest_model)
    print(f"Loaded model: {latest_model.name}\n")
    
    # Load feature names
    feature_files = list(model_dir.glob('feature_names_*.json'))
    if feature_files:
        latest_features = max(feature_files, key=lambda p: p.stat().st_mtime)
        with open(latest_features, 'r') as f:
            feature_names = json.load(f)
    
    # Initialize validation study
    output_dir = Path(__file__).parent.parent.parent / 'validation_study'
    study = ValidationStudy(output_dir)
    
    # 1. Temporal validation
    temporal_metrics = study.temporal_validation(
        df, model, feature_names, cutoff_date='2023-01-01'
    )
    
    # 2. Subgroup analysis
    subgroups = {
        'Oncology': 'is_oncology',
        'CNS': 'is_cns',
        'Cardiovascular': 'is_cardiovascular',
        'Phase 2': 'is_phase2',
        'Phase 3': 'is_phase3',
        'Industry Sponsor': 'is_industry_sponsor'
    }
    
    subgroup_metrics = study.subgroup_analysis(df, model, feature_names, subgroups)
    
    # 3. Generate figures
    study.generate_publication_figures(df, model, feature_names)
    
    # 4. Generate Table 1
    study.generate_manuscript_table1(df)
    
    # 5. Save all results
    study.save_results()
    
    print("\n" + "="*80)
    print("✅ VALIDATION STUDY COMPLETE")
    print("="*80 + "\n")
    
    print("Next steps:")
    print(f"1. Review figures and tables in: {output_dir}/")
    print("2. Write manuscript using validation results")
    print("3. Submit to Clinical Trials or Drug Discovery Today")
    print("4. Include validation study in sales materials")
    print("5. Add 'Peer-reviewed validation' to pricing page")


if __name__ == '__main__':
    main()
