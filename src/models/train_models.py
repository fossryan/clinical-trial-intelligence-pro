"""
Clinical Trial Risk Prediction Models
Train and evaluate models with SHAP explanations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import json

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, accuracy_score, f1_score
)

import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

import shap
import matplotlib.pyplot as plt
import seaborn as sns


class TrialRiskPredictor:
    """Train models to predict clinical trial success/failure"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.metrics = {}
        
    def prepare_data(self, df: pd.DataFrame, balance_classes: bool = True):
        """
        Prepare data for modeling
        
        Args:
            df: DataFrame with engineered features
            balance_classes: Whether to use SMOTE for class imbalance
        """
        
        # Define feature columns (excluding metadata and target)
        exclude_cols = [
            'nct_id', 'brief_title', 'official_title', 'trial_success',
            'overall_status', 'condition', 'intervention_name', 'countries',
            'lead_sponsor_name', 'phase', 'intervention_type',
            'start_date', 'completion_date', 'last_update',
            'start_date_dt', 'completion_date_dt', 'enrollment_size',
            'study_type', 'allocation', 'intervention_model', 'primary_purpose',
            'masking', 'min_age', 'max_age', 'sex', 'healthy_volunteers',
            'enrollment_type'
        ]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with missing target
        df_clean = df[df['trial_success'].notna()].copy()
        print(f"Samples with valid target: {len(df_clean)}")
        
        X = df_clean[feature_cols].copy()
        y = df_clean['trial_success'].astype(int)
        
        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Handle missing values in features
        X = X.fillna(X.median())
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Train-test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"\nClass distribution in training:")
        print(y_train.value_counts())
        
        # Balance classes with SMOTE if requested
        if balance_classes and y_train.value_counts().min() / len(y_train) < 0.3:
            print("\nApplying SMOTE for class balance...")
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"Balanced training set: {len(X_train)} samples")
            print(y_train.value_counts())
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train baseline logistic regression model"""
        
        print("\n" + "="*50)
        print("Training Logistic Regression (Baseline)")
        print("="*50)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        lr = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=self.random_state,
            penalty='l2',
            C=1.0
        )
        
        lr.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = lr.predict(X_test_scaled)
        y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate
        metrics = self._evaluate_model(y_test, y_pred, y_pred_proba, "Logistic Regression")
        
        # Store
        self.models['logistic_regression'] = lr
        self.scalers['logistic_regression'] = scaler
        self.metrics['logistic_regression'] = metrics
        
        return lr, metrics
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        
        print("\n" + "="*50)
        print("Training XGBoost")
        print("="*50)
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Train model
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = self._evaluate_model(y_test, y_pred, y_pred_proba, "XGBoost")
        
        # Store
        self.models['xgboost'] = xgb_model
        self.metrics['xgboost'] = metrics
        
        return xgb_model, metrics
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM model"""
        
        print("\n" + "="*50)
        print("Training LightGBM")
        print("="*50)
        
        # Calculate scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Train model
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            verbose=-1
        )
        
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # Predictions
        y_pred = lgb_model.predict(X_test)
        y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = self._evaluate_model(y_test, y_pred, y_pred_proba, "LightGBM")
        
        # Store
        self.models['lightgbm'] = lgb_model
        self.metrics['lightgbm'] = metrics
        
        return lgb_model, metrics
    
    def _evaluate_model(self, y_true, y_pred, y_pred_proba, model_name):
        """Evaluate model performance"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        print(f"\n{model_name} Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Failed', 'Success']))
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        return metrics
    
    def generate_shap_explanations(self, X_train, model_name='xgboost', max_samples=500):
        """
        Generate SHAP values for model explanations
        
        Args:
            X_train: Training features
            model_name: Which model to explain
            max_samples: Max samples for background (speed vs accuracy tradeoff)
        """
        
        print(f"\n{'='*50}")
        print(f"Generating SHAP Explanations for {model_name}")
        print(f"{'='*50}")
        
        model = self.models[model_name]
        
        # Sample background data for speed
        if len(X_train) > max_samples:
            background = X_train.sample(max_samples, random_state=self.random_state)
        else:
            background = X_train
        
        # Create explainer
        if model_name == 'xgboost':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
        elif model_name == 'lightgbm':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
        else:  # logistic regression
            scaler = self.scalers[model_name]
            X_train_scaled = scaler.transform(X_train)
            explainer = shap.LinearExplainer(model, X_train_scaled)
            shap_values = explainer.shap_values(X_train_scaled)
        
        print("SHAP values computed successfully")
        
        return explainer, shap_values
    
    def plot_feature_importance(self, model_name='xgboost', top_n=20):
        """Plot top feature importances"""
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot
            plt.figure(figsize=(10, 8))
            sns.barplot(
                data=feature_importance.head(top_n),
                x='importance',
                y='feature',
                palette='viridis'
            )
            plt.title(f'Top {top_n} Feature Importances - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            return feature_importance
        
        else:
            print(f"{model_name} does not have feature_importances_ attribute")
            return None
    
    def save_models(self, output_dir: Path):
        """Save trained models and metadata"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save each model
        for model_name, model in self.models.items():
            model_path = output_dir / f'{model_name}_{timestamp}.joblib'
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = output_dir / f'scaler_{scaler_name}_{timestamp}.joblib'
            joblib.dump(scaler, scaler_path)
        
        # Save feature names
        feature_path = output_dir / f'feature_names_{timestamp}.json'
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save metrics
        metrics_path = output_dir / f'metrics_{timestamp}.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"\nAll artifacts saved to {output_dir}")


def main():
    """Main training pipeline"""
    
    # Load processed data
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    feature_files = list(data_dir.glob('clinical_trials_features_*.csv'))
    
    if not feature_files:
        print("No processed data found. Run engineer_features.py first.")
        return
    
    latest_file = max(feature_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading data from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    print(f"Loaded {len(df)} trials")
    
    # Initialize predictor
    predictor = TrialRiskPredictor(random_state=42)
    
    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(
        df, balance_classes=True
    )
    
    # Train models
    lr_model, lr_metrics = predictor.train_logistic_regression(
        X_train, y_train, X_test, y_test
    )
    
    xgb_model, xgb_metrics = predictor.train_xgboost(
        X_train, y_train, X_test, y_test
    )
    
    lgb_model, lgb_metrics = predictor.train_lightgbm(
        X_train, y_train, X_test, y_test
    )
    
    # Generate SHAP explanations for best model
    print("\nGenerating SHAP explanations...")
    explainer, shap_values = predictor.generate_shap_explanations(
        X_train, model_name='xgboost', max_samples=300
    )
    
    # Save models
    model_dir = Path(__file__).parent.parent.parent / 'data' / 'models'
    predictor.save_models(model_dir)
    
    # Compare models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    for model_name, metrics in predictor.metrics.items():
        print(f"\n{model_name.upper()}:")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  F1:      {metrics['f1_score']:.4f}")
        print(f"  Accuracy:{metrics['accuracy']:.4f}")
    
    return predictor, explainer, shap_values


if __name__ == '__main__':
    predictor, explainer, shap_values = main()
