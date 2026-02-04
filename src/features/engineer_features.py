"""
Feature Engineering for Clinical Trial Risk Prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, List


class TrialFeatureEngineer:
    """Engineer features for clinical trial outcome prediction"""
    
    def __init__(self):
        # Define therapeutic area classifications
        self.oncology_keywords = [
            'cancer', 'carcinoma', 'tumor', 'tumour', 'lymphoma', 'leukemia',
            'melanoma', 'sarcoma', 'glioma', 'myeloma', 'metastatic', 'oncology'
        ]
        
        self.autoimmune_keywords = [
            'autoimmune', 'rheumatoid', 'lupus', 'crohn', 'colitis',
            'psoriasis', 'multiple sclerosis', 'arthritis', 'inflammatory'
        ]
        
        self.cns_keywords = [
            'alzheimer', 'parkinson', 'depression', 'schizophrenia',
            'anxiety', 'epilepsy', 'migraine', 'neurological', 'psychiatric'
        ]
        
        self.cardio_keywords = [
            'heart', 'cardiac', 'cardiovascular', 'hypertension',
            'arrhythmia', 'atherosclerosis', 'myocardial'
        ]
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for modeling
        
        Args:
            df: Raw clinical trials data
            
        Returns:
            DataFrame with engineered features
        """
        
        df = df.copy()
        
        print("Engineering features...")
        
        # 1. Target variable (trial success/failure)
        df['trial_success'] = self._create_target(df)
        
        # 2. Phase features
        df = self._engineer_phase_features(df)
        
        # 3. Therapeutic area features
        df = self._engineer_therapeutic_area(df)
        
        # 4. Enrollment features
        df = self._engineer_enrollment_features(df)
        
        # 5. Sponsor features
        df = self._engineer_sponsor_features(df)
        
        # 6. Geography features
        df = self._engineer_geography_features(df)
        
        # 7. Design features
        df = self._engineer_design_features(df)
        
        # 8. Intervention features
        df = self._engineer_intervention_features(df)
        
        # 9. Temporal features
        df = self._engineer_temporal_features(df)
        
        # 10. Complexity features
        df = self._engineer_complexity_features(df)
        
        print(f"Feature engineering complete. Shape: {df.shape}")
        
        return df
    
    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create binary target: 1 = success (completed), 0 = failure"""
        
        success_status = ['COMPLETED']
        failure_status = ['TERMINATED', 'WITHDRAWN', 'SUSPENDED']
        
        target = df['overall_status'].apply(
            lambda x: 1 if x in success_status 
            else 0 if x in failure_status 
            else np.nan
        )
        
        return target
    
    def _engineer_phase_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and encode phase information"""
        
        # Binary phase indicators
        df['is_phase1'] = df['phase'].str.contains('PHASE1', case=False, na=False).astype(int)
        df['is_phase2'] = df['phase'].str.contains('PHASE2', case=False, na=False).astype(int)
        df['is_phase3'] = df['phase'].str.contains('PHASE3', case=False, na=False).astype(int)
        df['is_phase4'] = df['phase'].str.contains('PHASE4', case=False, na=False).astype(int)
        
        # Combined phases (e.g., Phase 1/2)
        df['is_combined_phase'] = (df['phase'].str.count('\|') > 0).astype(int)
        
        # Phase number (ordinal: 1, 2, 3, 4, or mixed as 1.5, 2.5)
        def get_phase_numeric(phase_str):
            if pd.isna(phase_str) or phase_str == '':
                return np.nan
            if 'PHASE1' in phase_str and 'PHASE2' in phase_str:
                return 1.5
            elif 'PHASE2' in phase_str and 'PHASE3' in phase_str:
                return 2.5
            elif 'PHASE3' in phase_str:
                return 3.0
            elif 'PHASE2' in phase_str:
                return 2.0
            elif 'PHASE1' in phase_str:
                return 1.0
            elif 'PHASE4' in phase_str:
                return 4.0
            return np.nan
        
        df['phase_numeric'] = df['phase'].apply(get_phase_numeric)
        
        return df
    
    def _engineer_therapeutic_area(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify therapeutic areas"""
        
        condition_lower = df['condition'].fillna('').str.lower()
        
        df['is_oncology'] = condition_lower.apply(
            lambda x: any(kw in x for kw in self.oncology_keywords)
        ).astype(int)
        
        df['is_autoimmune'] = condition_lower.apply(
            lambda x: any(kw in x for kw in self.autoimmune_keywords)
        ).astype(int)
        
        df['is_cns'] = condition_lower.apply(
            lambda x: any(kw in x for kw in self.cns_keywords)
        ).astype(int)
        
        df['is_cardiovascular'] = condition_lower.apply(
            lambda x: any(kw in x for kw in self.cardio_keywords)
        ).astype(int)
        
        # Count conditions (some trials have multiple)
        df['condition_count'] = df['condition'].fillna('').str.count('\|') + 1
        
        return df
    
    def _engineer_enrollment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enrollment-related features"""
        
        df['enrollment'] = pd.to_numeric(df['enrollment'], errors='coerce')
        
        # Log-transformed enrollment (for modeling)
        df['log_enrollment'] = np.log1p(df['enrollment'].fillna(0))
        
        # Enrollment categories
        df['enrollment_size'] = pd.cut(
            df['enrollment'],
            bins=[0, 50, 200, 500, 10000],
            labels=['small', 'medium', 'large', 'xlarge']
        )
        
        # Small trial indicator (high risk)
        df['is_small_trial'] = (df['enrollment'] < 100).astype(int)
        
        # Enrollment type (actual vs anticipated)
        df['is_actual_enrollment'] = (
            df['enrollment_type'] == 'ACTUAL'
        ).astype(int)
        
        return df
    
    def _engineer_sponsor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sponsor-related features"""
        
        # Sponsor class binary indicators
        df['is_industry_sponsor'] = (
            df['lead_sponsor_class'] == 'INDUSTRY'
        ).astype(int)
        
        df['is_academic_sponsor'] = (
            df['lead_sponsor_class'].isin(['OTHER', 'OTHER_GOV', 'NIH'])
        ).astype(int)
        
        # Big pharma indicator (simplified - could be enhanced with company list)
        big_pharma = [
            'pfizer', 'novartis', 'roche', 'merck', 'gsk', 'sanofi',
            'abbvie', 'bristol', 'lilly', 'astrazeneca', 'amgen',
            'gilead', 'biogen', 'bms', 'takeda'
        ]
        
        sponsor_lower = df['lead_sponsor_name'].fillna('').str.lower()
        df['is_big_pharma'] = sponsor_lower.apply(
            lambda x: any(bp in x for bp in big_pharma)
        ).astype(int)
        
        # Collaborator count
        df['has_collaborators'] = (df['collaborator_count'] > 0).astype(int)
        
        return df
    
    def _engineer_geography_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create geography-related features"""
        
        df['location_count'] = pd.to_numeric(df['location_count'], errors='coerce').fillna(0)
        
        # Multi-site trial
        df['is_multisite'] = (df['location_count'] > 1).astype(int)
        
        # International trial
        df['is_international'] = (
            df['countries'].fillna('').str.count('\|') > 0
        ).astype(int)
        
        # US-based trial
        df['is_us_trial'] = (
            df['countries'].fillna('').str.contains('United States', case=False)
        ).astype(int)
        
        # Europe-based
        european_countries = ['France', 'Germany', 'Spain', 'Italy', 'United Kingdom']
        df['is_europe_trial'] = df['countries'].fillna('').apply(
            lambda x: any(country in x for country in european_countries)
        ).astype(int)
        
        return df
    
    def _engineer_design_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create study design features"""
        
        # Randomization
        df['is_randomized'] = (
            df['allocation'] == 'RANDOMIZED'
        ).astype(int)
        
        # Blinding/Masking
        df['is_blinded'] = (
            df['masking'].isin(['DOUBLE', 'TRIPLE', 'QUADRUPLE'])
        ).astype(int)
        
        # Primary purpose
        df['is_treatment_purpose'] = (
            df['primary_purpose'] == 'TREATMENT'
        ).astype(int)
        
        # Intervention model
        df['is_parallel'] = (
            df['intervention_model'] == 'PARALLEL'
        ).astype(int)
        
        # Outcome measures
        df['total_outcome_count'] = (
            df['primary_outcome_count'] + df['secondary_outcome_count']
        )
        
        df['has_secondary_outcomes'] = (
            df['secondary_outcome_count'] > 0
        ).astype(int)
        
        return df
    
    def _engineer_intervention_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create intervention-related features"""
        
        # Intervention type
        df['is_drug'] = df['intervention_type'].fillna('').str.contains(
            'DRUG', case=False
        ).astype(int)
        
        df['is_biological'] = df['intervention_type'].fillna('').str.contains(
            'BIOLOGICAL', case=False
        ).astype(int)
        
        df['is_device'] = df['intervention_type'].fillna('').str.contains(
            'DEVICE', case=False
        ).astype(int)
        
        # Multiple interventions
        df['intervention_count'] = (
            df['intervention_name'].fillna('').str.count('\|') + 1
        )
        
        df['has_multiple_interventions'] = (
            df['intervention_count'] > 1
        ).astype(int)
        
        return df
    
    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        
        # Convert dates
        df['start_date_dt'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['completion_date_dt'] = pd.to_datetime(df['completion_date'], errors='coerce')
        
        # Study duration (in days)
        df['study_duration_days'] = (
            df['completion_date_dt'] - df['start_date_dt']
        ).dt.days
        
        # Start year
        df['start_year'] = df['start_date_dt'].dt.year
        
        # Recent trial (started after 2015)
        df['is_recent_trial'] = (df['start_year'] >= 2015).astype(int)
        
        return df
    
    def _engineer_complexity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trial complexity indicators"""
        
        # Complexity score (composite of multiple factors)
        complexity_score = (
            df['is_combined_phase'] * 1 +
            df['has_multiple_interventions'] * 1 +
            df['is_international'] * 1 +
            (df['condition_count'] > 1).astype(int) * 1 +
            (df['total_outcome_count'] > 5).astype(int) * 1
        )
        
        df['complexity_score'] = complexity_score
        df['is_complex_trial'] = (complexity_score >= 3).astype(int)
        
        return df


def main():
    """Test feature engineering"""
    from pathlib import Path
    
    # Load most recent raw data
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
    raw_files = list(data_dir.glob('clinical_trials_raw_*.csv'))
    
    if not raw_files:
        print("No raw data found. Run collect_trials.py first.")
        return
    
    latest_file = max(raw_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading data from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    print(f"Loaded {len(df)} trials")
    
    # Engineer features
    engineer = TrialFeatureEngineer()
    df_features = engineer.create_features(df)
    
    # Save processed data
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'clinical_trials_features_{timestamp}.csv'
    
    df_features.to_csv(output_file, index=False)
    print(f"\nFeatures saved to: {output_file}")
    
    # Summary
    print(f"\n=== Feature Engineering Summary ===")
    print(f"Total features: {df_features.shape[1]}")
    print(f"\nTarget distribution:")
    print(df_features['trial_success'].value_counts())
    print(f"\nMissing target: {df_features['trial_success'].isna().sum()}")
    
    return df_features


if __name__ == '__main__':
    df = main()
