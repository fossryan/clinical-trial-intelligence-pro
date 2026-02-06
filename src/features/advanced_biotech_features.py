"""
Advanced Biotech Features for Clinical Trial Intelligence

This module adds enterprise-grade domain expertise features:
1. Regulatory Pathway Prediction (FDA/EMA approval probability)
2. Mechanism of Action (MOA) Analysis
3. Site & Investigator Quality Scoring
4. Clinical Endpoint Sophistication Analysis
5. Competitive Landscape Saturation
6. Protocol Amendment Risk Factors

These features separate you from competitors and create a defensible moat.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import re
from collections import Counter


class AdvancedBiotechFeatures:
    """
    Advanced domain-specific feature engineering for clinical trials
    
    These features require deep biotech knowledge and create competitive advantage:
    - Regulatory pathway complexity
    - Drug mechanism novelty
    - Site selection quality
    - Endpoint appropriateness
    """
    
    def __init__(self):
        # Regulatory designations that affect approval
        self.orphan_keywords = [
            'rare disease', 'orphan', 'ultra-rare', 'orphan drug',
            'rare disorder', 'orphan designation'
        ]
        
        self.breakthrough_keywords = [
            'unmet medical need', 'no approved treatment', 'life-threatening',
            'serious condition', 'breakthrough therapy', 'fast track',
            'priority review', 'accelerated approval'
        ]
        
        self.regenerative_keywords = [
            'regenerative medicine', 'cell therapy', 'gene therapy',
            'stem cell', 'car-t', 'tissue engineering', 'rmat'
        ]
        
        # Endpoint types (ordered by regulatory preference)
        self.survival_endpoints = [
            'overall survival', 'os ', ' os)', 'mortality',
            'death', 'survival rate', 'kaplan-meier'
        ]
        
        self.progression_endpoints = [
            'progression-free survival', 'pfs', 'disease-free survival',
            'dfs', 'relapse-free survival', 'event-free survival',
            'time to progression', 'ttp'
        ]
        
        self.surrogate_endpoints = [
            'response rate', 'orr', 'objective response',
            'tumor shrinkage', 'tumor size', 'biomarker',
            'laboratory', 'imaging', 'surrogate marker'
        ]
        
        self.patient_reported = [
            'quality of life', 'qol', 'patient-reported',
            'symptom', 'pain scale', 'functional status'
        ]
        
        # MOA classifications (mechanism of action)
        self.moa_categories = {
            'targeted_therapy': [
                'inhibitor', 'antagonist', 'antibody', 'mab',
                'kinase inhibitor', 'targeted therapy', 'monoclonal'
            ],
            'immunotherapy': [
                'immune checkpoint', 'pd-1', 'pd-l1', 'ctla-4',
                'car-t', 'immunotherapy', 'immune system'
            ],
            'small_molecule': [
                'small molecule', 'oral', 'tablet', 'capsule'
            ],
            'biologic': [
                'biologic', 'protein', 'peptide', 'antibody',
                'fusion protein', 'recombinant'
            ],
            'gene_therapy': [
                'gene therapy', 'gene editing', 'crispr',
                'viral vector', 'aav', 'lentiviral'
            ],
            'cell_therapy': [
                'cell therapy', 'stem cell', 'car-t', 'car t',
                'adoptive transfer', 'cellular'
            ]
        }
        
        # Academic Medical Centers (high quality sites)
        self.academic_medical_centers = [
            'mayo clinic', 'cleveland clinic', 'johns hopkins',
            'stanford', 'harvard', 'massachusetts general',
            'md anderson', 'memorial sloan kettering', 'msk',
            'dana-farber', 'ucsf', 'university of california',
            'university of texas', 'university of pennsylvania',
            'duke university', 'yale', 'columbia university'
        ]
    
    def engineer_all_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all advanced biotech features to dataset
        
        Args:
            df: DataFrame with basic features
            
        Returns:
            DataFrame with advanced features added
        """
        
        print("\nEngineering Advanced Biotech Features...")
        print("=" * 80)
        
        df = df.copy()
        
        # 1. Regulatory pathway features
        df = self._engineer_regulatory_features(df)
        print("  ✓ Regulatory pathway features")
        
        # 2. Mechanism of action analysis
        df = self._engineer_moa_features(df)
        print("  ✓ Mechanism of action features")
        
        # 3. Clinical endpoint sophistication
        df = self._engineer_endpoint_features(df)
        print("  ✓ Clinical endpoint features")
        
        # 4. Site network quality
        df = self._engineer_site_intelligence(df)
        print("  ✓ Site network intelligence")
        
        # 5. Competitive landscape
        df = self._engineer_competitive_intelligence(df)
        print("  ✓ Competitive landscape features")
        
        # 6. Protocol design risk factors
        df = self._engineer_protocol_risk_features(df)
        print("  ✓ Protocol risk factors")
        
        # 7. Sponsor track record (placeholder for proprietary data)
        df = self._engineer_sponsor_track_record(df)
        print("  ✓ Sponsor track record features")
        
        print("=" * 80)
        print(f"✓ Added {sum(1 for col in df.columns if col not in df.columns[:50])} advanced features\n")
        
        return df
    
    def _engineer_regulatory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Regulatory pathway complexity and special designations
        
        Features:
        - Orphan drug designation (faster approval, smaller trials)
        - Breakthrough therapy signals (expedited review)
        - Regenerative medicine (RMAT - different standards)
        - Adaptive trial design (regulatory flexibility)
        """
        
        # Combine text fields for searching
        df['combined_text'] = (
            df['brief_title'].fillna('') + ' ' + 
            df['official_title'].fillna('') + ' ' +
            df['condition'].fillna('')
        ).str.lower()
        
        # Orphan drug designation indicators
        df['has_orphan_signals'] = df['combined_text'].apply(
            lambda x: any(kw in x for kw in self.orphan_keywords)
        ).astype(int)
        
        # Breakthrough therapy signals
        df['breakthrough_signals'] = df['combined_text'].apply(
            lambda x: sum(1 for kw in self.breakthrough_keywords if kw in x)
        )
        df['has_breakthrough_signals'] = (df['breakthrough_signals'] > 0).astype(int)
        
        # Regenerative medicine advanced therapy (RMAT)
        df['is_regenerative_medicine'] = df['combined_text'].apply(
            lambda x: any(kw in x for kw in self.regenerative_keywords)
        ).astype(int)
        
        # Adaptive trial design (signals regulatory flexibility)
        adaptive_keywords = ['adaptive', 'seamless', 'platform trial', 'master protocol']
        df['is_adaptive_design'] = df['combined_text'].apply(
            lambda x: any(kw in x for kw in adaptive_keywords)
        ).astype(int)
        
        # First-in-human / First-in-class (higher risk)
        df['is_first_in_human'] = df['combined_text'].str.contains(
            'first in human|first-in-human|fih', case=False, na=False
        ).astype(int)
        
        # Pediatric trial (different regulatory pathway)
        df['is_pediatric'] = (
            (df['min_age'].fillna('').str.contains('month|year', case=False)) |
            (df['combined_text'].str.contains('pediatric|children|child', case=False))
        ).astype(int)
        
        # Composite regulatory complexity score
        df['regulatory_complexity_score'] = (
            df['has_orphan_signals'] * 0.5 +  # Orphan = simpler
            df['is_regenerative_medicine'] * 2.0 +  # RMAT = complex
            df['is_adaptive_design'] * 1.0 +
            df['is_first_in_human'] * 2.0 +
            df['is_pediatric'] * 1.5
        )
        
        return df
    
    def _engineer_moa_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mechanism of Action (MOA) classification
        
        Different MOAs have different success rates:
        - Immunotherapy: High interest, competitive
        - Gene therapy: High risk, high reward
        - Small molecules: Well-understood, lower risk
        """
        
        intervention_text = df['intervention_name'].fillna('').str.lower()
        
        # Classify by MOA category
        for moa_type, keywords in self.moa_categories.items():
            df[f'moa_{moa_type}'] = intervention_text.apply(
                lambda x: any(kw in x for kw in keywords)
            ).astype(int)
        
        # Count MOA signals (multi-modal therapies)
        df['moa_complexity'] = sum(
            df[f'moa_{moa_type}'] for moa_type in self.moa_categories.keys()
        )
        
        # Combination therapy (multiple interventions = more complex)
        df['is_combination_therapy'] = (
            df['intervention_name'].str.count('\|') > 0
        ).astype(int)
        
        # Novel MOA (no keywords match = potentially first-in-class)
        df['is_novel_moa'] = (df['moa_complexity'] == 0).astype(int)
        
        return df
    
    def _engineer_endpoint_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clinical endpoint sophistication and regulatory acceptability
        
        Endpoint hierarchy (regulatory preference):
        1. Overall Survival (gold standard)
        2. Progression-Free Survival (acceptable)
        3. Surrogate markers (risky, need validation)
        4. Patient-reported outcomes (supplementary)
        """
        
        text_fields = df['brief_title'].fillna('') + ' ' + df['official_title'].fillna('')
        text_lower = text_fields.str.lower()
        
        # Classify primary endpoint type
        df['has_survival_endpoint'] = text_lower.apply(
            lambda x: any(kw in x for kw in self.survival_endpoints)
        ).astype(int)
        
        df['has_progression_endpoint'] = text_lower.apply(
            lambda x: any(kw in x for kw in self.progression_endpoints)
        ).astype(int)
        
        df['has_surrogate_endpoint'] = text_lower.apply(
            lambda x: any(kw in x for kw in self.surrogate_endpoints)
        ).astype(int)
        
        df['has_patient_reported'] = text_lower.apply(
            lambda x: any(kw in x for kw in self.patient_reported)
        ).astype(int)
        
        # Endpoint quality score (higher = better for regulatory approval)
        df['endpoint_quality_score'] = (
            df['has_survival_endpoint'] * 3.0 +  # Best
            df['has_progression_endpoint'] * 2.0 +  # Good
            df['has_surrogate_endpoint'] * 1.0 +  # Risky
            df['has_patient_reported'] * 0.5  # Supplementary
        )
        
        # Multiple endpoints (harder to show significance on all)
        df['primary_outcome_complexity'] = df['primary_outcome_count'].fillna(1)
        df['has_multiple_primary_endpoints'] = (
            df['primary_outcome_complexity'] > 2
        ).astype(int)
        
        # Composite endpoints (difficult to interpret)
        df['has_composite_endpoint'] = text_lower.str.contains(
            'composite|combined endpoint', case=False, na=False
        ).astype(int)
        
        return df
    
    def _engineer_site_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Site network quality and investigator experience
        
        High-quality sites = better enrollment, data quality, retention
        """
        
        # Geographic diversity (reduces enrollment risk)
        df['country_count'] = df['countries'].fillna('').str.count('\|') + 1
        df['is_global_trial'] = (df['country_count'] >= 5).astype(int)
        df['is_us_only'] = (
            (df['countries'].fillna('').str.upper() == 'UNITED STATES') |
            (df['countries'].fillna('').str.upper() == 'US')
        ).astype(int)
        
        # Site count analysis
        df['location_count'] = df['location_count'].fillna(0)
        df['enrollment'] = pd.to_numeric(df['enrollment'], errors='coerce').fillna(0)
        
        # Sites per 100 patients (efficiency metric)
        df['sites_per_100_patients'] = np.where(
            df['enrollment'] > 0,
            (df['location_count'] / df['enrollment']) * 100,
            0
        )
        
        # Overseeded trial (too many sites = enrollment challenges)
        df['is_overseeded'] = (df['sites_per_100_patients'] > 5).astype(int)
        
        # Single-site trial (higher quality but enrollment risk)
        df['is_single_site'] = (df['location_count'] == 1).astype(int)
        
        # Multi-site coordination complexity
        df['site_coordination_complexity'] = np.log1p(df['location_count'])
        
        # Academic Medical Center participation (proxy)
        # In production: would join with site quality database
        df['has_amc_signals'] = (
            df['lead_sponsor_name'].fillna('').str.lower().apply(
                lambda x: any(amc in x for amc in self.academic_medical_centers)
            ) |
            df['brief_title'].fillna('').str.lower().apply(
                lambda x: any(amc in x for amc in self.academic_medical_centers)
            )
        ).astype(int)
        
        return df
    
    def _engineer_competitive_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Market saturation and competitive landscape
        
        High competition = harder to recruit, harder to differentiate
        """
        
        # Count concurrent trials per condition
        condition_counts = df.groupby('condition').size()
        df['condition_saturation'] = df['condition'].map(condition_counts).fillna(0)
        
        # High competition indicator (>20 concurrent trials)
        df['high_competition_space'] = (df['condition_saturation'] > 20).astype(int)
        
        # Niche indication (low competition, unmet need)
        df['niche_indication'] = (df['condition_saturation'] < 5).astype(int)
        
        # Sponsor trial count (portfolio size)
        sponsor_counts = df.groupby('lead_sponsor_name').size()
        df['sponsor_portfolio_size'] = df['lead_sponsor_name'].map(sponsor_counts).fillna(1)
        
        # Mega-sponsor (>100 trials = big pharma)
        df['is_mega_sponsor'] = (df['sponsor_portfolio_size'] > 100).astype(int)
        
        return df
    
    def _engineer_protocol_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Protocol design risk factors
        
        Complex protocols = higher dropout, lower compliance
        """
        
        # Total outcome measures (complexity)
        df['total_outcome_measures'] = (
            df['primary_outcome_count'].fillna(0) + 
            df['secondary_outcome_count'].fillna(0)
        )
        df['high_outcome_burden'] = (df['total_outcome_measures'] > 10).astype(int)
        
        # Study duration estimation (longer = higher dropout)
        # Placeholder - would calculate from start/completion dates
        df['estimated_duration_months'] = 24  # Default assumption
        df['long_duration_trial'] = (df['estimated_duration_months'] > 36).astype(int)
        
        # Blinding quality
        masking_map = {
            'QUADRUPLE': 4,
            'TRIPLE': 3,
            'DOUBLE': 2,
            'SINGLE': 1,
            'NONE': 0
        }
        df['blinding_score'] = df['masking'].map(masking_map).fillna(0)
        df['is_unblinded'] = (df['blinding_score'] == 0).astype(int)
        
        # Randomization quality
        df['is_randomized'] = (
            df['allocation'].fillna('').str.upper() == 'RANDOMIZED'
        ).astype(int)
        df['is_non_randomized'] = (
            df['allocation'].fillna('').str.upper() == 'NON_RANDOMIZED'
        ).astype(int)
        
        # Control arm type
        df['has_placebo_control'] = (
            df['intervention_name'].fillna('').str.lower().str.contains('placebo')
        ).astype(int)
        df['has_active_comparator'] = (
            df['intervention_model'].fillna('').str.upper() == 'PARALLEL'
        ).astype(int)
        
        # Study design quality score
        df['design_quality_score'] = (
            df['is_randomized'] * 2.0 +
            df['blinding_score'] * 0.5 +
            df['has_placebo_control'] * 1.0 +
            (1 - df['is_single_site']) * 0.5
        )
        
        return df
    
    def _engineer_sponsor_track_record(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sponsor historical success rate
        
        NOTE: This is a placeholder. In production, would join with:
        - Historical trial outcomes by sponsor
        - FDA approval success rates
        - Average time to approval
        - Protocol amendment frequency
        
        This data would come from:
        1. CRO partnership (proprietary data = your moat)
        2. Scraped FDA approvals database
        3. User-generated feedback loop
        """
        
        # Placeholder features
        # In production: calculate from historical data
        df['sponsor_historical_success_rate'] = np.nan
        df['sponsor_avg_enrollment_velocity'] = np.nan
        df['sponsor_protocol_amendment_rate'] = np.nan
        df['sponsor_fda_interactions'] = np.nan
        
        # Binary: does sponsor have track record in this indication?
        # (Would join with sponsor-indication history table)
        df['sponsor_has_indication_experience'] = 0
        
        # Sponsor quality tier (would come from external rating)
        df['sponsor_quality_tier'] = 'unknown'
        
        return df
    
    def calculate_site_investigator_score(
        self, 
        investigator_name: str,
        site_name: str,
        proprietary_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Calculate site/investigator quality score
        
        This would be a premium API endpoint powered by CRO data.
        
        Scoring factors:
        1. Number of trials completed (experience)
        2. Historical enrollment velocity (vs planned)
        3. Protocol violation rate
        4. Patient retention rate
        5. Data quality score (query rate)
        6. Publications in relevant area
        7. FDA inspection history
        8. Distance to patient population
        
        Args:
            investigator_name: PI name
            site_name: Site name and location
            proprietary_data: CRO partnership data (if available)
            
        Returns:
            Dict with quality scores and recommendations
        """
        
        # Placeholder implementation
        # In production: query proprietary database
        
        if proprietary_data is not None:
            # Calculate from real data
            site_data = proprietary_data[
                (proprietary_data['investigator'] == investigator_name) &
                (proprietary_data['site'] == site_name)
            ]
            
            if len(site_data) > 0:
                score = {
                    'overall_score': site_data['quality_score'].mean(),
                    'experience_score': len(site_data),
                    'enrollment_velocity': site_data['enrollment_rate'].mean(),
                    'retention_rate': site_data['retention_rate'].mean(),
                    'data_quality': site_data['data_quality'].mean(),
                    'recommendation': 'Based on historical performance'
                }
            else:
                score = self._default_site_score()
        else:
            score = self._default_site_score()
        
        return score
    
    def _default_site_score(self) -> Dict:
        """Default score when no data available"""
        return {
            'overall_score': None,
            'experience_score': None,
            'enrollment_velocity': None,
            'retention_rate': None,
            'data_quality': None,
            'recommendation': 'Insufficient data - consider sites with proven track record'
        }
    
    def predict_regulatory_approval_probability(
        self,
        trial_features: Dict,
        phase_2_success: bool = None,
        phase_3_success: bool = None
    ) -> Dict:
        """
        Predict FDA/EMA approval probability
        
        This is the "killer feature" that competitors don't have.
        
        Inputs:
        - Trial design features
        - Phase 2 results (if available)
        - Phase 3 results (if available)
        - Endpoint type
        - Unmet medical need
        - Safety profile
        - Comparator performance
        
        Returns:
            Approval probability and key risk factors
        """
        
        # Simplified scoring model (in production: train on FDA approval data)
        
        approval_score = 50  # Base 50%
        
        # Phase 2/3 success
        if phase_2_success is True:
            approval_score += 15
        if phase_3_success is True:
            approval_score += 25
        
        # Endpoint quality
        endpoint_quality = trial_features.get('endpoint_quality_score', 1)
        approval_score += endpoint_quality * 3
        
        # Unmet medical need
        if trial_features.get('has_breakthrough_signals', 0) == 1:
            approval_score += 10
        
        # Orphan drug (easier approval)
        if trial_features.get('has_orphan_signals', 0) == 1:
            approval_score += 8
        
        # Design quality
        design_quality = trial_features.get('design_quality_score', 2)
        approval_score += design_quality * 2
        
        # Cap at 95%
        approval_score = min(approval_score, 95)
        
        risk_factors = []
        if endpoint_quality < 2:
            risk_factors.append("Surrogate endpoint may require additional validation")
        if trial_features.get('is_adaptive_design', 0) == 1:
            risk_factors.append("Adaptive design requires pre-specified statistical plan")
        if trial_features.get('is_single_site', 0) == 1:
            risk_factors.append("Single-site trial may limit generalizability")
        
        return {
            'approval_probability': approval_score / 100,
            'risk_factors': risk_factors,
            'recommendation': self._get_approval_recommendation(approval_score)
        }
    
    def _get_approval_recommendation(self, score: float) -> str:
        """Get strategic recommendation based on approval probability"""
        
        if score >= 75:
            return "High confidence - proceed with regulatory filing preparation"
        elif score >= 60:
            return "Moderate confidence - consider additional supporting studies"
        elif score >= 45:
            return "Uncertain - recommend FDA Type C meeting for guidance"
        else:
            return "Low confidence - reassess indication or endpoint strategy"


def main():
    """
    Demonstrate advanced feature engineering
    """
    
    print("\n" + "="*80)
    print("ADVANCED BIOTECH FEATURES - DEMO")
    print("="*80 + "\n")
    
    # Load sample data
    from pathlib import Path
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    feature_files = list(data_dir.glob('clinical_trials_features_*.csv'))
    
    if not feature_files:
        print("❌ No processed data found. Run feature engineering first.")
        return
    
    latest_file = max(feature_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest_file.name}\n")
    
    df = pd.read_csv(latest_file)
    print(f"Loaded {len(df):,} trials\n")
    
    # Apply advanced features
    engine = AdvancedBiotechFeatures()
    df_enhanced = engine.engineer_all_advanced_features(df)
    
    # Summary statistics
    print("\n" + "="*80)
    print("ADVANCED FEATURES SUMMARY")
    print("="*80 + "\n")
    
    print("Regulatory Features:")
    print(f"  Orphan drug signals: {df_enhanced['has_orphan_signals'].sum():,} trials")
    print(f"  Breakthrough signals: {df_enhanced['has_breakthrough_signals'].sum():,} trials")
    print(f"  Regenerative medicine: {df_enhanced['is_regenerative_medicine'].sum():,} trials")
    print(f"  Adaptive designs: {df_enhanced['is_adaptive_design'].sum():,} trials")
    
    print("\nMechanism of Action:")
    for moa_type in engine.moa_categories.keys():
        count = df_enhanced[f'moa_{moa_type}'].sum()
        print(f"  {moa_type.replace('_', ' ').title()}: {count:,} trials")
    
    print("\nEndpoint Quality:")
    print(f"  Survival endpoints: {df_enhanced['has_survival_endpoint'].sum():,} trials")
    print(f"  Progression endpoints: {df_enhanced['has_progression_endpoint'].sum():,} trials")
    print(f"  Surrogate endpoints: {df_enhanced['has_surrogate_endpoint'].sum():,} trials")
    
    print("\nSite Network:")
    print(f"  Global trials (5+ countries): {df_enhanced['is_global_trial'].sum():,} trials")
    print(f"  US-only trials: {df_enhanced['is_us_only'].sum():,} trials")
    print(f"  Single-site trials: {df_enhanced['is_single_site'].sum():,} trials")
    print(f"  Academic center participation: {df_enhanced['has_amc_signals'].sum():,} trials")
    
    # Save enhanced dataset
    output_file = latest_file.parent / f"advanced_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_enhanced.to_csv(output_file, index=False)
    print(f"\n✓ Saved enhanced dataset to: {output_file.name}")
    
    print("\n" + "="*80)
    print("✅ ADVANCED FEATURES READY")
    print("="*80 + "\n")
    
    print("Next steps:")
    print("1. Retrain models with advanced features (expect +3-5% accuracy)")
    print("2. Use indication-specific models for best results")
    print("3. Partner with CRO to get proprietary site/investigator data")


if __name__ == '__main__':
    main()
