"""
Site Intelligence Module
Enterprise-grade site selection, performance tracking, and geographic analysis

Features:
- Site performance scoring
- Historical success rate tracking
- Geographic optimization
- Competitive trial density analysis
- Diversity & compliance metrics
- Predictive site selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from collections import defaultdict


@dataclass
class SiteProfile:
    """Comprehensive site profile"""
    site_id: str
    facility_name: str
    city: str
    state: str
    country: str
    latitude: float
    longitude: float
    
    # Performance metrics
    total_trials: int
    active_trials: int
    completed_trials: int
    terminated_trials: int
    success_rate: float
    
    # Operational metrics
    avg_enrollment_rate: float  # patients per month
    avg_time_to_first_patient: int  # days
    avg_trial_duration: int  # days
    data_quality_score: float  # 0-100
    
    # Therapeutic area experience
    therapeutic_areas: Dict[str, int]
    phase_experience: Dict[str, int]
    
    # Diversity metrics
    diversity_score: float  # 0-100
    demographic_breakdown: Dict[str, float]
    
    # Risk indicators
    dropout_rate: float
    protocol_deviation_rate: float
    competitive_load: int  # number of competing trials nearby
    
    # Overall score
    performance_score: float  # 0-100


@dataclass
class SiteRecommendation:
    """Site recommendation with reasoning"""
    site: SiteProfile
    match_score: float  # 0-100
    strengths: List[str]
    concerns: List[str]
    predicted_enrollment_rate: float
    estimated_cost: float
    confidence_level: str  # 'high', 'medium', 'low'


class SiteIntelligenceEngine:
    """
    Advanced site intelligence and selection system
    
    Capabilities:
    - Site performance analysis
    - Predictive site selection
    - Geographic optimization
    - Competitive landscape analysis
    - Diversity planning
    """
    
    def __init__(self):
        self.site_database = {}
        self.performance_weights = self._default_weights()
    
    def _default_weights(self) -> Dict[str, float]:
        """Default weights for performance scoring"""
        return {
            'success_rate': 0.25,
            'enrollment_rate': 0.20,
            'data_quality': 0.15,
            'therapeutic_experience': 0.15,
            'diversity_score': 0.10,
            'time_to_first_patient': 0.10,
            'dropout_rate': 0.05
        }
    
    def build_site_database(self, trials_df: pd.DataFrame) -> Dict[str, SiteProfile]:
        """
        Build comprehensive site database from trial data
        
        Args:
            trials_df: Clinical trials data with site information
            
        Returns:
            Dictionary of site_id -> SiteProfile
        """
        
        print("Building site intelligence database...")
        
        # Extract site-level data from trials
        site_data = defaultdict(lambda: {
            'trials': [],
            'locations': [],
            'statuses': [],
            'phases': [],
            'therapeutic_areas': [],
            'enrollments': [],
            'durations': []
        })
        
        # Aggregate data by site
        for _, trial in trials_df.iterrows():
            sites = self._extract_sites(trial)
            
            for site_info in sites:
                site_key = self._generate_site_key(site_info)
                
                site_data[site_key]['trials'].append(trial['nct_id'])
                site_data[site_key]['locations'].append(site_info)
                site_data[site_key]['statuses'].append(trial.get('overall_status'))
                site_data[site_key]['phases'].append(trial.get('phase', 'UNKNOWN'))
                site_data[site_key]['enrollments'].append(trial.get('enrollment_actual', 0))
                
                # Extract therapeutic areas
                if 'is_oncology' in trial and trial['is_oncology']:
                    site_data[site_key]['therapeutic_areas'].append('Oncology')
                if 'is_autoimmune' in trial and trial['is_autoimmune']:
                    site_data[site_key]['therapeutic_areas'].append('Autoimmune')
                if 'is_cns' in trial and trial['is_cns']:
                    site_data[site_key]['therapeutic_areas'].append('CNS')
                if 'is_cardiovascular' in trial and trial['is_cardiovascular']:
                    site_data[site_key]['therapeutic_areas'].append('Cardiovascular')
        
        # Build site profiles
        site_profiles = {}
        
        for site_key, data in site_data.items():
            if len(data['trials']) >= 2:  # Require at least 2 trials for meaningful stats
                profile = self._create_site_profile(site_key, data)
                site_profiles[site_key] = profile
        
        self.site_database = site_profiles
        print(f"✓ Built database with {len(site_profiles)} sites")
        
        return site_profiles
    
    def _extract_sites(self, trial: pd.Series) -> List[Dict]:
        """Extract site information from trial data"""
        
        # For now, use simple location extraction
        # In production, would parse locations from API
        
        sites = []
        
        # Extract from available location fields
        if 'facility_name' in trial and pd.notna(trial['facility_name']):
            sites.append({
                'facility_name': trial['facility_name'],
                'city': trial.get('city', 'Unknown'),
                'state': trial.get('state', ''),
                'country': trial.get('country', 'Unknown'),
                'latitude': trial.get('latitude', 0.0),
                'longitude': trial.get('longitude', 0.0)
            })
        
        return sites
    
    def _generate_site_key(self, site_info: Dict) -> str:
        """Generate unique key for site"""
        return f"{site_info['facility_name']}_{site_info['city']}_{site_info['country']}".lower().replace(' ', '_')
    
    def _create_site_profile(self, site_key: str, data: Dict) -> SiteProfile:
        """Create comprehensive site profile from aggregated data"""
        
        # Get location info (use first occurrence)
        location = data['locations'][0]
        
        # Calculate metrics
        total_trials = len(data['trials'])
        statuses = pd.Series(data['statuses'])
        completed = (statuses == 'COMPLETED').sum()
        terminated = (statuses == 'TERMINATED').sum()
        active = (statuses.isin(['RECRUITING', 'ACTIVE_NOT_RECRUITING'])).sum()
        
        success_rate = (completed / (completed + terminated) * 100) if (completed + terminated) > 0 else 50.0
        
        # Enrollment metrics
        enrollments = [e for e in data['enrollments'] if e > 0]
        avg_enrollment = np.mean(enrollments) if enrollments else 0
        avg_enrollment_rate = avg_enrollment / 12 if avg_enrollment > 0 else 0  # per month estimate
        
        # Therapeutic area breakdown
        therapeutic_counts = pd.Series(data['therapeutic_areas']).value_counts().to_dict()
        
        # Phase experience
        phase_counts = pd.Series(data['phases']).value_counts().to_dict()
        
        # Performance score (simplified)
        performance_score = self._calculate_performance_score({
            'success_rate': success_rate,
            'total_trials': total_trials,
            'avg_enrollment': avg_enrollment_rate
        })
        
        return SiteProfile(
            site_id=site_key,
            facility_name=location['facility_name'],
            city=location['city'],
            state=location['state'],
            country=location['country'],
            latitude=location['latitude'],
            longitude=location['longitude'],
            
            total_trials=total_trials,
            active_trials=int(active),
            completed_trials=int(completed),
            terminated_trials=int(terminated),
            success_rate=round(success_rate, 1),
            
            avg_enrollment_rate=round(avg_enrollment_rate, 2),
            avg_time_to_first_patient=60,  # Default estimate
            avg_trial_duration=365,  # Default estimate
            data_quality_score=85.0,  # Default estimate
            
            therapeutic_areas=therapeutic_counts,
            phase_experience=phase_counts,
            
            diversity_score=75.0,  # Default estimate
            demographic_breakdown={},  # Would come from detailed data
            
            dropout_rate=0.10,  # Default estimate
            protocol_deviation_rate=0.05,  # Default estimate
            competitive_load=2,  # Default estimate
            
            performance_score=round(performance_score, 1)
        )
    
    def _calculate_performance_score(self, metrics: Dict) -> float:
        """Calculate overall performance score for a site"""
        
        # Normalize metrics
        success_score = metrics['success_rate']  # Already 0-100
        
        trial_volume_score = min(100, metrics['total_trials'] * 5)  # Max at 20 trials
        
        enrollment_score = min(100, metrics['avg_enrollment'] * 10)  # Max at 10 pts/month
        
        # Weighted average
        score = (
            success_score * 0.4 +
            trial_volume_score * 0.3 +
            enrollment_score * 0.3
        )
        
        return score
    
    def rank_sites(self, 
                   trial_requirements: Dict,
                   top_n: int = 20,
                   filters: Optional[Dict] = None) -> List[Tuple[SiteProfile, float]]:
        """
        Rank sites based on trial requirements
        
        Args:
            trial_requirements: Dict with trial parameters
            top_n: Number of top sites to return
            filters: Optional filters (geographic, therapeutic area, etc.)
            
        Returns:
            List of (SiteProfile, match_score) tuples, sorted by score
        """
        
        if not self.site_database:
            raise ValueError("Site database not built. Call build_site_database() first.")
        
        scored_sites = []
        
        for site in self.site_database.values():
            # Apply filters
            if filters and not self._passes_filters(site, filters):
                continue
            
            # Calculate match score
            match_score = self._calculate_match_score(site, trial_requirements)
            scored_sites.append((site, match_score))
        
        # Sort by score
        scored_sites.sort(key=lambda x: x[1], reverse=True)
        
        return scored_sites[:top_n]
    
    def _passes_filters(self, site: SiteProfile, filters: Dict) -> bool:
        """Check if site passes all filters"""
        
        if 'countries' in filters:
            if site.country not in filters['countries']:
                return False
        
        if 'min_trials' in filters:
            if site.total_trials < filters['min_trials']:
                return False
        
        if 'min_success_rate' in filters:
            if site.success_rate < filters['min_success_rate']:
                return False
        
        if 'therapeutic_area' in filters:
            if filters['therapeutic_area'] not in site.therapeutic_areas:
                return False
        
        if 'phase' in filters:
            if filters['phase'] not in site.phase_experience:
                return False
        
        return True
    
    def _calculate_match_score(self, site: SiteProfile, requirements: Dict) -> float:
        """
        Calculate how well a site matches trial requirements
        
        Scoring factors:
        - Therapeutic area experience
        - Phase experience  
        - Performance history
        - Geographic suitability
        - Current capacity
        """
        
        score_components = []
        
        # 1. Base performance score (30%)
        score_components.append(('base_performance', site.performance_score, 0.30))
        
        # 2. Therapeutic area match (25%)
        therapeutic_score = 0
        if 'therapeutic_area' in requirements:
            required_area = requirements['therapeutic_area']
            if required_area in site.therapeutic_areas:
                # Score based on experience (number of trials)
                therapeutic_score = min(100, site.therapeutic_areas[required_area] * 20)
        score_components.append(('therapeutic_match', therapeutic_score, 0.25))
        
        # 3. Phase experience (20%)
        phase_score = 0
        if 'phase' in requirements:
            required_phase = requirements['phase']
            if required_phase in site.phase_experience:
                phase_score = min(100, site.phase_experience[required_phase] * 20)
        score_components.append(('phase_experience', phase_score, 0.20))
        
        # 4. Enrollment capacity (15%)
        enrollment_score = min(100, site.avg_enrollment_rate * 10)
        score_components.append(('enrollment_capacity', enrollment_score, 0.15))
        
        # 5. Availability (current load) (10%)
        # Penalize if too many active trials
        availability_score = max(0, 100 - (site.active_trials * 10))
        score_components.append(('availability', availability_score, 0.10))
        
        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in score_components)
        
        return round(total_score, 2)
    
    def recommend_sites(self, 
                       trial_requirements: Dict,
                       num_sites: int = 10) -> List[SiteRecommendation]:
        """
        Generate detailed site recommendations with reasoning
        
        Args:
            trial_requirements: Trial parameters
            num_sites: Number of recommendations to generate
            
        Returns:
            List of SiteRecommendation objects
        """
        
        # Get ranked sites
        ranked_sites = self.rank_sites(trial_requirements, top_n=num_sites)
        
        recommendations = []
        
        for site, match_score in ranked_sites:
            # Generate strengths
            strengths = self._identify_strengths(site, trial_requirements)
            
            # Identify concerns
            concerns = self._identify_concerns(site, trial_requirements)
            
            # Predict enrollment rate for this trial
            predicted_rate = self._predict_site_enrollment(site, trial_requirements)
            
            # Estimate cost
            estimated_cost = self._estimate_site_cost(site, trial_requirements)
            
            # Confidence level
            confidence = self._assess_confidence(site, match_score)
            
            recommendations.append(SiteRecommendation(
                site=site,
                match_score=match_score,
                strengths=strengths,
                concerns=concerns,
                predicted_enrollment_rate=predicted_rate,
                estimated_cost=estimated_cost,
                confidence_level=confidence
            ))
        
        return recommendations
    
    def _identify_strengths(self, site: SiteProfile, requirements: Dict) -> List[str]:
        """Identify site strengths"""
        strengths = []
        
        if site.success_rate > 80:
            strengths.append(f"Excellent success rate: {site.success_rate}%")
        
        if site.total_trials >= 10:
            strengths.append(f"Extensive experience: {site.total_trials} trials")
        
        if site.avg_enrollment_rate > 5:
            strengths.append(f"Strong enrollment: {site.avg_enrollment_rate:.1f} patients/month")
        
        if 'therapeutic_area' in requirements:
            area = requirements['therapeutic_area']
            if area in site.therapeutic_areas and site.therapeutic_areas[area] >= 5:
                strengths.append(f"Deep {area} expertise: {site.therapeutic_areas[area]} trials")
        
        if site.data_quality_score > 90:
            strengths.append(f"High data quality: {site.data_quality_score:.0f}/100")
        
        return strengths
    
    def _identify_concerns(self, site: SiteProfile, requirements: Dict) -> List[str]:
        """Identify potential concerns"""
        concerns = []
        
        if site.success_rate < 60:
            concerns.append(f"Lower success rate: {site.success_rate}%")
        
        if site.active_trials > 5:
            concerns.append(f"High current load: {site.active_trials} active trials")
        
        if site.dropout_rate > 0.20:
            concerns.append(f"Elevated dropout rate: {site.dropout_rate*100:.0f}%")
        
        if site.total_trials < 5:
            concerns.append(f"Limited track record: {site.total_trials} trials")
        
        if 'therapeutic_area' in requirements:
            area = requirements['therapeutic_area']
            if area not in site.therapeutic_areas:
                concerns.append(f"No prior {area} experience")
        
        return concerns
    
    def _predict_site_enrollment(self, site: SiteProfile, requirements: Dict) -> float:
        """Predict enrollment rate for this site on this trial"""
        
        # Base rate
        base_rate = site.avg_enrollment_rate
        
        # Adjust based on therapeutic area match
        if 'therapeutic_area' in requirements:
            area = requirements['therapeutic_area']
            if area in site.therapeutic_areas:
                # Boost if experienced
                base_rate *= 1.2
            else:
                # Reduce if new area
                base_rate *= 0.8
        
        # Adjust based on current load
        if site.active_trials > 5:
            base_rate *= 0.85
        elif site.active_trials < 2:
            base_rate *= 1.1
        
        return round(base_rate, 2)
    
    def _estimate_site_cost(self, site: SiteProfile, requirements: Dict) -> float:
        """Estimate cost for this site"""
        
        # Simple cost model (would be much more sophisticated in production)
        base_cost_per_patient = 5000  # USD
        
        # Adjust by country
        country_multipliers = {
            'United States': 1.5,
            'United Kingdom': 1.3,
            'Germany': 1.2,
            'France': 1.2,
            'Canada': 1.3,
            'India': 0.6,
            'China': 0.7
        }
        
        multiplier = country_multipliers.get(site.country, 1.0)
        
        # Estimate based on trial size
        estimated_patients = requirements.get('target_enrollment', 50)
        
        total_cost = base_cost_per_patient * estimated_patients * multiplier
        
        return round(total_cost, 0)
    
    def _assess_confidence(self, site: SiteProfile, match_score: float) -> str:
        """Assess confidence level in recommendation"""
        
        if match_score >= 80 and site.total_trials >= 10:
            return 'high'
        elif match_score >= 65 and site.total_trials >= 5:
            return 'medium'
        else:
            return 'low'
    
    def analyze_geographic_distribution(self, trial_requirements: Dict) -> Dict:
        """
        Analyze geographic distribution of suitable sites
        
        Returns:
            Geographic analysis with optimal site distribution
        """
        
        suitable_sites = self.rank_sites(trial_requirements, top_n=100)
        
        # Group by country
        country_stats = defaultdict(lambda: {
            'sites': [],
            'avg_score': 0,
            'total_sites': 0,
            'avg_enrollment_rate': 0
        })
        
        for site, score in suitable_sites:
            country = site.country
            country_stats[country]['sites'].append((site, score))
            country_stats[country]['total_sites'] += 1
        
        # Calculate averages
        for country, stats in country_stats.items():
            scores = [s for _, s in stats['sites']]
            enrollment_rates = [site.avg_enrollment_rate for site, _ in stats['sites']]
            
            stats['avg_score'] = round(np.mean(scores), 2)
            stats['avg_enrollment_rate'] = round(np.mean(enrollment_rates), 2)
            stats['top_sites'] = sorted(stats['sites'], key=lambda x: x[1], reverse=True)[:5]
        
        return dict(country_stats)
    
    def calculate_competitive_density(self, location: Dict, therapeutic_area: str, radius_km: float = 50) -> Dict:
        """
        Calculate competitive trial density around a location
        
        Args:
            location: Dict with latitude/longitude
            therapeutic_area: Therapeutic area to analyze
            radius_km: Radius to search (km)
            
        Returns:
            Competitive density metrics
        """
        
        # In production, would use geospatial queries
        # For now, simplified analysis
        
        nearby_sites = []
        for site in self.site_database.values():
            # Simple distance calculation (would use proper geospatial in production)
            if abs(site.latitude - location['latitude']) < 1.0 and \
               abs(site.longitude - location['longitude']) < 1.0:
                nearby_sites.append(site)
        
        # Count competing trials
        competing_trials = sum(site.active_trials for site in nearby_sites)
        
        # Calculate competition score
        competition_score = min(100, competing_trials * 5)
        
        return {
            'nearby_sites': len(nearby_sites),
            'competing_trials': competing_trials,
            'competition_score': competition_score,
            'recommendation': 'low' if competing_trials < 5 else 'medium' if competing_trials < 10 else 'high'
        }
    
    def export_site_rankings(self, recommendations: List[SiteRecommendation], format: str = 'dataframe') -> pd.DataFrame:
        """Export site rankings for analysis"""
        
        data = []
        for rec in recommendations:
            data.append({
                'site_id': rec.site.site_id,
                'facility_name': rec.site.facility_name,
                'city': rec.site.city,
                'country': rec.site.country,
                'match_score': rec.match_score,
                'predicted_enrollment_rate': rec.predicted_enrollment_rate,
                'estimated_cost': rec.estimated_cost,
                'confidence': rec.confidence_level,
                'success_rate': rec.site.success_rate,
                'total_trials': rec.site.total_trials,
                'active_trials': rec.site.active_trials,
                'performance_score': rec.site.performance_score,
                'strengths': '; '.join(rec.strengths),
                'concerns': '; '.join(rec.concerns)
            })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    print("Site Intelligence Engine - Demo")
    print("="*80)
    
    # Would load actual trial data here
    # For demo, create sample data
    sample_trials = pd.DataFrame({
        'nct_id': ['NCT001', 'NCT002', 'NCT003'],
        'overall_status': ['COMPLETED', 'RECRUITING', 'TERMINATED'],
        'phase': ['PHASE2', 'PHASE3', 'PHASE2'],
        'enrollment_actual': [100, 50, 30],
        'facility_name': ['Mass General', 'Mayo Clinic', 'Johns Hopkins'],
        'city': ['Boston', 'Rochester', 'Baltimore'],
        'country': ['United States', 'United States', 'United States'],
        'is_oncology': [True, False, True]
    })
    
    engine = SiteIntelligenceEngine()
    print("\n1. Building site database...")
    sites = engine.build_site_database(sample_trials)
    
    print("\n2. Recommending sites for oncology Phase 2 trial...")
    recommendations = engine.recommend_sites({
        'therapeutic_area': 'Oncology',
        'phase': 'PHASE2',
        'target_enrollment': 150
    }, num_sites=5)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n  Recommendation #{i}")
        print(f"  Facility: {rec.site.facility_name}")
        print(f"  Match Score: {rec.match_score:.1f}/100")
        print(f"  Predicted Rate: {rec.predicted_enrollment_rate:.1f} pts/month")
