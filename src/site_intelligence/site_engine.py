"""
FIXED Site Intelligence Engine
Improved API calls and demo data fallback
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SiteProfile:
    """Simplified site profile"""
    site_id: str
    facility_name: str
    city: str
    state: str
    country: str
    
    total_trials: int
    active_trials: int
    completed_trials: int
    success_rate: float
    
    avg_enrollment_rate: float
    therapeutic_areas: Dict[str, int]
    
    performance_score: float


@dataclass
class SiteRecommendation:
    """Site recommendation with reasoning"""
    site: SiteProfile
    match_score: float
    strengths: List[str]
    concerns: List[str]
    predicted_enrollment_rate: float
    estimated_cost: float
    confidence_level: str


class SiteIntelligenceEngine:
    """
    Fixed Site Intelligence Engine with:
    - Improved API calls with better error handling
    - Demo data fallback when API fails
    - Detailed debugging output
    """
    
    def __init__(self):
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
        self.site_database = {}
        self.session = requests.Session()
        
    def fetch_trials_for_sites(self, max_trials: int = 500) -> List[Dict]:
        """
        Fetch trials with location/facility data from ClinicalTrials.gov
        
        FIXED: Better API query and error handling
        """
        
        print(f"\n🔍 Fetching trials with location data from API...")
        print(f"API URL: {self.base_url}")
        
        # API parameters - get recruiting trials (simpler query that always works)
        # We'll filter for location data after fetching
        params = {
            'filter.overallStatus': 'RECRUITING,ACTIVE_NOT_RECRUITING',
            'pageSize': 100,  # Max per page
            'format': 'json'
        }
        
        print(f"Parameters: {params}")
        
        all_trials = []
        
        try:
            # Make API request
            print("Making API request...")
            response = self.session.get(self.base_url, params=params, timeout=30)
            
            print(f"Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ API Error: Status {response.status_code}")
                print(f"Response: {response.text[:500]}")
                return []
            
            data = response.json()
            
            # Check if we got studies
            if 'studies' not in data:
                print(f"❌ No 'studies' key in response")
                print(f"Response keys: {list(data.keys())}")
                return []
            
            studies = data.get('studies', [])
            print(f"✓ API returned {len(studies)} studies")
            
            # Extract trial data
            for study in studies:
                try:
                    trial_data = self._extract_trial_data(study)
                    if trial_data:
                        all_trials.append(trial_data)
                except Exception as e:
                    print(f"Warning: Could not extract data from study: {e}")
                    continue
            
            print(f"✓ Successfully extracted {len(all_trials)} trials")
            
            if len(all_trials) == 0:
                print("⚠ Warning: API returned studies but no data could be extracted")
                
            return all_trials
            
        except requests.exceptions.Timeout:
            print("❌ API Request Timeout")
            return []
        except requests.exceptions.ConnectionError:
            print("❌ API Connection Error")
            return []
        except Exception as e:
            print(f"❌ API Exception: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _extract_trial_data(self, study: Dict) -> Optional[Dict]:
        """Extract trial and site data from API response"""
        
        try:
            protocol_section = study.get('protocolSection', {})
            id_module = protocol_section.get('identificationModule', {})
            status_module = protocol_section.get('statusModule', {})
            
            # Extract location information
            locations_module = protocol_section.get('contactsLocationsModule', {})
            locations = locations_module.get('locations', [])
            
            if not locations:
                return None
            
            # Get first location for simplicity
            location = locations[0]
            facility = location.get('facility', 'Unknown Facility')
            city = location.get('city', 'Unknown')
            state = location.get('state', '')
            country = location.get('country', 'Unknown')
            
            trial_data = {
                'nct_id': id_module.get('nctId', 'UNKNOWN'),
                'overall_status': status_module.get('overallStatus', 'UNKNOWN'),
                'facility_name': facility,
                'city': city,
                'state': state,
                'country': country
            }
            
            return trial_data
            
        except Exception as e:
            print(f"Error extracting trial data: {e}")
            return None
    
    def _create_demo_sites(self):
        """Create realistic demo sites when API fails"""
        
        print("\n⚠ Creating demo site database for testing...")
        
        demo_sites = [
            {
                'facility_name': 'Memorial Sloan Kettering Cancer Center',
                'city': 'New York',
                'state': 'NY',
                'country': 'United States',
                'total_trials': 150,
                'active_trials': 25,
                'completed_trials': 120,
                'success_rate': 87.0,
                'therapeutic_areas': {'Oncology': 120, 'Hematology': 30},
                'avg_enrollment_rate': 12.5
            },
            {
                'facility_name': 'Mayo Clinic',
                'city': 'Rochester',
                'state': 'MN',
                'country': 'United States',
                'total_trials': 200,
                'active_trials': 30,
                'completed_trials': 165,
                'success_rate': 92.0,
                'therapeutic_areas': {'Oncology': 80, 'Cardiology': 60, 'CNS': 40},
                'avg_enrollment_rate': 15.0
            },
            {
                'facility_name': 'MD Anderson Cancer Center',
                'city': 'Houston',
                'state': 'TX',
                'country': 'United States',
                'total_trials': 180,
                'active_trials': 35,
                'completed_trials': 140,
                'success_rate': 89.0,
                'therapeutic_areas': {'Oncology': 150, 'Hematology': 30},
                'avg_enrollment_rate': 14.0
            },
            {
                'facility_name': 'Johns Hopkins Hospital',
                'city': 'Baltimore',
                'state': 'MD',
                'country': 'United States',
                'total_trials': 160,
                'active_trials': 28,
                'completed_trials': 130,
                'success_rate': 88.0,
                'therapeutic_areas': {'Oncology': 70, 'Neurology': 50, 'Cardiology': 40},
                'avg_enrollment_rate': 13.0
            },
            {
                'facility_name': 'Dana-Farber Cancer Institute',
                'city': 'Boston',
                'state': 'MA',
                'country': 'United States',
                'total_trials': 140,
                'active_trials': 22,
                'completed_trials': 115,
                'success_rate': 86.0,
                'therapeutic_areas': {'Oncology': 120, 'Hematology': 20},
                'avg_enrollment_rate': 11.5
            },
            {
                'facility_name': 'Cleveland Clinic',
                'city': 'Cleveland',
                'state': 'OH',
                'country': 'United States',
                'total_trials': 170,
                'active_trials': 26,
                'completed_trials': 140,
                'success_rate': 90.0,
                'therapeutic_areas': {'Cardiology': 80, 'Oncology': 50, 'Neurology': 40},
                'avg_enrollment_rate': 13.5
            },
            {
                'facility_name': 'Stanford Cancer Institute',
                'city': 'Stanford',
                'state': 'CA',
                'country': 'United States',
                'total_trials': 130,
                'active_trials': 24,
                'completed_trials': 105,
                'success_rate': 85.0,
                'therapeutic_areas': {'Oncology': 100, 'Immunology': 30},
                'avg_enrollment_rate': 10.5
            },
            {
                'facility_name': 'University of Pennsylvania',
                'city': 'Philadelphia',
                'state': 'PA',
                'country': 'United States',
                'total_trials': 145,
                'active_trials': 23,
                'completed_trials': 120,
                'success_rate': 87.0,
                'therapeutic_areas': {'Oncology': 80, 'CAR-T': 40, 'Immunology': 25},
                'avg_enrollment_rate': 12.0
            }
        ]
        
        for site_data in demo_sites:
            site_id = f"{site_data['facility_name']}_{site_data['city']}".lower().replace(' ', '_')
            
            profile = SiteProfile(
                site_id=site_id,
                facility_name=site_data['facility_name'],
                city=site_data['city'],
                state=site_data['state'],
                country=site_data['country'],
                total_trials=site_data['total_trials'],
                active_trials=site_data['active_trials'],
                completed_trials=site_data['completed_trials'],
                success_rate=site_data['success_rate'],
                avg_enrollment_rate=site_data['avg_enrollment_rate'],
                therapeutic_areas=site_data['therapeutic_areas'],
                performance_score=site_data['success_rate']
            )
            
            self.site_database[site_id] = profile
        
        print(f"✓ Created {len(demo_sites)} demo sites")
    
    def build_site_database(self, trials_df: Optional[pd.DataFrame] = None):
        """
        Build site database from trials data or API
        
        FIXED: Better error handling and demo data fallback
        """
        
        print("\n" + "="*80)
        print("Building Site Intelligence Database")
        print("="*80)
        
        if trials_df is not None:
            # Use provided dataframe
            print(f"Using provided dataframe with {len(trials_df)} trials")
            # ... existing dataframe processing code ...
            return self.site_database
        
        # Fetch from API
        trials = self.fetch_trials_for_sites()
        
        if not trials or len(trials) == 0:
            print("\n⚠ API returned no data - using demo sites instead")
            self._create_demo_sites()
            return self.site_database
        
        # Build database from API data
        print(f"\nProcessing {len(trials)} trials...")
        
        site_data = defaultdict(lambda: {
            'trials': [],
            'statuses': [],
            'facilities': []
        })
        
        for trial in trials:
            facility = trial.get('facility_name', '')
            city = trial.get('city', '')
            country = trial.get('country', '')
            
            site_key = f"{facility}_{city}_{country}".lower().replace(' ', '_')
            
            site_data[site_key]['trials'].append(trial['nct_id'])
            site_data[site_key]['statuses'].append(trial['overall_status'])
            site_data[site_key]['facilities'].append({
                'name': facility,
                'city': city,
                'state': trial.get('state', ''),
                'country': country
            })
        
        # Create site profiles
        for site_key, data in site_data.items():
            if len(data['trials']) >= 2:
                facility_info = data['facilities'][0]
                statuses = data['statuses']
                
                total_trials = len(data['trials'])
                completed = sum(1 for s in statuses if s == 'COMPLETED')
                active = sum(1 for s in statuses if s in ['RECRUITING', 'ACTIVE_NOT_RECRUITING'])
                
                success_rate = (completed / total_trials * 100) if total_trials > 0 else 50.0
                
                profile = SiteProfile(
                    site_id=site_key,
                    facility_name=facility_info['name'],
                    city=facility_info['city'],
                    state=facility_info['state'],
                    country=facility_info['country'],
                    total_trials=total_trials,
                    active_trials=active,
                    completed_trials=completed,
                    success_rate=success_rate,
                    avg_enrollment_rate=5.0,  # Default
                    therapeutic_areas={'General': total_trials},
                    performance_score=success_rate
                )
                
                self.site_database[site_key] = profile
        
        print(f"✓ Built database with {len(self.site_database)} sites")
        return self.site_database
    
    def rank_sites(self, trial_requirements: Dict, top_n: int = 20) -> List:
        """Rank sites based on requirements"""
        
        if not self.site_database:
            raise ValueError("Site database not built. Call build_site_database() first.")
        
        ranked = []
        
        for site in self.site_database.values():
            score = self._calculate_match_score(site, trial_requirements)
            ranked.append((site, score))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_n]
    
    def _calculate_match_score(self, site: SiteProfile, requirements: Dict) -> float:
        """Calculate match score for site"""
        
        score = site.performance_score
        
        # Boost for therapeutic area match
        if 'therapeutic_area' in requirements:
            area = requirements['therapeutic_area']
            if area in site.therapeutic_areas:
                score += 10
        
        # Adjust for capacity
        if site.active_trials < 30:
            score += 5
        
        return min(100, score)
    
    def recommend_sites(self, trial_requirements: Dict, num_sites: int = 10) -> List[SiteRecommendation]:
        """Generate site recommendations"""
        
        ranked_sites = self.rank_sites(trial_requirements, top_n=num_sites)
        
        recommendations = []
        for site, score in ranked_sites:
            rec = SiteRecommendation(
                site=site,
                match_score=score,
                strengths=[
                    f"Success rate: {site.success_rate:.0f}%",
                    f"Experience: {site.total_trials} trials"
                ],
                concerns=[
                    f"Active load: {site.active_trials} trials"
                ] if site.active_trials > 30 else [],
                predicted_enrollment_rate=site.avg_enrollment_rate,
                estimated_cost=250000.0,
                confidence_level='high' if score > 80 else 'medium'
            )
            recommendations.append(rec)
        
        return recommendations


if __name__ == "__main__":
    print("Site Intelligence Engine - Improved API Version")
    print("="*80)
    
    engine = SiteIntelligenceEngine()
    
    print("\n1. Building site database from API (with fallback to demo)...")
    sites = engine.build_site_database()
    
    if not sites:
        print("❌ Failed to build site database")
    else:
        print(f"\n2. Recommending sites for oncology Phase 2 trial...")
        recommendations = engine.recommend_sites({
            'therapeutic_area': 'Oncology',
            'phase': 'PHASE2',
            'target_enrollment': 150
        }, num_sites=5)
        
        print(f"\nTop {len(recommendations)} Recommendations:")
        print("-"*80)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n#{i}: {rec.site.facility_name} ({rec.site.city}, {rec.site.country})")
            print(f"  Match Score: {rec.match_score:.1f}/100")
            print(f"  Success Rate: {rec.site.success_rate:.0f}%")
            print(f"  Total Trials: {rec.site.total_trials}")
            print(f"  Predicted Rate: {rec.predicted_enrollment_rate:.1f} pts/month")
