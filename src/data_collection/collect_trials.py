"""
Clinical Trials Data Collector
Fetches data from ClinicalTrials.gov API v2
"""

import requests
import pandas as pd
import time
import json
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import sys


class ClinicalTrialsCollector:
    """Collect clinical trial data from ClinicalTrials.gov API"""
    
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
    
    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self.session = requests.Session()
        
    def fetch_trials(
        self,
        phases: Optional[List[str]] = None,
        statuses: Optional[List[str]] = None,
        max_studies: int = 2000,
        study_type: str = "INTERVENTIONAL"
    ) -> pd.DataFrame:
        """
        Fetch clinical trials from API
        
        Args:
            phases: ['PHASE1', 'PHASE2', 'PHASE3', 'PHASE4']
            statuses: ['COMPLETED', 'TERMINATED', 'RECRUITING', 'ACTIVE_NOT_RECRUITING']
            max_studies: Maximum number of studies to fetch
            study_type: Usually 'INTERVENTIONAL' for drug trials
        """
        
        all_studies = []
        next_page_token = None
        fetched = 0
        
        print(f"Fetching up to {max_studies} clinical trials...")
        
        while fetched < max_studies:
            params = {
                'pageSize': min(100, max_studies - fetched),
                'format': 'json'
            }
            
            # Build query filters
            query_parts = []
            if study_type:
                query_parts.append(f'AREA[StudyType]{study_type}')
            if phases:
                phase_query = ' OR '.join([f'AREA[Phase]{p}' for p in phases])
                query_parts.append(f'({phase_query})')
            if statuses:
                status_query = ' OR '.join([f'AREA[OverallStatus]{s}' for s in statuses])
                query_parts.append(f'({status_query})')
            
            if query_parts:
                params['query.term'] = ' AND '.join(query_parts)
            
            if next_page_token:
                params['pageToken'] = next_page_token
            
            try:
                response = self.session.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                studies = data.get('studies', [])
                if not studies:
                    print("No more studies found.")
                    break
                
                all_studies.extend(studies)
                fetched += len(studies)
                print(f"Fetched {fetched}/{max_studies} studies...")
                
                next_page_token = data.get('nextPageToken')
                if not next_page_token:
                    break
                
                time.sleep(self.rate_limit)
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        print(f"Total studies collected: {len(all_studies)}")
        
        # Parse into structured data
        df = self._parse_studies(all_studies)
        return df
    
    def _parse_studies(self, studies: List[Dict]) -> pd.DataFrame:
        """Parse raw JSON studies into structured DataFrame"""
        
        records = []
        
        for study in studies:
            try:
                protocol = study.get('protocolSection', {})
                
                # Identification
                ident = protocol.get('identificationModule', {})
                
                # Status
                status = protocol.get('statusModule', {})
                
                # Design
                design = protocol.get('designModule', {})
                
                # Arms/Interventions
                arms = protocol.get('armsInterventionsModule', {})
                
                # Outcomes
                outcomes = protocol.get('outcomesModule', {})
                
                # Eligibility
                eligibility = protocol.get('eligibilityModule', {})
                
                # Sponsor
                sponsor = protocol.get('sponsorCollaboratorsModule', {})
                
                # Locations
                locations = protocol.get('contactsLocationsModule', {})
                
                # Conditions
                conditions_mod = protocol.get('conditionsModule', {})
                
                record = {
                    # Identifiers
                    'nct_id': ident.get('nctId'),
                    'brief_title': ident.get('briefTitle'),
                    'official_title': ident.get('officialTitle'),
                    
                    # Status & Dates
                    'overall_status': status.get('overallStatus'),
                    'start_date': self._safe_date(status.get('startDateStruct')),
                    'completion_date': self._safe_date(status.get('completionDateStruct')),
                    'last_update': status.get('lastUpdatePostDateStruct', {}).get('date'),
                    
                    # Study Design
                    'study_type': design.get('studyType'),
                    'phase': '|'.join(design.get('phases', [])),
                    'enrollment': design.get('enrollmentInfo', {}).get('count'),
                    'enrollment_type': design.get('enrollmentInfo', {}).get('type'),
                    'allocation': design.get('designInfo', {}).get('allocation'),
                    'intervention_model': design.get('designInfo', {}).get('interventionModel'),
                    'primary_purpose': design.get('designInfo', {}).get('primaryPurpose'),
                    'masking': design.get('designInfo', {}).get('maskingInfo', {}).get('masking'),
                    
                    # Conditions
                    'condition': '|'.join(conditions_mod.get('conditions', [])),
                    
                    # Interventions
                    'intervention_name': '|'.join([
                        i.get('name', '') for i in arms.get('interventions', [])
                    ]),
                    'intervention_type': '|'.join(list(set([
                        i.get('type', '') for i in arms.get('interventions', [])
                    ]))),
                    
                    # Outcomes
                    'primary_outcome_count': len(outcomes.get('primaryOutcomes', [])),
                    'secondary_outcome_count': len(outcomes.get('secondaryOutcomes', [])),
                    
                    # Eligibility
                    'min_age': eligibility.get('minimumAge'),
                    'max_age': eligibility.get('maximumAge'),
                    'sex': eligibility.get('sex'),
                    'healthy_volunteers': eligibility.get('healthyVolunteers'),
                    
                    # Sponsor
                    'lead_sponsor_name': sponsor.get('leadSponsor', {}).get('name'),
                    'lead_sponsor_class': sponsor.get('leadSponsor', {}).get('class'),
                    'collaborator_count': len(sponsor.get('collaborators', [])),
                    
                    # Geography
                    'location_count': len(locations.get('locations', [])),
                    'countries': '|'.join(list(set([
                        loc.get('country', '') for loc in locations.get('locations', [])
                    ]))),
                }
                
                records.append(record)
                
            except Exception as e:
                print(f"Error parsing study: {e}")
                continue
        
        return pd.DataFrame(records)
    
    def _safe_date(self, date_struct: Optional[Dict]) -> Optional[str]:
        """Extract date safely from ClinicalTrials.gov date structure"""
        if not date_struct:
            return None
        return date_struct.get('date')


def main():
    """Main collection script for biotech-relevant trials"""
    
    collector = ClinicalTrialsCollector(rate_limit=0.5)
    
    # Focus on Phase 2 and 3 trials (most predictive of success/failure patterns)
    # Get both completed (success) and terminated (failure) for modeling
    print("\n=== Collecting Phase 2-3 Drug Development Trials ===\n")
    
    df = collector.fetch_trials(
        phases=['PHASE2', 'PHASE3'],
        statuses=['COMPLETED', 'TERMINATED', 'ACTIVE_NOT_RECRUITING', 'WITHDRAWN'],
        max_studies=2000,
        study_type='INTERVENTIONAL'
    )
    
    # Save data
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'clinical_trials_raw_{timestamp}.csv'
    
    df.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")
    print(f"Shape: {df.shape}")
    
    # Quick summary
    print("\n=== Data Summary ===")
    print(f"Total trials: {len(df)}")
    print(f"\nPhase distribution:")
    print(df['phase'].value_counts())
    print(f"\nStatus distribution:")
    print(df['overall_status'].value_counts())
    print(f"\nSponsor class distribution:")
    print(df['lead_sponsor_class'].value_counts())
    
    return df


if __name__ == '__main__':
    df = main()
