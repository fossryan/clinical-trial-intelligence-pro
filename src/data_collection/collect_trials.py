"""
Enhanced Clinical Trials Collector - 10,000+ Trials
Comprehensive collection strategy for maximum coverage and quality

Run time: 60-90 minutes (API rate limiting)
Output: 10,000+ clinical trials from ClinicalTrials.gov
"""

import requests
import pandas as pd
import time
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path


class EnhancedTrialCollector:
    """Collect 10,000+ trials using multi-phase strategy"""
    
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
    
    def __init__(self, rate_limit: float = 0.5):
        self.rate_limit = rate_limit
        self.session = requests.Session()
    
    def collect_comprehensive_dataset(self, target_total: int = 10000) -> pd.DataFrame:
        """
        Comprehensive collection strategy:
        - All phases (1, 2, 3, 4, combined)
        - Multiple statuses (completed, terminated, recruiting, active)
        - Recent trials (2010-2025)
        - Better sponsor diversity
        """
        
        all_trials = []
        
        # Multi-batch strategy for comprehensive coverage
        batches = [
            {
                'name': 'Phase 2-3 Completed/Terminated (Core Dataset)',
                'phases': ['PHASE2', 'PHASE3'],
                'statuses': ['COMPLETED', 'TERMINATED'],
                'years': '2010-01-01',
                'target': 3500
            },
            {
                'name': 'Phase 2-3 Active/Recruiting',
                'phases': ['PHASE2', 'PHASE3'],
                'statuses': ['RECRUITING', 'ACTIVE_NOT_RECRUITING', 'ENROLLING_BY_INVITATION'],
                'years': '2015-01-01',
                'target': 2000
            },
            {
                'name': 'Phase 1 All Statuses',
                'phases': ['PHASE1'],
                'statuses': ['COMPLETED', 'TERMINATED', 'RECRUITING', 'ACTIVE_NOT_RECRUITING'],
                'years': '2015-01-01',
                'target': 2000
            },
            {
                'name': 'Phase 4 Post-Marketing',
                'phases': ['PHASE4'],
                'statuses': ['COMPLETED', 'TERMINATED', 'RECRUITING'],
                'years': '2015-01-01',
                'target': 1500
            },
            {
                'name': 'Combined Phases (1/2, 2/3)',
                'phases': ['PHASE1|PHASE2', 'PHASE2|PHASE3'],
                'statuses': ['COMPLETED', 'TERMINATED', 'RECRUITING'],
                'years': '2015-01-01',
                'target': 1000
            }
        ]
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE TRIAL COLLECTION - TARGET: {target_total:,} TRIALS")
        print(f"{'='*80}\n")
        
        for i, batch in enumerate(batches, 1):
            print(f"\n[Batch {i}/{len(batches)}] {batch['name']}")
            print(f"Target: {batch['target']:,} trials")
            print("-" * 80)
            
            trials = self._fetch_batch(
                phases=batch['phases'],
                statuses=batch['statuses'],
                start_year=batch['years'],
                max_studies=batch['target']
            )
            
            all_trials.extend(trials)
            print(f"✓ Collected: {len(trials):,} trials")
            print(f"✓ Running total: {len(all_trials):,} trials")
            
            # Be nice to the API
            time.sleep(2)
            
            # Stop if we hit target
            if len(all_trials) >= target_total:
                print(f"\n✓ Target of {target_total:,} trials reached!")
                break
        
        # Parse and clean
        print(f"\n{'='*80}")
        print("PROCESSING DATA...")
        print(f"{'='*80}\n")
        
        df = self._parse_studies(all_trials)
        
        # Remove duplicates
        original = len(df)
        df = df.drop_duplicates(subset=['nct_id'], keep='first')
        removed = original - len(df)
        
        print(f"✓ Parsed {original:,} trials")
        print(f"✓ Removed {removed:,} duplicates")
        print(f"✓ Final unique trials: {len(df):,}")
        
        return df
    
    def _fetch_batch(
        self,
        phases: List[str],
        statuses: List[str],
        start_year: str,
        max_studies: int
    ) -> List[Dict]:
        """Fetch one batch of trials"""
        
        all_studies = []
        next_page_token = None
        fetched = 0
        page = 1
        
        while fetched < max_studies:
            # Build query
            query_parts = []
            
            # Interventional studies only
            query_parts.append('AREA[StudyType]INTERVENTIONAL')
            
            # Phases
            phase_query = ' OR '.join([f'AREA[Phase]{p}' for p in phases])
            query_parts.append(f'({phase_query})')
            
            # Statuses
            status_query = ' OR '.join([f'AREA[OverallStatus]{s}' for s in statuses])
            query_parts.append(f'({status_query})')
            
            # Date range
            query_parts.append(f'AREA[StartDate]RANGE[{start_year},MAX]')
            
            params = {
                'query.term': ' AND '.join(query_parts),
                'pageSize': min(100, max_studies - fetched),
                'format': 'json'
            }
            
            if next_page_token:
                params['pageToken'] = next_page_token
            
            try:
                response = self.session.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                studies = data.get('studies', [])
                if not studies:
                    break
                
                all_studies.extend(studies)
                fetched += len(studies)
                
                print(f"  Page {page}: {len(studies)} trials (total: {fetched:,}/{max_studies:,})")
                page += 1
                
                next_page_token = data.get('nextPageToken')
                if not next_page_token:
                    break
                
                time.sleep(self.rate_limit)
                
            except Exception as e:
                print(f"  ⚠ Error: {e}")
                break
        
        return all_studies
    
    def _parse_studies(self, studies: List[Dict]) -> pd.DataFrame:
        """Parse JSON studies into DataFrame"""
        
        records = []
        errors = 0
        
        for study in studies:
            try:
                protocol = study.get('protocolSection', {})
                
                # Modules
                ident = protocol.get('identificationModule', {})
                status = protocol.get('statusModule', {})
                design = protocol.get('designModule', {})
                arms = protocol.get('armsInterventionsModule', {})
                outcomes = protocol.get('outcomesModule', {})
                eligibility = protocol.get('eligibilityModule', {})
                sponsor = protocol.get('sponsorCollaboratorsModule', {})
                locations = protocol.get('contactsLocationsModule', {})
                conditions_mod = protocol.get('conditionsModule', {})
                
                record = {
                    # Identity
                    'nct_id': ident.get('nctId'),
                    'brief_title': ident.get('briefTitle'),
                    'official_title': ident.get('officialTitle'),
                    
                    # Status & Dates
                    'overall_status': status.get('overallStatus'),
                    'start_date': self._safe_date(status.get('startDateStruct')),
                    'completion_date': self._safe_date(status.get('completionDateStruct')),
                    'last_update': status.get('lastUpdatePostDateStruct', {}).get('date'),
                    
                    # Design
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
                    'intervention_name': '|'.join([i.get('name', '') for i in arms.get('interventions', [])]),
                    'intervention_type': '|'.join(list(set([i.get('type', '') for i in arms.get('interventions', [])]))),
                    
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
                errors += 1
                continue
        
        if errors > 0:
            print(f"  ⚠ {errors} parsing errors (non-critical)")
        
        return pd.DataFrame(records)
    
    def _safe_date(self, date_struct: Optional[Dict]) -> Optional[str]:
        """Extract date safely"""
        if not date_struct:
            return None
        return date_struct.get('date')


def main():
    """
    Main execution - Collect 10,000+ trials
    
    Estimated runtime: 60-90 minutes
    """
    
    print("\n" + "="*80)
    print("ENHANCED CLINICAL TRIAL COLLECTION - 10,000+ TRIALS")
    print("="*80)
    print("\n📊 Collection Strategy:")
    print("  • All phases: 1, 2, 3, 4, combined")
    print("  • All statuses: completed, terminated, recruiting, active")
    print("  • Date range: 2010-2025 (15 years)")
    print("  • Study type: Interventional only")
    print("\n⏱ Estimated time: 60-90 minutes")
    print("💾 Output: data/raw/clinical_trials_raw_10k_YYYYMMDD_HHMMSS.csv")
    print("="*80 + "\n")
    
    # Confirm
    response = input("Start collection? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ Collection cancelled.")
        return None
    
    start_time = datetime.now()
    print(f"\n🚀 Starting collection at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Collect
    collector = EnhancedTrialCollector(rate_limit=0.5)
    df = collector.collect_comprehensive_dataset(target_total=10000)
    
    # Save
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'clinical_trials_raw_10k_{timestamp}.csv'
    
    df.to_csv(output_file, index=False)
    
    # Summary
    elapsed = datetime.now() - start_time
    minutes = elapsed.total_seconds() / 60
    
    print(f"\n{'='*80}")
    print("✅ COLLECTION COMPLETE!")
    print(f"{'='*80}\n")
    print(f"📁 File: {output_file}")
    print(f"📊 Trials: {len(df):,}")
    print(f"⏱ Time: {minutes:.1f} minutes")
    print(f"📏 Size: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Phase distribution
    print(f"\n{'='*80}")
    print("PHASE DISTRIBUTION")
    print(f"{'='*80}")
    phase_counts = df['phase'].value_counts().head(15)
    for phase, count in phase_counts.items():
        pct = (count / len(df)) * 100
        bar = '█' * int(pct / 2)
        print(f"{phase:25s}: {count:5,} ({pct:5.1f}%) {bar}")
    
    # Status distribution
    print(f"\n{'='*80}")
    print("STATUS DISTRIBUTION")
    print(f"{'='*80}")
    status_counts = df['overall_status'].value_counts()
    for status, count in status_counts.items():
        pct = (count / len(df)) * 100
        bar = '█' * int(pct / 2)
        print(f"{status:30s}: {count:5,} ({pct:5.1f}%) {bar}")
    
    # Sponsor types
    print(f"\n{'='*80}")
    print("SPONSOR TYPES")
    print(f"{'='*80}")
    sponsor_counts = df['lead_sponsor_class'].value_counts()
    for sponsor_type, count in sponsor_counts.items():
        pct = (count / len(df)) * 100
        bar = '█' * int(pct / 2)
        print(f"{sponsor_type:20s}: {count:5,} ({pct:5.1f}%) {bar}")
    
    # Top sponsors
    print(f"\n{'='*80}")
    print("TOP 30 SPONSORS BY TRIAL COUNT")
    print(f"{'='*80}")
    top_sponsors = df['lead_sponsor_name'].value_counts().head(30)
    for rank, (sponsor, count) in enumerate(top_sponsors.items(), 1):
        print(f"{rank:2d}. {sponsor[:60]:60s}: {count:4,} trials")
    
    # Year distribution
    df_temp = df.copy()
    df_temp['start_year'] = pd.to_datetime(df_temp['start_date'], errors='coerce').dt.year
    year_dist = df_temp['start_year'].value_counts().sort_index()
    
    print(f"\n{'='*80}")
    print("TRIALS BY START YEAR (Recent 10 years)")
    print(f"{'='*80}")
    recent_years = sorted([y for y in year_dist.index if pd.notna(y) and y >= 2015], reverse=True)
    for year in recent_years[:10]:
        count = year_dist[year]
        bar = '█' * int(count / 50)
        print(f"{int(year)}: {count:5,} trials {bar}")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}\n")
    print("1️⃣  Run feature engineering:")
    print("   python src/features/engineer_features.py")
    print("\n2️⃣  Train models on expanded dataset:")
    print("   python src/models/train_models.py")
    print("\n3️⃣  Your app will now have 5X more data!")
    print("   • Better competitive intelligence (more companies)")
    print("   • Higher model accuracy")
    print("   • More statistical significance")
    print(f"\n{'='*80}\n")
    
    return df


if __name__ == '__main__':
    df = main()
