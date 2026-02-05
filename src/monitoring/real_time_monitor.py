"""
Real-Time Trial Monitoring System
Enterprise-grade live trial tracking and risk alerting

Features:
- Live enrollment velocity tracking
- Automated risk detection
- Site-level performance monitoring
- Predictive timeline forecasting
- Multi-trial dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
from dataclasses import dataclass
import json


@dataclass
class TrialAlert:
    """Data class for trial alerts"""
    trial_id: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metric_value: float
    threshold_value: float
    timestamp: datetime
    recommended_action: str


@dataclass
class EnrollmentMetrics:
    """Enrollment performance metrics"""
    trial_id: str
    current_enrollment: int
    target_enrollment: int
    enrollment_rate: float  # patients per week
    weeks_elapsed: int
    weeks_remaining: int
    percent_complete: float
    velocity_score: float  # 0-100
    projected_completion: datetime
    on_track: bool


class RealTimeTrialMonitor:
    """
    Real-time monitoring system for active clinical trials
    
    Capabilities:
    - Live data refresh from ClinicalTrials.gov
    - Enrollment velocity tracking
    - Risk scoring and alerts
    - Timeline predictions
    - Site performance analysis
    """
    
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
    
    def __init__(self):
        self.session = requests.Session()
        self.alert_thresholds = self._default_thresholds()
        
    def _default_thresholds(self) -> Dict:
        """Default risk thresholds for alerting"""
        return {
            'enrollment_velocity_min': 0.7,  # 70% of target rate
            'enrollment_velocity_critical': 0.5,  # 50% of target rate
            'timeline_delay_warning': 1.2,  # 20% delay
            'timeline_delay_critical': 1.5,  # 50% delay
            'site_performance_min': 0.6,  # 60% of expected
            'dropout_rate_max': 0.15,  # 15% dropout threshold
            'data_quality_min': 0.85  # 85% data quality score
        }
    
    def fetch_active_trials(self, sponsor: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch all active trials (recruiting or active)
        
        Args:
            sponsor: Optional sponsor name to filter
            
        Returns:
            DataFrame of active trials with current status
        """
        
        params = {
            'query.term': f'AREA[OverallStatus](RECRUITING OR ACTIVE_NOT_RECRUITING)',
            'pageSize': 100,
            'format': 'json'
        }
        
        if sponsor:
            params['query.term'] += f' AND AREA[LeadSponsorName]{sponsor}'
        
        all_trials = []
        
        print(f"Fetching active trials{' for ' + sponsor if sponsor else ''}...")
        
        try:
            response = self.session.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'studies' in data:
                for study in data['studies']:
                    trial_data = self._extract_trial_data(study)
                    all_trials.append(trial_data)
            
            print(f"✓ Fetched {len(all_trials)} active trials")
            
        except Exception as e:
            print(f"Error fetching trials: {str(e)}")
            
        return pd.DataFrame(all_trials)
    
    def _extract_trial_data(self, study: Dict) -> Dict:
        """Extract key fields from API response"""
        
        protocol_section = study.get('protocolSection', {})
        id_module = protocol_section.get('identificationModule', {})
        status_module = protocol_section.get('statusModule', {})
        design_module = protocol_section.get('designModule', {})
        eligibility_module = protocol_section.get('eligibilityModule', {})
        
        return {
            'nct_id': id_module.get('nctId'),
            'title': id_module.get('briefTitle'),
            'status': status_module.get('overallStatus'),
            'phase': ','.join(design_module.get('phases', [])),
            'enrollment_actual': status_module.get('enrollmentInfo', {}).get('count', 0),
            'enrollment_type': status_module.get('enrollmentInfo', {}).get('type'),
            'start_date': status_module.get('startDateStruct', {}).get('date'),
            'completion_date': status_module.get('primaryCompletionDateStruct', {}).get('date'),
            'sponsor': protocol_section.get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}).get('name'),
            'last_update': status_module.get('lastUpdatePostDateStruct', {}).get('date')
        }
    
    def calculate_enrollment_metrics(self, trial: pd.Series, target_enrollment: Optional[int] = None) -> EnrollmentMetrics:
        """
        Calculate detailed enrollment metrics for a trial
        
        Args:
            trial: Trial data as pandas Series
            target_enrollment: Override target if known (uses actual if estimated)
            
        Returns:
            EnrollmentMetrics object with all metrics
        """
        
        # Parse dates
        start_date = pd.to_datetime(trial['start_date'], errors='coerce')
        completion_date = pd.to_datetime(trial['completion_date'], errors='coerce')
        current_date = datetime.now()
        
        # Current enrollment
        current_enrollment = trial.get('enrollment_actual', 0)
        if target_enrollment is None:
            target_enrollment = current_enrollment if trial.get('enrollment_type') == 'ACTUAL' else int(current_enrollment * 1.2)
        
        # Time calculations
        if pd.notna(start_date):
            weeks_elapsed = max(1, (current_date - start_date).days / 7)
        else:
            weeks_elapsed = 52  # default 1 year
            
        if pd.notna(completion_date):
            weeks_total = max(1, (completion_date - start_date).days / 7) if pd.notna(start_date) else 104
            weeks_remaining = max(0, (completion_date - current_date).days / 7)
        else:
            weeks_total = 104  # default 2 years
            weeks_remaining = weeks_total - weeks_elapsed
        
        # Enrollment rate
        enrollment_rate = current_enrollment / weeks_elapsed if weeks_elapsed > 0 else 0
        
        # Percent complete
        percent_complete = (current_enrollment / target_enrollment * 100) if target_enrollment > 0 else 0
        
        # Velocity score (0-100)
        # Compares actual rate vs. required rate
        required_rate = target_enrollment / weeks_total if weeks_total > 0 else 1
        velocity_score = min(100, (enrollment_rate / required_rate * 100)) if required_rate > 0 else 0
        
        # Projected completion
        if enrollment_rate > 0:
            remaining_patients = max(0, target_enrollment - current_enrollment)
            weeks_to_complete = remaining_patients / enrollment_rate
            projected_completion = current_date + timedelta(weeks=weeks_to_complete)
        else:
            projected_completion = completion_date if pd.notna(completion_date) else current_date + timedelta(weeks=weeks_remaining)
        
        # On track assessment
        time_progress = weeks_elapsed / weeks_total if weeks_total > 0 else 0
        enrollment_progress = percent_complete / 100
        on_track = enrollment_progress >= (time_progress * 0.9)  # Allow 10% variance
        
        return EnrollmentMetrics(
            trial_id=trial['nct_id'],
            current_enrollment=int(current_enrollment),
            target_enrollment=int(target_enrollment),
            enrollment_rate=round(enrollment_rate, 2),
            weeks_elapsed=int(weeks_elapsed),
            weeks_remaining=int(weeks_remaining),
            percent_complete=round(percent_complete, 1),
            velocity_score=round(velocity_score, 1),
            projected_completion=projected_completion,
            on_track=on_track
        )
    
    def detect_risks(self, trial: pd.Series, metrics: EnrollmentMetrics) -> List[TrialAlert]:
        """
        Detect risks and generate alerts for a trial
        
        Args:
            trial: Trial data
            metrics: Enrollment metrics
            
        Returns:
            List of TrialAlert objects
        """
        
        alerts = []
        
        # 1. Enrollment velocity risk
        if metrics.velocity_score < self.alert_thresholds['enrollment_velocity_critical'] * 100:
            alerts.append(TrialAlert(
                trial_id=metrics.trial_id,
                alert_type='enrollment_velocity',
                severity='critical',
                message=f'Critical: Enrollment at {metrics.velocity_score:.0f}% of target velocity',
                metric_value=metrics.velocity_score,
                threshold_value=self.alert_thresholds['enrollment_velocity_critical'] * 100,
                timestamp=datetime.now(),
                recommended_action='Consider protocol amendments, add sites, or increase recruitment budget'
            ))
        elif metrics.velocity_score < self.alert_thresholds['enrollment_velocity_min'] * 100:
            alerts.append(TrialAlert(
                trial_id=metrics.trial_id,
                alert_type='enrollment_velocity',
                severity='high',
                message=f'Warning: Enrollment at {metrics.velocity_score:.0f}% of target velocity',
                metric_value=metrics.velocity_score,
                threshold_value=self.alert_thresholds['enrollment_velocity_min'] * 100,
                timestamp=datetime.now(),
                recommended_action='Monitor closely and review recruitment strategies'
            ))
        
        # 2. Timeline delay risk
        if pd.notna(trial['completion_date']):
            planned_completion = pd.to_datetime(trial['completion_date'])
            delay_ratio = (metrics.projected_completion - planned_completion).days / max(1, (planned_completion - datetime.now()).days)
            
            if delay_ratio > self.alert_thresholds['timeline_delay_critical']:
                alerts.append(TrialAlert(
                    trial_id=metrics.trial_id,
                    alert_type='timeline_delay',
                    severity='critical',
                    message=f'Critical: Projected {int(delay_ratio * 100)}% timeline delay',
                    metric_value=delay_ratio * 100,
                    threshold_value=self.alert_thresholds['timeline_delay_critical'] * 100,
                    timestamp=datetime.now(),
                    recommended_action='Immediate action required - consider protocol modifications or additional sites'
                ))
            elif delay_ratio > self.alert_thresholds['timeline_delay_warning']:
                alerts.append(TrialAlert(
                    trial_id=metrics.trial_id,
                    alert_type='timeline_delay',
                    severity='high',
                    message=f'Warning: Projected {int(delay_ratio * 100)}% timeline delay',
                    metric_value=delay_ratio * 100,
                    threshold_value=self.alert_thresholds['timeline_delay_warning'] * 100,
                    timestamp=datetime.now(),
                    recommended_action='Review enrollment strategies and site performance'
                ))
        
        # 3. Enrollment stagnation (if no recent progress)
        if metrics.enrollment_rate < 0.5 and metrics.percent_complete < 90:
            alerts.append(TrialAlert(
                trial_id=metrics.trial_id,
                alert_type='enrollment_stagnation',
                severity='high',
                message=f'Enrollment rate critically low: {metrics.enrollment_rate:.2f} patients/week',
                metric_value=metrics.enrollment_rate,
                threshold_value=0.5,
                timestamp=datetime.now(),
                recommended_action='Investigate site issues, patient eligibility criteria, or competitive trials'
            ))
        
        # 4. Behind schedule risk
        if not metrics.on_track and metrics.percent_complete < 75:
            alerts.append(TrialAlert(
                trial_id=metrics.trial_id,
                alert_type='behind_schedule',
                severity='medium',
                message=f'Trial behind schedule: {metrics.percent_complete:.1f}% complete vs expected progress',
                metric_value=metrics.percent_complete,
                threshold_value=metrics.weeks_elapsed / (metrics.weeks_elapsed + metrics.weeks_remaining) * 100,
                timestamp=datetime.now(),
                recommended_action='Review and adjust recruitment plan'
            ))
        
        return alerts
    
    def monitor_portfolio(self, trials_df: pd.DataFrame) -> Dict:
        """
        Monitor entire portfolio of trials
        
        Args:
            trials_df: DataFrame of trials to monitor
            
        Returns:
            Dictionary with portfolio analytics and alerts
        """
        
        all_metrics = []
        all_alerts = []
        
        for _, trial in trials_df.iterrows():
            # Calculate metrics
            metrics = self.calculate_enrollment_metrics(trial)
            all_metrics.append(metrics)
            
            # Detect risks
            alerts = self.detect_risks(trial, metrics)
            all_alerts.extend(alerts)
        
        # Portfolio statistics
        metrics_df = pd.DataFrame([vars(m) for m in all_metrics])
        
        portfolio_stats = {
            'total_trials': len(trials_df),
            'on_track_trials': metrics_df['on_track'].sum(),
            'at_risk_trials': len(trials_df) - metrics_df['on_track'].sum(),
            'avg_velocity_score': metrics_df['velocity_score'].mean(),
            'avg_enrollment_rate': metrics_df['enrollment_rate'].mean(),
            'total_alerts': len(all_alerts),
            'critical_alerts': len([a for a in all_alerts if a.severity == 'critical']),
            'high_alerts': len([a for a in all_alerts if a.severity == 'high']),
            'medium_alerts': len([a for a in all_alerts if a.severity == 'medium']),
            'low_alerts': len([a for a in all_alerts if a.severity == 'low'])
        }
        
        return {
            'metrics': all_metrics,
            'alerts': all_alerts,
            'portfolio_stats': portfolio_stats,
            'metrics_df': metrics_df
        }
    
    def forecast_enrollment_timeline(self, trial: pd.Series, scenarios: Dict[str, float] = None) -> Dict:
        """
        Forecast enrollment completion under different scenarios
        
        Args:
            trial: Trial data
            scenarios: Dict of scenario_name: enrollment_rate_multiplier
            
        Returns:
            Dictionary with scenario forecasts
        """
        
        if scenarios is None:
            scenarios = {
                'current_pace': 1.0,
                'optimistic_20%': 1.2,
                'pessimistic_20%': 0.8,
                'double_recruitment': 2.0
            }
        
        base_metrics = self.calculate_enrollment_metrics(trial)
        forecasts = {}
        
        for scenario_name, multiplier in scenarios.items():
            adjusted_rate = base_metrics.enrollment_rate * multiplier
            remaining_patients = max(0, base_metrics.target_enrollment - base_metrics.current_enrollment)
            
            if adjusted_rate > 0:
                weeks_to_complete = remaining_patients / adjusted_rate
                completion_date = datetime.now() + timedelta(weeks=weeks_to_complete)
            else:
                weeks_to_complete = float('inf')
                completion_date = None
            
            forecasts[scenario_name] = {
                'enrollment_rate': round(adjusted_rate, 2),
                'weeks_to_complete': round(weeks_to_complete, 1) if weeks_to_complete != float('inf') else None,
                'completion_date': completion_date,
                'patients_per_month': round(adjusted_rate * 4.33, 1)
            }
        
        return {
            'current_status': {
                'enrolled': base_metrics.current_enrollment,
                'target': base_metrics.target_enrollment,
                'remaining': base_metrics.target_enrollment - base_metrics.current_enrollment,
                'percent_complete': base_metrics.percent_complete
            },
            'scenarios': forecasts
        }
    
    def generate_risk_report(self, trials_df: pd.DataFrame, output_format: str = 'dict') -> Dict:
        """
        Generate comprehensive risk report for trials
        
        Args:
            trials_df: Trials to analyze
            output_format: 'dict' or 'json'
            
        Returns:
            Risk report with all analytics
        """
        
        monitoring_results = self.monitor_portfolio(trials_df)
        
        # Organize alerts by severity
        alerts_by_severity = {
            'critical': [a for a in monitoring_results['alerts'] if a.severity == 'critical'],
            'high': [a for a in monitoring_results['alerts'] if a.severity == 'high'],
            'medium': [a for a in monitoring_results['alerts'] if a.severity == 'medium'],
            'low': [a for a in monitoring_results['alerts'] if a.severity == 'low']
        }
        
        # Organize alerts by type
        alerts_by_type = {}
        for alert in monitoring_results['alerts']:
            if alert.alert_type not in alerts_by_type:
                alerts_by_type[alert.alert_type] = []
            alerts_by_type[alert.alert_type].append(alert)
        
        # Generate summary
        report = {
            'generated_at': datetime.now().isoformat(),
            'portfolio_overview': monitoring_results['portfolio_stats'],
            'high_risk_trials': [
                m for m in monitoring_results['metrics'] 
                if not m.on_track or m.velocity_score < 70
            ],
            'alerts_by_severity': {
                k: [vars(a) for a in v] 
                for k, v in alerts_by_severity.items()
            },
            'alerts_by_type': {
                k: [vars(a) for a in v] 
                for k, v in alerts_by_type.items()
            },
            'recommendations': self._generate_recommendations(monitoring_results)
        }
        
        if output_format == 'json':
            return json.dumps(report, default=str, indent=2)
        
        return report
    
    def _generate_recommendations(self, monitoring_results: Dict) -> List[Dict]:
        """Generate actionable recommendations based on monitoring results"""
        
        recommendations = []
        
        # High-level recommendations based on portfolio stats
        stats = monitoring_results['portfolio_stats']
        
        if stats['at_risk_trials'] > stats['on_track_trials']:
            recommendations.append({
                'priority': 'critical',
                'area': 'portfolio_health',
                'recommendation': 'More than 50% of trials are at risk. Consider portfolio-wide review of recruitment strategies.',
                'impact': 'high'
            })
        
        if stats['avg_velocity_score'] < 70:
            recommendations.append({
                'priority': 'high',
                'area': 'enrollment_velocity',
                'recommendation': f'Average velocity score is {stats["avg_velocity_score"]:.0f}%. Review site performance and recruitment tactics.',
                'impact': 'high'
            })
        
        # Alert-based recommendations
        if stats['critical_alerts'] > 0:
            recommendations.append({
                'priority': 'critical',
                'area': 'immediate_action',
                'recommendation': f'{stats["critical_alerts"]} critical alerts require immediate attention. Review high-severity alerts first.',
                'impact': 'critical'
            })
        
        return recommendations


if __name__ == "__main__":
    # Example usage
    monitor = RealTimeTrialMonitor()
    
    # Fetch active trials
    active_trials = monitor.fetch_active_trials()
    
    if not active_trials.empty:
        print(f"\nMonitoring {len(active_trials)} active trials...")
        
        # Generate risk report
        report = monitor.generate_risk_report(active_trials)
        
        print(f"\n{'='*80}")
        print("PORTFOLIO RISK REPORT")
        print(f"{'='*80}")
        print(f"Total Trials: {report['portfolio_overview']['total_trials']}")
        print(f"On Track: {report['portfolio_overview']['on_track_trials']}")
        print(f"At Risk: {report['portfolio_overview']['at_risk_trials']}")
        print(f"Average Velocity Score: {report['portfolio_overview']['avg_velocity_score']:.1f}")
        print(f"\nTotal Alerts: {report['portfolio_overview']['total_alerts']}")
        print(f"  Critical: {report['portfolio_overview']['critical_alerts']}")
        print(f"  High: {report['portfolio_overview']['high_alerts']}")
        print(f"  Medium: {report['portfolio_overview']['medium_alerts']}")
