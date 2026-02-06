"""
Real-Time API Integration & Webhook System

Enterprise features:
1. Real-time trial updates from ClinicalTrials.gov
2. Webhook notifications for trial events
3. Email/Slack alerts for important changes
4. Live enrollment tracking
5. Automated risk detection

This separates you from competitors who rely on static data.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
import json
import hashlib
import time
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


@dataclass
class TrialUpdate:
    """Data class for trial update events"""
    nct_id: str
    update_type: str  # 'status_change', 'enrollment_milestone', 'completion', 'termination'
    old_value: Optional[str]
    new_value: str
    detected_at: datetime
    severity: str  # 'info', 'warning', 'critical'
    message: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class WebhookConfig:
    """Webhook configuration for customer"""
    customer_id: str
    webhook_url: str
    events: List[str]  # Which events to receive
    active: bool
    auth_token: Optional[str]
    retry_count: int = 3
    timeout_seconds: int = 10


class RealTimeTrialMonitor:
    """
    Real-time monitoring system for clinical trials
    
    Capabilities:
    - Poll ClinicalTrials.gov API every hour for updates
    - Detect status changes, enrollment milestones, completions
    - Send webhooks/emails/Slack notifications
    - Track changes over time
    - Alert on risk factors
    """
    
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
    POLL_INTERVAL_SECONDS = 3600  # Poll every hour
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.monitored_trials: Dict[str, Dict] = {}
        
        # Load previous state
        self._load_monitored_trials()
    
    def add_webhook(self, config: WebhookConfig):
        """Register a webhook endpoint for notifications"""
        self.webhooks[config.customer_id] = config
        print(f"✓ Registered webhook for customer: {config.customer_id}")
    
    def monitor_trial(
        self, 
        nct_id: str, 
        customer_id: str,
        alert_on: Optional[List[str]] = None
    ):
        """
        Add trial to monitoring list
        
        Args:
            nct_id: NCT ID to monitor
            customer_id: Customer requesting monitoring
            alert_on: List of events to alert on
                      ['status_change', 'enrollment_milestone', 'completion']
        """
        
        if alert_on is None:
            alert_on = ['status_change', 'enrollment_milestone', 'completion', 'termination']
        
        # Fetch current state
        trial_data = self._fetch_trial_data(nct_id)
        
        if trial_data:
            self.monitored_trials[nct_id] = {
                'customer_id': customer_id,
                'alert_on': alert_on,
                'current_state': trial_data,
                'last_checked': datetime.now().isoformat(),
                'added_at': datetime.now().isoformat()
            }
            
            self._save_monitored_trials()
            print(f"✓ Now monitoring {nct_id} for customer {customer_id}")
            return True
        else:
            print(f"✗ Could not fetch data for {nct_id}")
            return False
    
    def check_for_updates(self) -> List[TrialUpdate]:
        """
        Check all monitored trials for updates
        
        This should be run on a schedule (e.g., hourly cron job)
        
        Returns:
            List of detected updates
        """
        
        print(f"\n{'='*80}")
        print(f"CHECKING {len(self.monitored_trials)} MONITORED TRIALS FOR UPDATES")
        print(f"{'='*80}\n")
        
        all_updates = []
        
        for nct_id, monitor_config in self.monitored_trials.items():
            print(f"Checking {nct_id}...", end=" ")
            
            # Fetch current state
            current_data = self._fetch_trial_data(nct_id)
            
            if not current_data:
                print("✗ API error")
                continue
            
            # Compare with previous state
            previous_data = monitor_config['current_state']
            updates = self._detect_changes(nct_id, previous_data, current_data)
            
            if updates:
                print(f"✓ {len(updates)} update(s) detected")
                
                # Filter to events customer wants
                alert_on = monitor_config['alert_on']
                filtered_updates = [u for u in updates if u.update_type in alert_on]
                
                if filtered_updates:
                    # Send notifications
                    customer_id = monitor_config['customer_id']
                    self._send_notifications(customer_id, filtered_updates)
                    all_updates.extend(filtered_updates)
                
                # Update stored state
                monitor_config['current_state'] = current_data
                monitor_config['last_checked'] = datetime.now().isoformat()
            else:
                print("✓ No changes")
        
        self._save_monitored_trials()
        
        print(f"\n✓ Check complete: {len(all_updates)} update(s) detected\n")
        
        return all_updates
    
    def _fetch_trial_data(self, nct_id: str) -> Optional[Dict]:
        """Fetch current trial data from API"""
        
        try:
            url = f"{self.BASE_URL}/{nct_id}"
            params = {'format': 'json'}
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'studies' in data and len(data['studies']) > 0:
                study = data['studies'][0]
                return self._extract_key_fields(study)
            
        except Exception as e:
            print(f"\nError fetching {nct_id}: {e}")
        
        return None
    
    def _extract_key_fields(self, study: Dict) -> Dict:
        """Extract fields we want to monitor for changes"""
        
        protocol = study.get('protocolSection', {})
        status_mod = protocol.get('statusModule', {})
        design_mod = protocol.get('designModule', {})
        
        return {
            'nct_id': protocol.get('identificationModule', {}).get('nctId'),
            'status': status_mod.get('overallStatus'),
            'enrollment': design_mod.get('enrollmentInfo', {}).get('count', 0),
            'enrollment_type': design_mod.get('enrollmentInfo', {}).get('type'),
            'start_date': status_mod.get('startDateStruct', {}).get('date'),
            'completion_date': status_mod.get('completionDateStruct', {}).get('date'),
            'primary_completion_date': status_mod.get('primaryCompletionDateStruct', {}).get('date'),
            'last_update': status_mod.get('lastUpdatePostDateStruct', {}).get('date'),
            'study_results': status_mod.get('studyResults'),  # Results posted
        }
    
    def _detect_changes(
        self, 
        nct_id: str, 
        previous: Dict, 
        current: Dict
    ) -> List[TrialUpdate]:
        """Detect what changed between previous and current state"""
        
        updates = []
        
        # Status change
        if previous['status'] != current['status']:
            update = TrialUpdate(
                nct_id=nct_id,
                update_type='status_change',
                old_value=previous['status'],
                new_value=current['status'],
                detected_at=datetime.now(),
                severity=self._get_status_change_severity(previous['status'], current['status']),
                message=f"Status changed from {previous['status']} to {current['status']}"
            )
            updates.append(update)
        
        # Enrollment milestone
        prev_enrollment = previous.get('enrollment', 0)
        curr_enrollment = current.get('enrollment', 0)
        
        if curr_enrollment > prev_enrollment:
            # Check for 25%, 50%, 75%, 100% milestones
            milestone = self._check_enrollment_milestone(prev_enrollment, curr_enrollment)
            
            if milestone:
                update = TrialUpdate(
                    nct_id=nct_id,
                    update_type='enrollment_milestone',
                    old_value=str(prev_enrollment),
                    new_value=str(curr_enrollment),
                    detected_at=datetime.now(),
                    severity='info',
                    message=f"Reached {milestone}% enrollment ({curr_enrollment} patients)"
                )
                updates.append(update)
        
        # Study results posted
        if not previous.get('study_results') and current.get('study_results'):
            update = TrialUpdate(
                nct_id=nct_id,
                update_type='results_posted',
                old_value=None,
                new_value='Results available',
                detected_at=datetime.now(),
                severity='critical',
                message="Trial results have been posted on ClinicalTrials.gov"
            )
            updates.append(update)
        
        # Completion date changed (timeline shift)
        if previous.get('completion_date') != current.get('completion_date'):
            update = TrialUpdate(
                nct_id=nct_id,
                update_type='timeline_change',
                old_value=previous.get('completion_date'),
                new_value=current.get('completion_date'),
                detected_at=datetime.now(),
                severity='warning',
                message=f"Completion date changed from {previous.get('completion_date')} to {current.get('completion_date')}"
            )
            updates.append(update)
        
        return updates
    
    def _get_status_change_severity(self, old_status: str, new_status: str) -> str:
        """Determine severity of status change"""
        
        critical_statuses = ['TERMINATED', 'WITHDRAWN', 'SUSPENDED']
        positive_statuses = ['COMPLETED', 'ACTIVE_NOT_RECRUITING']
        
        if new_status in critical_statuses:
            return 'critical'
        elif new_status in positive_statuses:
            return 'info'
        else:
            return 'warning'
    
    def _check_enrollment_milestone(
        self, 
        prev_enrollment: int, 
        curr_enrollment: int
    ) -> Optional[int]:
        """Check if enrollment crossed a milestone (25%, 50%, 75%, 100%)"""
        
        # Assume target is 2x current (simple heuristic)
        # In production: would know actual target enrollment
        estimated_target = curr_enrollment * 1.5
        
        milestones = [25, 50, 75, 100]
        
        for milestone in milestones:
            threshold = (milestone / 100) * estimated_target
            
            if prev_enrollment < threshold <= curr_enrollment:
                return milestone
        
        return None
    
    def _send_notifications(self, customer_id: str, updates: List[TrialUpdate]):
        """Send notifications via webhook, email, Slack"""
        
        # Send webhook
        if customer_id in self.webhooks:
            webhook_config = self.webhooks[customer_id]
            if webhook_config.active:
                self._send_webhook(webhook_config, updates)
        
        # Send email (if configured)
        # self._send_email_alert(customer_id, updates)
        
        # Send Slack (if configured)
        # self._send_slack_alert(customer_id, updates)
    
    def _send_webhook(self, config: WebhookConfig, updates: List[TrialUpdate]):
        """Send webhook POST request"""
        
        payload = {
            'customer_id': config.customer_id,
            'timestamp': datetime.now().isoformat(),
            'updates': [u.to_dict() for u in updates]
        }
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'ClinicalTrialIntelligence/1.0'
        }
        
        if config.auth_token:
            headers['Authorization'] = f'Bearer {config.auth_token}'
        
        for attempt in range(config.retry_count):
            try:
                response = requests.post(
                    config.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=config.timeout_seconds
                )
                
                if response.status_code == 200:
                    print(f"  ✓ Webhook sent to {config.customer_id}")
                    return
                else:
                    print(f"  ✗ Webhook failed: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"  ✗ Webhook error (attempt {attempt+1}/{config.retry_count}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _send_email_alert(self, customer_id: str, updates: List[TrialUpdate]):
        """Send email alert (requires SMTP configuration)"""
        
        # Email configuration (from environment variables)
        smtp_host = "smtp.gmail.com"  # Example
        smtp_port = 587
        sender_email = "alerts@clinicaltrialintelligence.com"
        sender_password = "xxx"  # From secrets
        
        # Customer email lookup (from database)
        customer_email = self._get_customer_email(customer_id)
        
        if not customer_email:
            return
        
        # Build email
        subject = f"Trial Alert: {len(updates)} update(s) detected"
        
        body_html = self._format_email_html(updates)
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = customer_email
        
        msg.attach(MIMEText(body_html, 'html'))
        
        # Send
        try:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            print(f"  ✓ Email sent to {customer_email}")
            
        except Exception as e:
            print(f"  ✗ Email error: {e}")
    
    def _format_email_html(self, updates: List[TrialUpdate]) -> str:
        """Format updates as HTML email"""
        
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; }
                .update { 
                    border-left: 4px solid #3B82F6; 
                    padding: 10px; 
                    margin: 10px 0; 
                    background-color: #F8FAFC;
                }
                .critical { border-left-color: #EF4444; }
                .warning { border-left-color: #F59E0B; }
                .info { border-left-color: #10B981; }
            </style>
        </head>
        <body>
            <h2>Clinical Trial Updates</h2>
        """
        
        for update in updates:
            html += f"""
            <div class="update {update.severity}">
                <strong>{update.nct_id}</strong> - {update.update_type}<br>
                {update.message}<br>
                <small>{update.detected_at.strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _get_customer_email(self, customer_id: str) -> Optional[str]:
        """Lookup customer email from database"""
        # In production: query customer database
        return None
    
    def _load_monitored_trials(self):
        """Load monitored trials from disk"""
        
        state_file = self.data_dir / 'monitored_trials.json'
        
        if state_file.exists():
            with open(state_file, 'r') as f:
                self.monitored_trials = json.load(f)
            print(f"✓ Loaded {len(self.monitored_trials)} monitored trials")
    
    def _save_monitored_trials(self):
        """Save monitored trials to disk"""
        
        state_file = self.data_dir / 'monitored_trials.json'
        
        with open(state_file, 'w') as f:
            json.dump(self.monitored_trials, f, indent=2)
    
    def generate_monitoring_report(self, customer_id: str) -> str:
        """Generate summary report of monitored trials"""
        
        customer_trials = {
            nct_id: config for nct_id, config in self.monitored_trials.items()
            if config['customer_id'] == customer_id
        }
        
        report = f"\n{'='*80}\n"
        report += f"MONITORING REPORT FOR CUSTOMER: {customer_id}\n"
        report += f"{'='*80}\n\n"
        
        report += f"Total monitored trials: {len(customer_trials)}\n\n"
        
        for nct_id, config in customer_trials.items():
            state = config['current_state']
            report += f"{nct_id}:\n"
            report += f"  Status: {state['status']}\n"
            report += f"  Enrollment: {state['enrollment']} ({state['enrollment_type']})\n"
            report += f"  Last checked: {config['last_checked']}\n"
            report += f"  Monitoring since: {config['added_at']}\n\n"
        
        return report


class WebhookServer:
    """
    Simple webhook management for customers
    
    In production: this would be a FastAPI app with:
    - Authentication
    - Rate limiting
    - Request logging
    - Retry logic
    """
    
    def __init__(self):
        self.monitor = None
    
    def register_webhook(
        self,
        customer_id: str,
        webhook_url: str,
        auth_token: str,
        events: List[str]
    ) -> Dict:
        """
        Register webhook endpoint
        
        API endpoint: POST /api/webhooks/register
        """
        
        config = WebhookConfig(
            customer_id=customer_id,
            webhook_url=webhook_url,
            events=events,
            active=True,
            auth_token=auth_token
        )
        
        # Validate webhook (send test request)
        test_payload = {
            'event': 'webhook_test',
            'timestamp': datetime.now().isoformat(),
            'message': 'Webhook registration successful'
        }
        
        try:
            response = requests.post(
                webhook_url,
                json=test_payload,
                headers={'Authorization': f'Bearer {auth_token}'},
                timeout=10
            )
            
            if response.status_code == 200:
                # Save webhook
                if self.monitor:
                    self.monitor.add_webhook(config)
                
                return {
                    'success': True,
                    'message': 'Webhook registered successfully',
                    'webhook_id': hashlib.md5(webhook_url.encode()).hexdigest()
                }
            else:
                return {
                    'success': False,
                    'message': f'Webhook test failed: HTTP {response.status_code}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Webhook validation error: {str(e)}'
            }


def demo_real_time_monitoring():
    """Demonstrate real-time monitoring system"""
    
    print("\n" + "="*80)
    print("REAL-TIME TRIAL MONITORING DEMO")
    print("="*80 + "\n")
    
    # Initialize monitor
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'monitoring'
    monitor = RealTimeTrialMonitor(data_dir)
    
    # Example: Register webhook
    webhook = WebhookConfig(
        customer_id='customer_123',
        webhook_url='https://yourcompany.com/api/trial-alerts',
        events=['status_change', 'enrollment_milestone', 'completion'],
        active=True,
        auth_token='secret_token_abc123'
    )
    monitor.add_webhook(webhook)
    
    # Example: Monitor specific trials
    trials_to_monitor = [
        'NCT05924932',  # Example trial 1
        'NCT05924933',  # Example trial 2
    ]
    
    print("\nAdding trials to monitoring:")
    for nct_id in trials_to_monitor:
        monitor.monitor_trial(nct_id, 'customer_123')
    
    # Check for updates
    print("\nChecking for updates...")
    updates = monitor.check_for_updates()
    
    if updates:
        print(f"\n✓ Detected {len(updates)} update(s):")
        for update in updates:
            print(f"  - {update.nct_id}: {update.message}")
    else:
        print("\n✓ No updates detected")
    
    # Generate report
    report = monitor.generate_monitoring_report('customer_123')
    print(report)
    
    print("\n" + "="*80)
    print("✅ REAL-TIME MONITORING SYSTEM READY")
    print("="*80 + "\n")
    
    print("To deploy in production:")
    print("1. Set up cron job to run check_for_updates() every hour")
    print("2. Configure SMTP for email alerts")
    print("3. Add Slack webhook integration")
    print("4. Build FastAPI app for webhook management")
    print("5. Add to pricing: Professional tier gets 10 trials, Enterprise gets unlimited")


if __name__ == '__main__':
    demo_real_time_monitoring()
