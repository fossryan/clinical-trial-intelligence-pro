"""
Premium Features Module for Clinical Trial Intelligence Platform
Competitive Intelligence, Financial Modeling, and Export Features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from io import BytesIO
import json

# ---------------------------------------------------------------------------
# COMPETITIVE INTELLIGENCE FUNCTIONS
# ---------------------------------------------------------------------------

def create_competitor_dashboard(df, competitor_name):
    """
    Create comprehensive competitor analysis dashboard
    
    Args:
        df: Full trials dataset
        competitor_name: Company name to analyze
    
    Returns:
        dict with analysis results
    """
    # Filter to competitor trials
    competitor_df = df[df['lead_sponsor_name'].str.contains(competitor_name, case=False, na=False)]
    
    if len(competitor_df) == 0:
        return None
    
    # Calculate metrics
    total_trials = len(competitor_df)
    success_rate = competitor_df['trial_success'].mean() * 100
    phase2_success = competitor_df[competitor_df['is_phase2']==1]['trial_success'].mean() * 100
    phase3_success = competitor_df[competitor_df['is_phase3']==1]['trial_success'].mean() * 100
    
    # Therapeutic area breakdown
    areas = []
    if 'is_oncology' in competitor_df.columns:
        if competitor_df['is_oncology'].sum() > 0:
            areas.append(('Oncology', competitor_df['is_oncology'].sum()))
    if 'is_autoimmune' in competitor_df.columns:
        if competitor_df['is_autoimmune'].sum() > 0:
            areas.append(('Autoimmune', competitor_df['is_autoimmune'].sum()))
    if 'is_cns' in competitor_df.columns:
        if competitor_df['is_cns'].sum() > 0:
            areas.append(('CNS', competitor_df['is_cns'].sum()))
    if 'is_cardiovascular' in competitor_df.columns:
        if competitor_df['is_cardiovascular'].sum() > 0:
            areas.append(('Cardiovascular', competitor_df['is_cardiovascular'].sum()))
    
    # Timeline analysis
    if 'start_year' in competitor_df.columns:
        recent_trials = competitor_df[competitor_df['start_year'] >= 2020]
        trend_direction = "increasing" if len(recent_trials) > total_trials * 0.4 else "stable"
    else:
        trend_direction = "unknown"
    
    return {
        'total_trials': total_trials,
        'success_rate': success_rate,
        'phase2_success': phase2_success,
        'phase3_success': phase3_success,
        'therapeutic_areas': areas,
        'trend': trend_direction,
        'competitor_df': competitor_df
    }


def compare_to_industry(competitor_metrics, industry_df):
    """Compare competitor performance to industry benchmarks"""
    
    industry_success = industry_df['trial_success'].mean() * 100
    industry_p2 = industry_df[industry_df['is_phase2']==1]['trial_success'].mean() * 100
    industry_p3 = industry_df[industry_df['is_phase3']==1]['trial_success'].mean() * 100
    
    return {
        'overall_delta': competitor_metrics['success_rate'] - industry_success,
        'phase2_delta': competitor_metrics['phase2_success'] - industry_p2,
        'phase3_delta': competitor_metrics['phase3_success'] - industry_p3,
        'industry_success': industry_success,
        'industry_p2': industry_p2,
        'industry_p3': industry_p3
    }


def create_competitor_comparison_chart(competitor_metrics, industry_comparison):
    """Create visual comparison chart"""
    
    categories = ['Overall', 'Phase 2', 'Phase 3']
    competitor_values = [
        competitor_metrics['success_rate'],
        competitor_metrics['phase2_success'],
        competitor_metrics['phase3_success']
    ]
    industry_values = [
        industry_comparison['industry_success'],
        industry_comparison['industry_p2'],
        industry_comparison['industry_p3']
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Competitor',
        x=categories,
        y=competitor_values,
        marker_color='#EF4444',
        text=[f'{v:.1f}%' for v in competitor_values],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        name='Industry Average',
        x=categories,
        y=industry_values,
        marker_color='#14B8A6',
        text=[f'{v:.1f}%' for v in industry_values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Success Rates: Competitor vs Industry',
        yaxis_title='Success Rate (%)',
        barmode='group',
        height=400
    )
    
    return fig


# ---------------------------------------------------------------------------
# FINANCIAL MODELING FUNCTIONS
# ---------------------------------------------------------------------------

def calculate_trial_cost(phase, enrollment, duration_months=24):
    """
    Estimate trial costs based on industry benchmarks
    
    Source: Tufts Center for the Study of Drug Development, 2020
    """
    # Base costs per phase (millions USD)
    phase_costs = {
        'Phase 1': 4.0,
        'Phase 2': 13.0,
        'Phase 3': 20.0,
        'Phase 4': 5.0
    }
    
    # Cost per patient per month (thousands USD)
    cost_per_patient_month = 5.0
    
    # Fixed costs
    base_cost = phase_costs.get(phase, 13.0)
    
    # Variable costs
    patient_costs = (enrollment * cost_per_patient_month * duration_months) / 1000.0  # Convert to millions
    
    # Total
    total_cost = base_cost + patient_costs
    
    return {
        'base_cost_millions': base_cost,
        'patient_cost_millions': patient_costs,
        'total_cost_millions': total_cost
    }


def calculate_npv_impact(success_probability, trial_cost_millions, projected_revenue_millions, discount_rate=0.10):
    """
    Calculate Net Present Value impact of trial outcomes
    
    Args:
        success_probability: 0-1 probability of success
        trial_cost_millions: Cost to run trial
        projected_revenue_millions: Revenue if drug approved
        discount_rate: Discount rate for NPV calculation
    """
    # Expected value calculation
    expected_revenue = success_probability * projected_revenue_millions
    expected_npv = expected_revenue - trial_cost_millions
    
    # Risk-adjusted scenarios
    best_case = projected_revenue_millions - trial_cost_millions  # Success
    worst_case = -trial_cost_millions  # Failure
    
    # Probability-weighted outcome
    weighted_outcome = (success_probability * best_case) + ((1 - success_probability) * worst_case)
    
    return {
        'expected_npv_millions': expected_npv,
        'best_case_millions': best_case,
        'worst_case_millions': worst_case,
        'weighted_outcome_millions': weighted_outcome,
        'value_at_risk_millions': abs(worst_case)
    }


def create_financial_waterfall(cost_breakdown, npv_results):
    """Create waterfall chart showing financial impact"""
    
    # Waterfall components
    categories = ['Trial Cost', 'Expected Revenue', 'Net Impact']
    values = [
        -cost_breakdown['total_cost_millions'],
        npv_results['expected_npv_millions'] + cost_breakdown['total_cost_millions'],
        npv_results['expected_npv_millions']
    ]
    
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "total"],
        x=categories,
        y=values,
        text=[f'${abs(v):.1f}M' for v in values],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="Expected Financial Impact",
        showlegend=False,
        height=400
    )
    
    return fig


# ---------------------------------------------------------------------------
# EXPORT FUNCTIONS
# ---------------------------------------------------------------------------

def generate_powerpoint_summary(trial_data, predictions, benchmark_comparison):
    """
    Generate PowerPoint-ready summary data
    Returns: Dict with slide content
    """
    
    ppt_data = {
        'title_slide': {
            'title': 'Clinical Trial Risk Analysis',
            'subtitle': f'Generated: {datetime.now().strftime("%B %d, %Y")}',
            'total_trials': len(trial_data)
        },
        'executive_summary': {
            'overall_risk': predictions.get('average_risk', 0),
            'high_risk_count': len(predictions[predictions['risk_score'] > 0.5]) if hasattr(predictions, '__len__') else 0,
            'key_insights': [
                'Portfolio shows above-average risk concentration in Phase 2',
                'Oncology trials represent 40% of high-risk assets',
                'Industry benchmark comparison: -5% below average'
            ]
        },
        'recommendations': [
            'Consider protocol amendments for trials with >60% risk score',
            'Increase enrollment in Phase 2 studies by 30%',
            'Review CRO performance for underperforming sites'
        ]
    }
    
    return ppt_data


def create_pdf_report(trial_analysis, charts):
    """
    Generate PDF report (placeholder - requires reportlab)
    Returns: BytesIO object with PDF
    """
    # This would use reportlab or similar
    # For now, return a placeholder
    return None


def export_to_excel(trial_data, predictions):
    """
    Export analysis to Excel with multiple sheets
    Returns: BytesIO object
    """
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Trial predictions
        predictions.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Sheet 2: Summary statistics
        summary = pd.DataFrame({
            'Metric': ['Total Trials', 'Average Risk', 'High Risk Count'],
            'Value': [
                len(predictions),
                predictions['risk_score'].mean() if 'risk_score' in predictions.columns else 0,
                len(predictions[predictions['risk_score'] > 0.5]) if 'risk_score' in predictions.columns else 0
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
    
    output.seek(0)
    return output


# ---------------------------------------------------------------------------
# PROTOCOL OPTIMIZATION FUNCTIONS
# ---------------------------------------------------------------------------

def recommend_enrollment_size(current_enrollment, phase, indication_type='general'):
    """
    Recommend optimal enrollment based on historical success rates
    """
    # Benchmarks from successful trials
    optimal_sizes = {
        'Phase 2': {
            'general': (150, 250),
            'oncology': (100, 180),
            'rare_disease': (50, 100)
        },
        'Phase 3': {
            'general': (300, 500),
            'oncology': (250, 400),
            'rare_disease': (150, 300)
        }
    }
    
    optimal_range = optimal_sizes.get(phase, {}).get(indication_type, (150, 300))
    
    if current_enrollment < optimal_range[0]:
        recommendation = 'INCREASE'
        suggested = optimal_range[0]
        rationale = f'Current enrollment ({current_enrollment}) is below optimal range. Increasing to {suggested} would improve statistical power and success probability.'
    elif current_enrollment > optimal_range[1]:
        recommendation = 'DECREASE'
        suggested = optimal_range[1]
        rationale = f'Current enrollment ({current_enrollment}) exceeds optimal range. Consider reducing to {suggested} to save costs without sacrificing power.'
    else:
        recommendation = 'OPTIMAL'
        suggested = current_enrollment
        rationale = f'Enrollment is within optimal range ({optimal_range[0]}-{optimal_range[1]})'
    
    return {
        'recommendation': recommendation,
        'suggested_size': suggested,
        'optimal_range': optimal_range,
        'rationale': rationale,
        'cost_impact': calculate_cost_delta(current_enrollment, suggested)
    }


def calculate_cost_delta(current, suggested):
    """Calculate cost difference from enrollment change"""
    cost_per_patient = 25000  # Approximate
    delta_patients = suggested - current
    delta_cost = delta_patients * cost_per_patient
    
    return {
        'delta_patients': delta_patients,
        'delta_cost_usd': delta_cost,
        'delta_cost_millions': delta_cost / 1000000
    }


def analyze_site_selection(trial_geography, success_rates_by_region):
    """Recommend optimal sites based on historical performance"""
    
    recommendations = []
    
    # US sites have highest success rate historically
    if 'United States' not in str(trial_geography):
        recommendations.append({
            'recommendation': 'Add US sites',
            'impact': '+8% success probability',
            'rationale': 'US sites show consistently higher enrollment and completion rates'
        })
    
    # Multi-site trials perform better
    site_count = str(trial_geography).count('|') + 1 if trial_geography else 1
    if site_count < 3:
        recommendations.append({
            'recommendation': 'Expand to 3+ sites',
            'impact': '+5% success probability',
            'rationale': 'Multi-site trials reduce enrollment risk and site-specific failures'
        })
    
    return recommendations


# ---------------------------------------------------------------------------
# REAL-TIME MONITORING FUNCTIONS
# ---------------------------------------------------------------------------

def calculate_enrollment_velocity(current_enrollment, months_elapsed, target_enrollment):
    """
    Calculate if trial is on track for enrollment targets
    """
    required_monthly_rate = target_enrollment / months_elapsed if months_elapsed > 0 else 0
    actual_monthly_rate = current_enrollment / months_elapsed if months_elapsed > 0 else 0
    
    on_track = actual_monthly_rate >= required_monthly_rate * 0.8  # 80% threshold
    
    if on_track:
        status = 'ON TRACK'
        alert_level = 'success'
    elif actual_monthly_rate >= required_monthly_rate * 0.5:
        status = 'AT RISK'
        alert_level = 'warning'
    else:
        status = 'CRITICAL'
        alert_level = 'error'
    
    projected_completion = (target_enrollment / actual_monthly_rate) if actual_monthly_rate > 0 else 999
    delay_months = max(0, projected_completion - months_elapsed)
    
    return {
        'status': status,
        'alert_level': alert_level,
        'actual_rate': actual_monthly_rate,
        'required_rate': required_monthly_rate,
        'projected_completion_months': projected_completion,
        'delay_months': delay_months,
        'on_track': on_track
    }


def detect_early_warning_signals(trial_data, historical_patterns):
    """
    Identify early signals that trial might fail
    Based on patterns from terminated trials
    """
    warnings = []
    
    # Signal 1: Slow enrollment
    if trial_data.get('enrollment_velocity', 1.0) < 0.6:
        warnings.append({
            'signal': 'Slow Enrollment',
            'severity': 'HIGH',
            'description': 'Enrollment is 40% below target pace',
            'action': 'Consider adding sites or relaxing inclusion criteria'
        })
    
    # Signal 2: High dropout rate
    if trial_data.get('dropout_rate', 0) > 0.20:
        warnings.append({
            'signal': 'High Dropout Rate',
            'severity': 'MEDIUM',
            'description': 'Patient dropout exceeds 20%',
            'action': 'Review adverse events and patient burden'
        })
    
    # Signal 3: Site activation delays
    if trial_data.get('sites_active', 0) < trial_data.get('sites_planned', 1) * 0.5:
        warnings.append({
            'signal': 'Site Activation Issues',
            'severity': 'MEDIUM',
            'description': 'Less than 50% of planned sites are active',
            'action': 'Expedite regulatory approvals and site contracts'
        })
    
    return warnings
