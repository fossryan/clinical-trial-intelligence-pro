"""
Premium Pages for Real-Time Monitoring and Site Intelligence
Enterprise-grade UI components for live trial tracking and site selection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from monitoring.real_time_monitor import RealTimeTrialMonitor, EnrollmentMetrics, TrialAlert
from site_intelligence.site_engine import SiteIntelligenceEngine, SiteRecommendation


def render_real_time_monitoring_page(df: pd.DataFrame):
    """
    Real-Time Trial Monitoring Dashboard
    
    Features:
    - Live enrollment tracking
    - Risk alerts
    - Portfolio overview
    - Timeline forecasting
    """
    
    st.markdown('<h1 class="main-header">⚡ Real-Time Trial Monitoring</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Live tracking and risk detection for active trials</p>', unsafe_allow_html=True)
    
    # Initialize monitor
    monitor = RealTimeTrialMonitor()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Portfolio Dashboard",
        "⚠️ Risk Alerts", 
        "📈 Enrollment Tracking",
        "🔮 Timeline Forecasts"
    ])
    
    with tab1:
        render_portfolio_dashboard(df, monitor)
    
    with tab2:
        render_risk_alerts(df, monitor)
    
    with tab3:
        render_enrollment_tracking(df, monitor)
    
    with tab4:
        render_timeline_forecasts(df, monitor)


def render_portfolio_dashboard(df: pd.DataFrame, monitor: RealTimeTrialMonitor):
    """Portfolio overview with key metrics"""
    
    st.markdown("### 📊 Portfolio Health Overview")
    
    # Filter to active/recruiting trials
    active_df = df[df['overall_status'].isin(['RECRUITING', 'ACTIVE_NOT_RECRUITING', 'ENROLLING_BY_INVITATION'])]
    
    if active_df.empty:
        st.info("No active trials found in dataset. Using all trials for demo.")
        active_df = df.head(20)
    
    # Generate portfolio monitoring
    with st.spinner("Analyzing portfolio..."):
        monitoring_results = monitor.monitor_portfolio(active_df)
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    stats = monitoring_results['portfolio_stats']
    
    with col1:
        st.metric(
            "Total Active Trials",
            stats['total_trials'],
            help="Trials currently recruiting or active"
        )
    
    with col2:
        on_track_pct = (stats['on_track_trials'] / stats['total_trials'] * 100) if stats['total_trials'] > 0 else 0
        st.metric(
            "On Track",
            f"{stats['on_track_trials']} ({on_track_pct:.0f}%)",
            delta=f"{stats['on_track_trials'] - stats['at_risk_trials']}" if stats['on_track_trials'] > stats['at_risk_trials'] else None,
            delta_color="normal",
            help="Trials meeting enrollment targets"
        )
    
    with col3:
        st.metric(
            "Avg Velocity Score",
            f"{stats['avg_velocity_score']:.1f}/100",
            delta="Good" if stats['avg_velocity_score'] > 70 else "Needs Attention",
            delta_color="normal" if stats['avg_velocity_score'] > 70 else "inverse",
            help="Average enrollment velocity across portfolio"
        )
    
    with col4:
        st.metric(
            "Total Alerts",
            stats['total_alerts'],
            delta=f"🔴 {stats['critical_alerts']} Critical" if stats['critical_alerts'] > 0 else "✅ No Critical",
            delta_color="inverse" if stats['critical_alerts'] > 0 else "normal",
            help="Active risk alerts requiring attention"
        )
    
    st.markdown("---")
    
    # Alert severity breakdown
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Alert Severity Breakdown")
        
        alert_data = pd.DataFrame({
            'Severity': ['Critical', 'High', 'Medium', 'Low'],
            'Count': [
                stats['critical_alerts'],
                stats['high_alerts'],
                stats['medium_alerts'],
                stats['low_alerts']
            ]
        })
        
        fig = px.bar(
            alert_data,
            x='Severity',
            y='Count',
            color='Severity',
            color_discrete_map={
                'Critical': '#DC2626',
                'High': '#F59E0B',
                'Medium': '#FCD34D',
                'Low': '#10B981'
            },
            title="Alerts by Severity"
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Trial Status Distribution")
        
        metrics_df = monitoring_results['metrics_df']
        status_data = pd.DataFrame({
            'Status': ['On Track', 'At Risk'],
            'Count': [
                stats['on_track_trials'],
                stats['at_risk_trials']
            ]
        })
        
        fig = px.pie(
            status_data,
            values='Count',
            names='Status',
            color='Status',
            color_discrete_map={'On Track': '#10B981', 'At Risk': '#F59E0B'},
            title="Trial Health Status"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Trial details table
    st.markdown("### 📋 Trial Details")
    
    # Create summary DataFrame
    summary_data = []
    for metric in monitoring_results['metrics']:
        trial_data = active_df[active_df['nct_id'] == metric.trial_id].iloc[0]
        summary_data.append({
            'NCT ID': metric.trial_id,
            'Title': trial_data.get('brief_title', 'N/A')[:50] + '...' if len(str(trial_data.get('brief_title', ''))) > 50 else trial_data.get('brief_title', 'N/A'),
            'Enrolled': f"{metric.current_enrollment}/{metric.target_enrollment}",
            'Progress': f"{metric.percent_complete:.1f}%",
            'Velocity': f"{metric.velocity_score:.0f}/100",
            'Rate': f"{metric.enrollment_rate:.1f} pts/wk",
            'Status': '✅ On Track' if metric.on_track else '⚠️ At Risk',
            'Completion': metric.projected_completion.strftime('%Y-%m-%d')
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Color code status
    def color_status(val):
        if '✅' in val:
            return 'background-color: #ECFDF5'
        else:
            return 'background-color: #FEF2F2'
    
    styled_df = summary_df.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True, height=400)


def render_risk_alerts(df: pd.DataFrame, monitor: RealTimeTrialMonitor):
    """Detailed risk alerts view"""
    
    st.markdown("### ⚠️ Active Risk Alerts")
    
    # Filter to active trials
    active_df = df[df['overall_status'].isin(['RECRUITING', 'ACTIVE_NOT_RECRUITING'])]
    
    if active_df.empty:
        active_df = df.head(20)
    
    # Generate monitoring
    with st.spinner("Detecting risks..."):
        monitoring_results = monitor.monitor_portfolio(active_df)
    
    alerts = monitoring_results['alerts']
    
    if not alerts:
        st.success("🎉 No active alerts! All trials are performing well.")
        return
    
    # Filter controls
    col1, col2 = st.columns(2)
    
    with col1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            ['critical', 'high', 'medium', 'low'],
            default=['critical', 'high']
        )
    
    with col2:
        alert_type_filter = st.multiselect(
            "Filter by Type",
            list(set(a.alert_type for a in alerts)),
            default=list(set(a.alert_type for a in alerts))
        )
    
    # Filter alerts
    filtered_alerts = [
        a for a in alerts 
        if a.severity in severity_filter and a.alert_type in alert_type_filter
    ]
    
    st.markdown(f"**Showing {len(filtered_alerts)} of {len(alerts)} alerts**")
    
    # Display alerts by severity
    for severity in ['critical', 'high', 'medium', 'low']:
        severity_alerts = [a for a in filtered_alerts if a.severity == severity]
        
        if not severity_alerts:
            continue
        
        # Severity header with icon
        severity_icons = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }
        
        st.markdown(f"#### {severity_icons[severity]} {severity.upper()} Priority ({len(severity_alerts)} alerts)")
        
        for alert in severity_alerts:
            with st.expander(f"**{alert.trial_id}** - {alert.alert_type.replace('_', ' ').title()}", expanded=severity=='critical'):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Message:** {alert.message}")
                    st.markdown(f"**Recommended Action:** {alert.recommended_action}")
                    st.caption(f"Detected: {alert.timestamp.strftime('%Y-%m-%d %H:%M')}")
                
                with col2:
                    st.metric(
                        "Current Value",
                        f"{alert.metric_value:.1f}",
                        delta=f"{alert.metric_value - alert.threshold_value:.1f} vs threshold",
                        delta_color="inverse"
                    )


def render_enrollment_tracking(df: pd.DataFrame, monitor: RealTimeTrialMonitor):
    """Detailed enrollment tracking and velocity analysis"""
    
    st.markdown("### 📈 Enrollment Velocity Tracking")
    
    # Trial selector
    active_df = df[df['overall_status'].isin(['RECRUITING', 'ACTIVE_NOT_RECRUITING'])]
    
    if active_df.empty:
        active_df = df.head(20)
    
    trial_options = {
        f"{row['nct_id']}: {row.get('brief_title', 'Unknown')[:60]}": row['nct_id']
        for _, row in active_df.iterrows()
    }
    
    selected_trial_display = st.selectbox(
        "Select Trial to Analyze",
        options=list(trial_options.keys())
    )
    
    selected_trial_id = trial_options[selected_trial_display]
    trial_data = active_df[active_df['nct_id'] == selected_trial_id].iloc[0]
    
    # Calculate metrics
    metrics = monitor.calculate_enrollment_metrics(trial_data)
    
    # Display metrics
    st.markdown("#### Current Enrollment Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current / Target",
            f"{metrics.current_enrollment} / {metrics.target_enrollment}",
            f"{metrics.percent_complete:.1f}%"
        )
    
    with col2:
        st.metric(
            "Enrollment Rate",
            f"{metrics.enrollment_rate:.1f} pts/wk",
            f"{metrics.enrollment_rate * 4.33:.1f} pts/mo"
        )
    
    with col3:
        st.metric(
            "Velocity Score",
            f"{metrics.velocity_score:.0f}/100",
            "Good" if metrics.velocity_score > 70 else "Needs Attention",
            delta_color="normal" if metrics.velocity_score > 70 else "inverse"
        )
    
    with col4:
        st.metric(
            "Status",
            "On Track" if metrics.on_track else "At Risk",
            delta_color="normal" if metrics.on_track else "inverse"
        )
    
    # Enrollment progress chart
    st.markdown("#### Enrollment Progress")
    
    # Create progress visualization
    weeks = list(range(metrics.weeks_elapsed + metrics.weeks_remaining + 1))
    
    # Historical actual (simulated)
    actual_enrollment = []
    for w in range(metrics.weeks_elapsed + 1):
        enrolled = int(metrics.current_enrollment * (w / metrics.weeks_elapsed)) if metrics.weeks_elapsed > 0 else 0
        actual_enrollment.append(enrolled)
    
    # Projected future
    for w in range(metrics.weeks_elapsed + 1, len(weeks)):
        weeks_from_now = w - metrics.weeks_elapsed
        projected = metrics.current_enrollment + (metrics.enrollment_rate * weeks_from_now)
        actual_enrollment.append(min(projected, metrics.target_enrollment))
    
    # Target trajectory
    target_trajectory = [int(metrics.target_enrollment * (w / len(weeks))) for w in weeks]
    
    fig = go.Figure()
    
    # Actual/Historical
    fig.add_trace(go.Scatter(
        x=weeks[:metrics.weeks_elapsed + 1],
        y=actual_enrollment[:metrics.weeks_elapsed + 1],
        name='Actual Enrollment',
        line=dict(color='#10B981', width=3),
        mode='lines+markers'
    ))
    
    # Projected
    fig.add_trace(go.Scatter(
        x=weeks[metrics.weeks_elapsed:],
        y=actual_enrollment[metrics.weeks_elapsed:],
        name='Projected Enrollment',
        line=dict(color='#10B981', width=2, dash='dash'),
        mode='lines'
    ))
    
    # Target
    fig.add_trace(go.Scatter(
        x=weeks,
        y=target_trajectory,
        name='Target Trajectory',
        line=dict(color='#3B82F6', width=2, dash='dot'),
        mode='lines'
    ))
    
    # Target line
    fig.add_hline(
        y=metrics.target_enrollment,
        line_dash="dash",
        line_color="red",
        annotation_text="Target Enrollment"
    )
    
    # Current time marker
    fig.add_vline(
        x=metrics.weeks_elapsed,
        line_dash="dot",
        line_color="gray",
        annotation_text="Today"
    )
    
    fig.update_layout(
        title="Enrollment Trajectory",
        xaxis_title="Weeks Since Start",
        yaxis_title="Patients Enrolled",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Timeline analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Timeline Analysis")
        st.write(f"**Weeks Elapsed:** {metrics.weeks_elapsed}")
        st.write(f"**Weeks Remaining (Projected):** {metrics.weeks_remaining}")
        st.write(f"**Projected Completion:** {metrics.projected_completion.strftime('%B %d, %Y')}")
        
        if pd.notna(trial_data.get('completion_date')):
            planned = pd.to_datetime(trial_data['completion_date'])
            delay_days = (metrics.projected_completion - planned).days
            if delay_days > 0:
                st.error(f"⚠️ Projected {delay_days} days behind schedule")
            else:
                st.success(f"✅ On schedule (or {abs(delay_days)} days ahead)")
    
    with col2:
        st.markdown("#### Velocity Analysis")
        
        velocity_status = "Excellent" if metrics.velocity_score > 90 else "Good" if metrics.velocity_score > 70 else "Fair" if metrics.velocity_score > 50 else "Poor"
        st.write(f"**Velocity Status:** {velocity_status}")
        st.write(f"**Velocity Score:** {metrics.velocity_score:.1f}/100")
        
        if metrics.velocity_score < 70:
            st.warning("💡 Consider increasing recruitment efforts")
        else:
            st.success("✅ Enrollment velocity is healthy")


def render_timeline_forecasts(df: pd.DataFrame, monitor: RealTimeTrialMonitor):
    """Timeline forecasting with scenarios"""
    
    st.markdown("### 🔮 Timeline Forecasting")
    st.markdown("Model different enrollment scenarios to predict completion dates")
    
    # Trial selector
    active_df = df[df['overall_status'].isin(['RECRUITING', 'ACTIVE_NOT_RECRUITING'])]
    
    if active_df.empty:
        active_df = df.head(20)
    
    trial_options = {
        f"{row['nct_id']}: {row.get('brief_title', 'Unknown')[:60]}": row['nct_id']
        for _, row in active_df.iterrows()
    }
    
    selected_trial_display = st.selectbox(
        "Select Trial",
        options=list(trial_options.keys()),
        key='forecast_trial'
    )
    
    selected_trial_id = trial_options[selected_trial_display]
    trial_data = active_df[active_df['nct_id'] == selected_trial_id].iloc[0]
    
    # Scenario configuration
    st.markdown("#### Scenario Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scenario1_mult = st.slider(
            "Scenario 1: Rate Multiplier",
            0.5, 2.0, 1.0, 0.1,
            help="1.0 = current pace"
        )
        scenario1_name = st.text_input("Scenario 1 Name", "Current Pace")
    
    with col2:
        scenario2_mult = st.slider(
            "Scenario 2: Rate Multiplier",
            0.5, 2.0, 1.2, 0.1
        )
        scenario2_name = st.text_input("Scenario 2 Name", "Optimistic (+20%)")
    
    with col3:
        scenario3_mult = st.slider(
            "Scenario 3: Rate Multiplier",
            0.5, 2.0, 0.8, 0.1
        )
        scenario3_name = st.text_input("Scenario 3 Name", "Pessimistic (-20%)")
    
    # Generate forecasts
    scenarios = {
        scenario1_name: scenario1_mult,
        scenario2_name: scenario2_mult,
        scenario3_name: scenario3_mult
    }
    
    with st.spinner("Generating forecasts..."):
        forecasts = monitor.forecast_enrollment_timeline(trial_data, scenarios)
    
    # Display results
    st.markdown("#### Forecast Results")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Current Status**")
        status = forecasts['current_status']
        st.write(f"Enrolled: {status['enrolled']} / {status['target']}")
        st.write(f"Remaining: {status['remaining']} patients")
        st.write(f"Progress: {status['percent_complete']:.1f}%")
    
    with col2:
        st.markdown("**Scenario Comparison**")
        
        scenario_data = []
        for name, forecast in forecasts['scenarios'].items():
            if forecast['completion_date']:
                scenario_data.append({
                    'Scenario': name,
                    'Completion Date': forecast['completion_date'].strftime('%Y-%m-%d'),
                    'Weeks to Complete': forecast['weeks_to_complete'],
                    'Pts/Month': forecast['patients_per_month']
                })
        
        scenario_df = pd.DataFrame(scenario_data)
        st.dataframe(scenario_df, use_container_width=True)
    
    # Visualization
    st.markdown("#### Scenario Timeline Comparison")
    
    fig = go.Figure()
    
    colors = ['#10B981', '#3B82F6', '#F59E0B']
    
    for i, (name, forecast) in enumerate(forecasts['scenarios'].items()):
        if forecast['completion_date']:
            fig.add_trace(go.Bar(
                name=name,
                x=[name],
                y=[(forecast['completion_date'] - datetime.now()).days],
                text=f"{forecast['weeks_to_complete']:.0f} weeks",
                textposition='auto',
                marker_color=colors[i % len(colors)]
            ))
    
    fig.update_layout(
        title="Days to Completion by Scenario",
        yaxis_title="Days from Today",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_site_intelligence_page(df: pd.DataFrame):
    """
    Site Intelligence and Selection Dashboard
    
    Features:
    - Site performance rankings
    - Geographic optimization
    - Predictive site selection
    - Competitive analysis
    """
    
    st.markdown('<h1 class="main-header">🏥 Site Intelligence & Selection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered site selection and performance optimization</p>', unsafe_allow_html=True)
    
    # Initialize engine
    engine = SiteIntelligenceEngine()
    
    # Build site database
    with st.spinner("Building site intelligence database..."):
        site_profiles = engine.build_site_database(df)
    
    if not site_profiles:
        st.error("Unable to build site database. Need more complete trial data with site information.")
        return
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Site Recommendations",
        "📊 Site Performance",
        "🗺️ Geographic Analysis",
        "🔍 Competitive Density"
    ])
    
    with tab1:
        render_site_recommendations(df, engine)
    
    with tab2:
        render_site_performance(df, engine, site_profiles)
    
    with tab3:
        render_geographic_analysis(df, engine)
    
    with tab4:
        render_competitive_density(df, engine, site_profiles)


def render_site_recommendations(df: pd.DataFrame, engine: SiteIntelligenceEngine):
    """Generate AI-powered site recommendations"""
    
    st.markdown("### 🎯 Get Site Recommendations")
    st.markdown("Configure your trial requirements to get optimal site recommendations")
    
    # Trial requirements
    col1, col2, col3 = st.columns(3)
    
    with col1:
        therapeutic_area = st.selectbox(
            "Therapeutic Area",
            ['Oncology', 'Autoimmune', 'CNS', 'Cardiovascular', 'Metabolic', 'Respiratory']
        )
    
    with col2:
        phase = st.selectbox(
            "Phase",
            ['PHASE1', 'PHASE2', 'PHASE3', 'PHASE4']
        )
    
    with col3:
        target_enrollment = st.number_input(
            "Target Enrollment",
            min_value=10,
            max_value=1000,
            value=100
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        countries = st.multiselect(
            "Preferred Countries (Optional)",
            ['United States', 'United Kingdom', 'Germany', 'France', 'Canada', 'Australia'],
            default=[]
        )
    
    with col2:
        num_sites = st.slider(
            "Number of Recommendations",
            min_value=5,
            max_value=20,
            value=10
        )
    
    if st.button("Generate Recommendations", type="primary"):
        
        trial_requirements = {
            'therapeutic_area': therapeutic_area,
            'phase': phase,
            'target_enrollment': target_enrollment
        }
        
        with st.spinner("Analyzing sites and generating recommendations..."):
            recommendations = engine.recommend_sites(trial_requirements, num_sites=num_sites)
        
        if not recommendations:
            st.warning("No sites found matching your criteria. Try broadening your search.")
            return
        
        st.success(f"✅ Generated {len(recommendations)} site recommendations")
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"#{i} - {rec.site.facility_name} ({rec.site.city}, {rec.site.country}) - Match: {rec.match_score:.0f}/100", expanded=i<=3):
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Performance Metrics**")
                    st.metric("Match Score", f"{rec.match_score:.0f}/100")
                    st.metric("Success Rate", f"{rec.site.success_rate:.0f}%")
                    st.metric("Performance Score", f"{rec.site.performance_score:.0f}/100")
                
                with col2:
                    st.markdown("**Operational Metrics**")
                    st.metric("Total Trials", rec.site.total_trials)
                    st.metric("Predicted Rate", f"{rec.predicted_enrollment_rate:.1f} pts/mo")
                    st.metric("Estimated Cost", f"${rec.estimated_cost:,.0f}")
                
                with col3:
                    st.markdown("**Status**")
                    st.metric("Active Trials", rec.site.active_trials)
                    st.metric("Confidence", rec.confidence_level.title())
                    
                    confidence_colors = {
                        'high': '🟢',
                        'medium': '🟡',
                        'low': '🔴'
                    }
                    st.write(f"Confidence: {confidence_colors[rec.confidence_level]} {rec.confidence_level.title()}")
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Strengths** ✅")
                    if rec.strengths:
                        for strength in rec.strengths:
                            st.markdown(f"- {strength}")
                    else:
                        st.write("No specific strengths identified")
                
                with col2:
                    st.markdown("**Concerns** ⚠️")
                    if rec.concerns:
                        for concern in rec.concerns:
                            st.markdown(f"- {concern}")
                    else:
                        st.write("No major concerns")
        
        # Export option
        st.markdown("---")
        if st.button("📥 Export Recommendations to CSV"):
            export_df = engine.export_site_rankings(recommendations)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"site_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


def render_site_performance(df: pd.DataFrame, engine: SiteIntelligenceEngine, site_profiles: Dict):
    """View and analyze individual site performance"""
    
    st.markdown("### 📊 Site Performance Analysis")
    
    # Site selector
    site_options = {
        f"{site.facility_name} ({site.city}, {site.country})": site_id
        for site_id, site in site_profiles.items()
    }
    
    if not site_options:
        st.info("No site data available. Build site database first.")
        return
    
    selected_site_display = st.selectbox(
        "Select Site to Analyze",
        options=list(site_options.keys())
    )
    
    selected_site_id = site_options[selected_site_display]
    site = site_profiles[selected_site_id]
    
    # Display site profile
    st.markdown(f"## {site.facility_name}")
    st.markdown(f"**Location:** {site.city}, {site.state} {site.country}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Performance Score", f"{site.performance_score:.0f}/100")
    
    with col2:
        st.metric("Success Rate", f"{site.success_rate:.0f}%")
    
    with col3:
        st.metric("Total Trials", site.total_trials)
    
    with col4:
        st.metric("Active Trials", site.active_trials)
    
    st.markdown("---")
    
    # Detailed metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Trial History")
        
        history_data = pd.DataFrame({
            'Status': ['Completed', 'Active', 'Terminated'],
            'Count': [site.completed_trials, site.active_trials, site.terminated_trials]
        })
        
        fig = px.bar(
            history_data,
            x='Status',
            y='Count',
            color='Status',
            color_discrete_map={
                'Completed': '#10B981',
                'Active': '#3B82F6',
                'Terminated': '#EF4444'
            },
            title="Trial Status Distribution"
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Therapeutic Area Experience")
        
        if site.therapeutic_areas:
            area_data = pd.DataFrame({
                'Area': list(site.therapeutic_areas.keys()),
                'Trials': list(site.therapeutic_areas.values())
            })
            
            fig = px.pie(
                area_data,
                values='Trials',
                names='Area',
                title="Therapeutic Areas"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No therapeutic area data available")
    
    # Operational metrics
    st.markdown("#### Operational Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Avg Enrollment Rate",
            f"{site.avg_enrollment_rate:.1f} pts/mo",
            help="Average patients enrolled per month"
        )
    
    with col2:
        st.metric(
            "Data Quality Score",
            f"{site.data_quality_score:.0f}/100",
            help="Quality of data submitted"
        )
    
    with col3:
        st.metric(
            "Dropout Rate",
            f"{site.dropout_rate*100:.0f}%",
            help="Patient dropout rate",
            delta_color="inverse"
        )


def render_geographic_analysis(df: pd.DataFrame, engine: SiteIntelligenceEngine):
    """Geographic distribution and optimization"""
    
    st.markdown("### 🗺️ Geographic Site Distribution")
    st.info("Geographic analysis requires trial requirements. Configure below:")
    
    # Simple requirements
    col1, col2 = st.columns(2)
    
    with col1:
        therapeutic_area = st.selectbox(
            "Therapeutic Area",
            ['Oncology', 'Autoimmune', 'CNS', 'Cardiovascular'],
            key='geo_area'
        )
    
    with col2:
        phase = st.selectbox(
            "Phase",
            ['PHASE2', 'PHASE3'],
            key='geo_phase'
        )
    
    if st.button("Analyze Geographic Distribution"):
        
        requirements = {
            'therapeutic_area': therapeutic_area,
            'phase': phase
        }
        
        with st.spinner("Analyzing geographic distribution..."):
            geo_analysis = engine.analyze_geographic_distribution(requirements)
        
        if not geo_analysis:
            st.warning("No geographic data available")
            return
        
        # Country summary
        st.markdown("#### Sites by Country")
        
        country_data = []
        for country, stats in geo_analysis.items():
            country_data.append({
                'Country': country,
                'Total Sites': stats['total_sites'],
                'Avg Match Score': f"{stats['avg_score']:.1f}",
                'Avg Enrollment Rate': f"{stats['avg_enrollment_rate']:.1f}"
            })
        
        country_df = pd.DataFrame(country_data).sort_values('Total Sites', ascending=False)
        st.dataframe(country_df, use_container_width=True)
        
        # Visualization
        fig = px.bar(
            country_df,
            x='Country',
            y='Total Sites',
            title="Site Distribution by Country",
            color='Total Sites',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_competitive_density(df: pd.DataFrame, engine: SiteIntelligenceEngine, site_profiles: Dict):
    """Analyze competitive trial density"""
    
    st.markdown("### 🔍 Competitive Trial Density")
    st.markdown("Analyze how many competing trials are active in a region")
    
    st.info("Feature coming soon: Will show competitive trial density by location and therapeutic area")
    
    # Placeholder visualization
    st.markdown("**Competitive Density Score**")
    st.markdown("- Low (<5 trials): 🟢 Good opportunity")
    st.markdown("- Medium (5-10 trials): 🟡 Moderate competition")
    st.markdown("- High (>10 trials): 🔴 High competition")
