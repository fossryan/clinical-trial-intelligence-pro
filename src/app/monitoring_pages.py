"""
Monitoring Pages Module
Enterprise real-time monitoring and site intelligence features
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import monitoring modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from monitoring.real_time_monitor import RealTimeTrialMonitor
    from site_intelligence.site_engine import SiteIntelligenceEngine
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import monitoring modules: {e}")
    MODULES_AVAILABLE = False


def render_real_time_monitoring_page(df=None):
    """Render the real-time monitoring dashboard"""
    st.title("🔴 Real-Time Trial Monitoring")
    
    if not MODULES_AVAILABLE:
        st.error("Monitoring modules not available. Please check installation.")
        return
    
    st.markdown("""
    **Enterprise Feature**: Live trial tracking, enrollment velocity monitoring, and automated risk alerts.
    """)
    
    monitor = RealTimeTrialMonitor()
    
    # Check if dataframe was passed from main app
    if df is not None and not df.empty:
        st.success(f"✓ Using loaded trial data ({len(df)} trials)")
        trials_df = df
        skip_fetch = True
    else:
        skip_fetch = False
        trials_df = None
    
    # Fetch controls (only if not using passed df)
    if not skip_fetch:
        col1, col2 = st.columns([3, 1])
        with col1:
            sponsor_filter = st.text_input("Filter by Sponsor (optional)", "")
        with col2:
            if st.button("Refresh Data", type="primary"):
                st.rerun()
        
        # Fetch active trials
        with st.spinner("Fetching active trials from ClinicalTrials.gov..."):
            if sponsor_filter:
                trials_df = monitor.fetch_active_trials(sponsor=sponsor_filter)
            else:
                trials_df = monitor.fetch_active_trials()
    
    if trials_df.empty:
        st.warning("No active trials found.")
        return
    
    st.success(f"✓ Monitoring {len(trials_df)} active trials")
    
    # Generate risk report
    with st.spinner("Analyzing trial portfolio..."):
        report = monitor.generate_risk_report(trials_df)
    
    # Portfolio Overview
    st.subheader("📊 Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trials", report['portfolio_overview']['total_trials'])
    with col2:
        st.metric("On Track", report['portfolio_overview']['on_track_trials'], 
                 delta=f"{report['portfolio_overview']['on_track_trials']/report['portfolio_overview']['total_trials']*100:.0f}%")
    with col3:
        st.metric("At Risk", report['portfolio_overview']['at_risk_trials'],
                 delta=f"-{report['portfolio_overview']['at_risk_trials']/report['portfolio_overview']['total_trials']*100:.0f}%",
                 delta_color="inverse")
    with col4:
        st.metric("Avg Velocity", f"{report['portfolio_overview']['avg_velocity_score']:.1f}%")
    
    # Alerts Summary
    st.subheader("🚨 Active Alerts")
    
    alert_col1, alert_col2, alert_col3, alert_col4 = st.columns(4)
    with alert_col1:
        st.metric("Critical", report['portfolio_overview']['critical_alerts'], 
                 delta="Immediate Action" if report['portfolio_overview']['critical_alerts'] > 0 else "None")
    with alert_col2:
        st.metric("High", report['portfolio_overview']['high_alerts'])
    with alert_col3:
        st.metric("Medium", report['portfolio_overview']['medium_alerts'])
    with alert_col4:
        st.metric("Low", report['portfolio_overview']['low_alerts'])
    
    # Alert Details
    if report['alerts_by_severity']['critical']:
        st.error("**Critical Alerts Require Immediate Attention**")
        for alert in report['alerts_by_severity']['critical'][:5]:
            with st.expander(f"🔴 {alert['trial_id']}: {alert['message']}", expanded=True):
                st.write(f"**Type**: {alert['alert_type']}")
                st.write(f"**Metric**: {alert['metric_value']:.2f} (Threshold: {alert['threshold_value']:.2f})")
                st.write(f"**Recommended Action**: {alert['recommended_action']}")
    
    # Trial Details Table
    st.subheader("📋 Trial Details")
    
    # Display trials with metrics
    if 'metrics_df' in report:
        display_df = report['metrics_df'][['trial_id', 'current_enrollment', 'target_enrollment', 
                                           'percent_complete', 'velocity_score', 'on_track']]
        st.dataframe(display_df, use_container_width=True)


def render_site_intelligence_page(df=None):
    """Render the site intelligence dashboard"""
    st.title("🏥 Site Intelligence")
    
    if not MODULES_AVAILABLE:
        st.error("Site intelligence modules not available. Please check installation.")
        return
    
    st.markdown("""
    **Enterprise Feature**: AI-powered site selection, performance tracking, and geographic optimization.
    """)
    
    # Check if dataframe was passed from main app
    if df is not None and not df.empty:
        st.success(f"✓ Using loaded trial data ({len(df)} trials)")
        trials_df = df
        use_uploaded = False
    else:
        st.info("""
        **Note**: This feature requires clinical trial data with site/location information. 
        Upload a dataset or connect to your proprietary trial database.
        """)
        use_uploaded = True
        trials_df = None
    
    # File upload for site data (only if no df passed)
    if use_uploaded:
        uploaded_file = st.file_uploader("Upload trial data with site information (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            trials_df = pd.read_csv(uploaded_file)
            st.success(f"✓ Loaded {len(trials_df)} trials")
    
    if trials_df is not None:
        
        st.success(f"✓ Loaded {len(trials_df)} trials")
        
        # Build site database
        engine = SiteIntelligenceEngine()
        
        with st.spinner("Building site intelligence database..."):
            sites = engine.build_site_database(trials_df)
        
        if not sites:
            st.warning("No sites found in the data. Ensure your data includes facility_name, city, and country columns.")
            return
        
        st.success(f"✓ Analyzed {len(sites)} unique sites")
        
        # Site Selection Tool
        st.subheader("🎯 Site Selection")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            therapeutic_area = st.selectbox("Therapeutic Area", 
                                           ["Oncology", "Cardiology", "CNS", "Autoimmune", "Other"])
        with col2:
            phase = st.selectbox("Phase", ["PHASE1", "PHASE2", "PHASE3", "PHASE4"])
        with col3:
            target_enrollment = st.number_input("Target Enrollment", min_value=10, value=100)
        
        if st.button("Get Site Recommendations", type="primary"):
            with st.spinner("Ranking sites..."):
                recommendations = engine.recommend_sites({
                    'therapeutic_area': therapeutic_area,
                    'phase': phase,
                    'target_enrollment': target_enrollment
                }, num_sites=10)
            
            st.subheader("Top Recommended Sites")
            
            for i, rec in enumerate(recommendations[:5], 1):
                with st.expander(f"#{i} - {rec.site.facility_name} ({rec.site.city}, {rec.site.country})", expanded=(i <= 3)):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Match Score", f"{rec.match_score:.1f}/100")
                    with col2:
                        st.metric("Predicted Rate", f"{rec.predicted_enrollment_rate:.1f} pts/mo")
                    with col3:
                        st.metric("Est. Cost", f"${rec.estimated_cost:,.0f}")
                    
                    st.write("**Strengths:**")
                    for strength in rec.strengths:
                        st.write(f"- {strength}")
                    
                    if rec.concerns:
                        st.write("**Considerations:**")
                        for concern in rec.concerns:
                            st.write(f"- {concern}")
    else:
        if use_uploaded:
            st.info("👆 Upload trial data to begin site analysis")
