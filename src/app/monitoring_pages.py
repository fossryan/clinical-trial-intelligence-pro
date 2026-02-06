"""
Monitoring Pages Module
Enterprise real-time monitoring and site intelligence features
"""

import streamlit as st
import pandas as pd
import sys
import random
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
    
    # ALWAYS fetch live data from API - don't use historical ML training data
    # Historical data lacks the real-time fields needed (current enrollment, start dates, etc.)
    st.info("🔴 Fetching live data from ClinicalTrials.gov API...")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        sponsor_filter = st.text_input("Filter by Sponsor (optional)", "")
    with col2:
        if st.button("Refresh Data", type="primary"):
            st.rerun()
    
    # Fetch active trials from API
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
        st.dataframe(display_df, width="stretch")


def render_site_intelligence_page(df=None):
    """Render the site intelligence dashboard"""
    st.title("🏥 Site Intelligence")
    
    if not MODULES_AVAILABLE:
        st.error("Site intelligence modules not available. Please check installation.")
        return
    
    st.markdown("""
    **Enterprise Feature**: AI-powered site selection, performance tracking, and geographic optimization.
    """)
    
    # IMPORTANT: Site Intelligence needs facility-level data
    # The historical ML training data (df) doesn't have facility_name, city, country columns
    # So we ALWAYS require user to fetch from API or upload CSV
    
    st.warning("""
    ⚠️ **Site Intelligence requires facility-level location data** (facility_name, city, country, state).
    
    The historical ML training dataset does not include individual site/facility details. 
    Please choose a data source below:
    """)
    
    engine = SiteIntelligenceEngine()
    
    data_source = st.radio(
        "**Choose Data Source:**",
        ["🌐 Fetch from ClinicalTrials.gov API", "📁 Upload CSV with site data"],
        horizontal=True,
        help="API will fetch trials with location data. CSV should include columns: facility_name, city, country, state"
    )
    
    trials_df = None
    
    if data_source == "🌐 Fetch from ClinicalTrials.gov API":
        st.info("""
        **What this does**:
        - Fetches up to 500 trials from ClinicalTrials.gov
        - Extracts facility names, cities, states, countries
        - Builds site performance database
        - May take 20-30 seconds
        """)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            fetch_button = st.button("🚀 Fetch Site Data from API", type="primary", width="stretch")
        with col2:
            use_demo = st.button("📊 Use Demo Data", help="Load sample data for testing")
        
        if fetch_button:
            with st.spinner("Fetching trials with location data from ClinicalTrials.gov..."):
                try:
                    # Fetch trials with facility information
                    trials_list = engine.fetch_trials_for_sites(max_trials=500)
                    
                    if trials_list:
                        trials_df = pd.DataFrame(trials_list)
                        st.success(f"✅ Successfully fetched {len(trials_df)} trials with site information!")
                        
                        # Show a preview of what was fetched
                        with st.expander("📊 Preview fetched data"):
                            st.write(f"**Columns**: {', '.join(trials_df.columns.tolist()[:10])}...")
                            st.dataframe(trials_df.head(3))
                    else:
                        st.error("""
                        ❌ **Unable to fetch site data from API**
                        
                        Possible causes:
                        - API temporarily unavailable
                        - Network connectivity issues
                        - Rate limit exceeded
                        
                        **Try**: 
                        1. Wait 1 minute and try again
                        2. Click "Use Demo Data" button
                        3. Or upload a CSV file instead
                        """)
                except Exception as e:
                    st.error(f"""
                    ❌ **Error fetching from API**
                    
                    Error: {str(e)}
                    
                    **Try**:
                    1. Click "Use Demo Data" for testing
                    2. Or upload a CSV file
                    """)
        
        elif use_demo:
            # Create demo data for testing
            st.info("Loading demo data with sample facilities...")
            
            demo_data = []
            facilities = [
                ("Massachusetts General Hospital", "Boston", "MA", "United States"),
                ("Mayo Clinic", "Rochester", "MN", "United States"),
                ("Cleveland Clinic", "Cleveland", "OH", "United States"),
                ("Johns Hopkins Hospital", "Baltimore", "MD", "United States"),
                ("UCLA Medical Center", "Los Angeles", "CA", "United States"),
                ("MD Anderson Cancer Center", "Houston", "TX", "United States"),
                ("Memorial Sloan Kettering", "New York", "NY", "United States"),
                ("Stanford Health Care", "Stanford", "CA", "United States"),
                ("UCSF Medical Center", "San Francisco", "CA", "United States"),
                ("Cedars-Sinai Medical Center", "Los Angeles", "CA", "United States"),
            ]
            
            phases = ["PHASE1", "PHASE2", "PHASE3"]
            therapeutic_areas = ["Oncology", "Cardiology", "CNS", "Autoimmune"]
            
            random.seed(42)
            
            trial_counter = 1
            for facility_name, city, state, country in facilities:
                # Each facility participates in 3-8 trials
                num_trials = random.randint(3, 8)
                for _ in range(num_trials):
                    demo_data.append({
                        'nct_id': f'NCT{trial_counter:08d}',
                        'facility_name': facility_name,
                        'city': city,
                        'state': state,
                        'country': country,
                        'phase': random.choice(phases),
                        'therapeutic_area': random.choice(therapeutic_areas),
                        'enrollment': random.randint(50, 300),
                        'status': random.choice(['COMPLETED', 'RECRUITING', 'ACTIVE_NOT_RECRUITING']),
                    })
                    trial_counter += 1
            
            trials_df = pd.DataFrame(demo_data)
            st.success(f"✅ Loaded demo data with {len(trials_df)} trial-site records from {len(facilities)} facilities")
            st.info("💡 This is sample data for demonstration purposes")
    
    else:  # Upload CSV
        st.info("""
        **CSV Requirements**:
        Your file should include these columns:
        - `facility_name` or `site_name` - Name of the facility/hospital
        - `city` - City name
        - `country` - Country name
        - `state` (optional) - State/province
        - Plus any trial metadata (phase, therapeutic area, etc.)
        """)
        
        uploaded_file = st.file_uploader(
            "Upload trial data with site information (CSV)", 
            type=['csv'],
            help="CSV file with facility_name, city, and country columns"
        )
        
        if uploaded_file is not None:
            trials_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(trials_df)} trials from CSV")
            
            # Validate columns
            required_cols = ['facility_name', 'city', 'country']
            missing_cols = [col for col in required_cols if col not in trials_df.columns]
            
            if missing_cols:
                st.error(f"""
                ❌ **Missing required columns**: {', '.join(missing_cols)}
                
                **Your columns**: {', '.join(trials_df.columns.tolist())}
                
                Please ensure your CSV includes: facility_name, city, country
                """)
                trials_df = None
            else:
                st.success("✅ All required columns present!")
    
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
