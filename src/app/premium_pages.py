"""
Premium Pages for Clinical Trial Intelligence Platform
Add these pages to your main streamlit_app.py

To integrate: Copy the page functions and add to your main() sidebar menu
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

# Import premium features module
sys.path.append(str(Path(__file__).parent))
from premium_features import *


# ===========================================================================
# PAGE: COMPETITIVE INTELLIGENCE
# ===========================================================================

def render_competitive_intelligence_page(df):
    """Competitive Intelligence Dashboard - Premium Feature"""
    
    st.header("🎯 Competitive Intelligence Dashboard")
    st.markdown("Track competitor trials, success rates, and strategic positioning.")
    
    # Premium feature callout
    st.markdown("""
    <div style="background-color: #FEF3C7; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #F59E0B; margin: 1rem 0;">
        <strong>💎 Premium Feature</strong><br>
        Upgrade to Professional tier ($25K/year) for full competitive intelligence:<br>
        • Track up to 10 competitors<br>
        • Real-time email alerts on new trials<br>
        • Historical trend analysis<br>
        • Strategic positioning reports
    </div>
    """, unsafe_allow_html=True)
    
    # Get unique sponsors for dropdown
    if 'lead_sponsor_name' in df.columns:
        sponsors = sorted(df['lead_sponsor_name'].dropna().unique().tolist())
        
        # Competitor selection
        st.subheader("Select Competitor to Analyze")
        competitor = st.selectbox(
            "Company Name",
            options=[''] + sponsors[:50],  # Show top 50 for demo
            help="Search for pharmaceutical or biotech company"
        )
        
        if competitor:
            # Run competitive analysis
            competitor_analysis = create_competitor_dashboard(df, competitor)
            
            if competitor_analysis:
                # Key metrics row
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Trials", competitor_analysis['total_trials'])
                col2.metric("Success Rate", f"{competitor_analysis['success_rate']:.1f}%")
                col3.metric("Phase 2 Success", f"{competitor_analysis['phase2_success']:.1f}%")
                col4.metric("Phase 3 Success", f"{competitor_analysis['phase3_success']:.1f}%")
                
                # Industry comparison
                st.subheader("vs. Industry Benchmark")
                industry_comp = compare_to_industry(competitor_analysis, df)
                
                # Comparison chart
                comp_chart = create_competitor_comparison_chart(competitor_analysis, industry_comp)
                st.plotly_chart(comp_chart, use_container_width=True)
                
                # Delta metrics
                col1, col2, col3 = st.columns(3)
                
                delta_overall = industry_comp['overall_delta']
                col1.metric(
                    "Overall vs Industry",
                    f"{delta_overall:+.1f}%",
                    delta=f"{delta_overall:.1f}%",
                    delta_color="normal"
                )
                
                delta_p2 = industry_comp['phase2_delta']
                col2.metric(
                    "Phase 2 vs Industry",
                    f"{delta_p2:+.1f}%",
                    delta=f"{delta_p2:.1f}%",
                    delta_color="normal"
                )
                
                delta_p3 = industry_comp['phase3_delta']
                col3.metric(
                    "Phase 3 vs Industry",
                    f"{delta_p3:+.1f}%",
                    delta=f"{delta_p3:.1f}%",
                    delta_color="normal"
                )
                
                # Therapeutic area breakdown
                st.subheader("Therapeutic Area Focus")
                if competitor_analysis['therapeutic_areas']:
                    areas_df = pd.DataFrame(
                        competitor_analysis['therapeutic_areas'],
                        columns=['Area', 'Trial Count']
                    )
                    
                    fig = px.pie(
                        areas_df,
                        values='Trial Count',
                        names='Area',
                        title='Trial Distribution by Therapeutic Area'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Strategic insights
                st.subheader("Strategic Insights")
                
                if delta_overall > 10:
                    st.success(f"✅ {competitor} significantly outperforms industry (+{delta_overall:.1f}%). Strong competitive positioning.")
                elif delta_overall < -10:
                    st.info(f"📊 {competitor} underperforms industry ({delta_overall:.1f}%). Potential acquisition target or partner opportunity.")
                else:
                    st.info(f"📊 {competitor} performs in line with industry ({delta_overall:+.1f}%).")
                
                # Recent activity
                st.subheader("Recent Activity")
                st.markdown(f"**Trend:** {competitor_analysis['trend'].upper()}")
                
                # Download report
                st.subheader("Export Analysis")
                
                # Generate downloadable report - convert numpy types to Python types
                def convert_to_serializable(obj):
                    """Convert numpy/pandas types to JSON-serializable types"""
                    import numpy as np
                    import pandas as pd
                    
                    if isinstance(obj, (np.integer, np.int64)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, pd.DataFrame):
                        return obj.to_dict(orient='records')
                    elif isinstance(obj, dict):
                        return {key: convert_to_serializable(value) for key, value in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_to_serializable(item) for item in obj]
                    else:
                        return obj
                
                # Clean the data
                clean_metrics = {
                    'total_trials': int(competitor_analysis['total_trials']),
                    'success_rate': float(competitor_analysis['success_rate']),
                    'phase2_success': float(competitor_analysis['phase2_success']) if not pd.isna(competitor_analysis['phase2_success']) else None,
                    'phase3_success': float(competitor_analysis['phase3_success']) if not pd.isna(competitor_analysis['phase3_success']) else None,
                    'therapeutic_areas': [(area, int(count)) for area, count in competitor_analysis['therapeutic_areas']],
                    'trend': competitor_analysis['trend']
                }
                
                clean_industry_comp = {
                    'overall_delta': float(industry_comp['overall_delta']),
                    'phase2_delta': float(industry_comp['phase2_delta']) if not pd.isna(industry_comp['phase2_delta']) else None,
                    'phase3_delta': float(industry_comp['phase3_delta']) if not pd.isna(industry_comp['phase3_delta']) else None,
                    'industry_success': float(industry_comp['industry_success']),
                    'industry_p2': float(industry_comp['industry_p2']) if not pd.isna(industry_comp['industry_p2']) else None,
                    'industry_p3': float(industry_comp['industry_p3']) if not pd.isna(industry_comp['industry_p3']) else None
                }
                
                report_data = {
                    'competitor': competitor,
                    'analysis_date': datetime.now().isoformat(),
                    'metrics': clean_metrics,
                    'industry_comparison': clean_industry_comp
                }
                
                st.download_button(
                    label="📥 Download Competitor Report (JSON)",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"{competitor.replace(' ', '_')}_analysis_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            else:
                st.warning(f"No trials found for {competitor}. Try a different spelling or check if they use a different legal name.")
    else:
        st.error("Sponsor data not available in current dataset.")


# ===========================================================================
# PAGE: FINANCIAL IMPACT CALCULATOR
# ===========================================================================

def render_financial_calculator_page():
    """Financial Impact Calculator - Premium Feature"""
    
    st.header("💰 Financial Impact Calculator")
    st.markdown("Convert risk scores to dollar impact and ROI projections.")
    
    # Premium callout
    st.markdown("""
    <div style="background-color: #FEF3C7; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #F59E0B; margin: 1rem 0;">
        <strong>💎 Premium Feature</strong><br>
        Full financial modeling available in Professional tier ($25K/year):<br>
        • Portfolio-level NPV analysis<br>
        • Risk-adjusted valuations<br>
        • Budget optimization recommendations<br>
        • Monte Carlo simulations
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Trial Economics Calculator")
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Trial Parameters")
        phase = st.selectbox("Phase", ["Phase 1", "Phase 2", "Phase 3", "Phase 4"])
        enrollment = st.number_input("Planned Enrollment", min_value=10, max_value=5000, value=200)
        duration_months = st.number_input("Duration (months)", min_value=6, max_value=120, value=24)
        success_prob = st.slider("Success Probability (%)", 0, 100, 60) / 100
    
    with col2:
        st.markdown("### Financial Assumptions")
        projected_revenue = st.number_input(
            "Projected Annual Revenue ($M)",
            min_value=0,
            max_value=10000,
            value=500,
            help="Annual peak sales if drug approved"
        )
        discount_rate = st.slider(
            "Discount Rate (%)",
            0, 20, 10,
            help="Typically 8-12% for biotech"
        ) / 100
    
    if st.button("Calculate Financial Impact", type="primary"):
        # Calculate costs
        cost_breakdown = calculate_trial_cost(phase, enrollment, duration_months)
        
        # Calculate NPV
        npv_results = calculate_npv_impact(
            success_prob,
            cost_breakdown['total_cost_millions'],
            projected_revenue,
            discount_rate
        )
        
        # Display results
        st.markdown("---")
        st.subheader("Financial Analysis Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trial Cost", f"${cost_breakdown['total_cost_millions']:.1f}M")
        col2.metric("Expected NPV", f"${npv_results['expected_npv_millions']:.1f}M")
        col3.metric("Best Case", f"${npv_results['best_case_millions']:.1f}M")
        col4.metric("Value at Risk", f"${npv_results['value_at_risk_millions']:.1f}M")
        
        # Cost breakdown
        st.subheader("Cost Breakdown")
        cost_df = pd.DataFrame({
            'Category': ['Base Cost', 'Patient Costs', 'Total'],
            'Amount ($M)': [
                cost_breakdown['base_cost_millions'],
                cost_breakdown['patient_cost_millions'],
                cost_breakdown['total_cost_millions']
            ]
        })
        
        fig = px.bar(
            cost_df,
            x='Category',
            y='Amount ($M)',
            title='Trial Cost Components',
            text='Amount ($M)'
        )
        fig.update_traces(texttemplate='$%{text:.1f}M', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Waterfall chart
        st.subheader("Expected Value Waterfall")
        waterfall = create_financial_waterfall(cost_breakdown, npv_results)
        st.plotly_chart(waterfall, use_container_width=True)
        
        # Scenario analysis
        st.subheader("Scenario Analysis")
        
        scenarios = pd.DataFrame({
            'Scenario': ['Optimistic (Success)', 'Expected', 'Pessimistic (Failure)'],
            'Probability': [f"{success_prob*100:.0f}%", "—", f"{(1-success_prob)*100:.0f}%"],
            'Outcome ($M)': [
                npv_results['best_case_millions'],
                npv_results['weighted_outcome_millions'],
                npv_results['worst_case_millions']
            ]
        })
        
        st.dataframe(scenarios, use_container_width=True)
        
        # Recommendations
        st.subheader("Financial Recommendations")
        
        if npv_results['expected_npv_millions'] > 0:
            st.success(f"✅ **PROCEED** - Positive expected NPV of ${npv_results['expected_npv_millions']:.1f}M")
        elif npv_results['expected_npv_millions'] > -cost_breakdown['total_cost_millions'] * 0.5:
            st.warning(f"⚠️ **CAUTION** - Marginally negative NPV. Consider protocol optimization or risk mitigation.")
        else:
            st.error(f"❌ **HALT** - Highly negative expected NPV. Recommend not proceeding unless strategic rationale exists.")
        
        # Sensitivity analysis preview
        st.markdown("---")
        st.info("💎 **Premium Feature**: Full sensitivity analysis available in Enterprise tier - model impact of enrollment changes, duration extensions, and success probability ranges.")


# ===========================================================================
# PAGE: PROTOCOL OPTIMIZER
# ===========================================================================

def render_protocol_optimizer_page():
    """AI Protocol Optimizer - Premium Feature"""
    
    st.header("🔬 AI Protocol Optimizer")
    st.markdown("Data-driven recommendations to improve trial design and success probability.")
    
    # Premium callout
    st.markdown("""
    <div style="background-color: #FEF3C7; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #F59E0B; margin: 1rem 0;">
        <strong>💎 Premium Feature</strong><br>
        Full protocol optimization available in Enterprise tier ($75K/year):<br>
        • Custom protocol analysis reports<br>
        • Endpoint selection recommendations<br>
        • Inclusion/exclusion criteria optimization<br>
        • Site selection strategy<br>
        • Comparator arm suggestions
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Enrollment Optimization Demo")
    
    # Input
    col1, col2 = st.columns(2)
    
    with col1:
        current_enrollment = st.number_input(
            "Current Planned Enrollment",
            min_value=10,
            max_value=2000,
            value=150
        )
        phase = st.selectbox("Trial Phase", ["Phase 2", "Phase 3"])
    
    with col2:
        indication = st.selectbox(
            "Indication Type",
            ["general", "oncology", "rare_disease"]
        )
    
    if st.button("Get Enrollment Recommendation"):
        recommendation = recommend_enrollment_size(
            current_enrollment,
            phase,
            indication
        )
        
        st.markdown("---")
        st.subheader("Optimization Results")
        
        # Status indicator
        if recommendation['recommendation'] == 'OPTIMAL':
            st.success("✅ Your enrollment size is optimal!")
        elif recommendation['recommendation'] == 'INCREASE':
            st.warning(f"⬆️ Recommend increasing enrollment to {recommendation['suggested_size']}")
        else:
            st.info(f"⬇️ Consider reducing enrollment to {recommendation['suggested_size']}")
        
        # Details
        col1, col2, col3 = st.columns(3)
        col1.metric("Current", current_enrollment)
        col2.metric("Suggested", recommendation['suggested_size'])
        col3.metric(
            "Cost Impact",
            f"${recommendation['cost_impact']['delta_cost_millions']:.1f}M"
        )
        
        # Rationale
        st.markdown("### Rationale")
        st.info(recommendation['rationale'])
        
        # Optimal range visualization
        st.markdown("### Optimal Range")
        
        optimal_min, optimal_max = recommendation['optimal_range']
        
        fig = go.Figure()
        
        # Optimal range
        fig.add_trace(go.Scatter(
            x=[optimal_min, optimal_max],
            y=[1, 1],
            mode='lines',
            line=dict(color='green', width=20),
            name='Optimal Range',
            showlegend=True
        ))
        
        # Current position
        fig.add_trace(go.Scatter(
            x=[current_enrollment],
            y=[1],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Current',
            showlegend=True
        ))
        
        # Suggested position
        if recommendation['recommendation'] != 'OPTIMAL':
            fig.add_trace(go.Scatter(
                x=[recommendation['suggested_size']],
                y=[1],
                mode='markers',
                marker=dict(size=15, color='blue', symbol='star'),
                name='Suggested',
                showlegend=True
            ))
        
        fig.update_layout(
            yaxis=dict(visible=False),
            xaxis_title="Enrollment Size",
            height=200,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        st.markdown("### Additional Optimization Opportunities")
        st.info("💎 **Premium**: Unlock full protocol analysis including site selection, endpoint optimization, and inclusion criteria recommendations.")


# ===========================================================================
# PAGE: EXPORT CENTER
# ===========================================================================

def render_export_center_page(df, predictions):
    """Export Center for Reports - Available to all tiers"""
    
    st.header("📤 Export Center")
    st.markdown("Download your analysis in multiple formats for presentations and reports.")
    
    st.subheader("Available Exports")
    
    # Excel export
    st.markdown("### 📊 Excel Workbook")
    st.markdown("Complete analysis with multiple worksheets - predictions, summaries, and charts.")
    
    if st.button("Generate Excel Report"):
        if predictions is not None and len(predictions) > 0:
            excel_file = export_to_excel(df, predictions)
            
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_file,
                file_name=f"trial_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.success("✅ Excel report generated successfully!")
        else:
            st.warning("No prediction data available. Run batch prediction first.")
    
    # PowerPoint summary
    st.markdown("---")
    st.markdown("### 📊 PowerPoint Summary")
    st.markdown("Executive summary slides ready for Board presentations.")
    
    st.info("💎 **Premium Feature**: Automated PowerPoint generation available in Professional tier ($25K/year)")
    
    # PDF report
    st.markdown("---")
    st.markdown("### 📄 PDF Report")
    st.markdown("Comprehensive written report with charts and recommendations.")
    
    st.info("💎 **Premium Feature**: Custom PDF reports available in Enterprise tier ($75K/year)")
    
    # API access
    st.markdown("---")
    st.markdown("### 🔌 API Access")
    st.markdown("Programmatic access for integration with internal systems.")
    
    st.info("💎 **Premium Feature**: API access available in Enterprise+ tier ($150K/year)")


# ===========================================================================
# INTEGRATION INSTRUCTIONS
# ===========================================================================

"""
TO ADD THESE PAGES TO YOUR MAIN APP:

1. Save this file as 'premium_pages.py' in src/app/

2. In your main streamlit_app.py, add these imports at the top:
   from premium_pages import (
       render_competitive_intelligence_page,
       render_financial_calculator_page,
       render_protocol_optimizer_page,
       render_export_center_page
   )

3. In your sidebar menu, add these pages:
   page = st.sidebar.selectbox(
       "Navigation",
       ["📊 Overview", "🔮 Trial Predictor", "📁 Batch Upload", 
        "📈 Portfolio Analytics", "🔍 Deep Dive",
        "🎯 Competitive Intelligence",  # NEW
        "💰 Financial Calculator",      # NEW
        "🔬 Protocol Optimizer",        # NEW
        "📤 Export Center",             # NEW
        "📊 Model Performance"]
   )

4. In your main() function, add elif blocks:
   elif page == "🎯 Competitive Intelligence":
       render_competitive_intelligence_page(df)
   elif page == "💰 Financial Calculator":
       render_financial_calculator_page()
   elif page == "🔬 Protocol Optimizer":
       render_protocol_optimizer_page()
   elif page == "📤 Export Center":
       render_export_center_page(df, user_results if 'user_results' in locals() else None)
"""
