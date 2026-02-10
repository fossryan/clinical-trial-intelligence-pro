"""
Model Performance Monitoring Dashboard
Real-time monitoring of model health and performance degradation
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime, timedelta


def load_performance_history():
    """Load performance history from file"""
    history_file = Path("data/models/performance_history.json")
    
    if not history_file.exists():
        return None
    
    with open(history_file, 'r') as f:
        return json.load(f)


def render_model_health_dashboard():
    """Render model health monitoring dashboard"""
    
    st.markdown("## 📊 Model Health Dashboard")
    st.markdown("Monitor model performance and detect degradation in real-time")
    
    st.markdown("---")
    
    # Load performance history
    history = load_performance_history()
    
    if not history or 'models' not in history:
        st.warning("⚠️ No performance history available")
        st.info("Models will be monitored after first retraining cycle")
        return
    
    # Model selector
    available_models = list(history['models'].keys())
    
    if not available_models:
        st.warning("⚠️ No model history found")
        return
    
    selected_model = st.selectbox(
        "Select Model",
        available_models,
        index=0 if 'lightgbm' in available_models else 0
    )
    
    model_history = history['models'][selected_model]
    
    if not model_history:
        st.warning(f"⚠️ No history for {selected_model}")
        return
    
    # Convert to DataFrame for easy plotting
    records = []
    for record in model_history:
        records.append({
            'timestamp': datetime.fromisoformat(record['timestamp']),
            'accuracy': record['metrics']['accuracy'],
            'roc_auc': record['metrics']['roc_auc'],
            'f1_score': record['metrics']['f1_score'],
            'precision': record['metrics'].get('precision', 0),
            'recall': record['metrics'].get('recall', 0),
            'dataset_size': record['dataset_size']
        })
    
    df = pd.DataFrame(records)
    
    # Current status
    latest = records[-1]
    baseline = records[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy_change = latest['accuracy'] - baseline['accuracy']
        st.metric(
            "Current Accuracy",
            f"{latest['accuracy']:.2%}",
            f"{accuracy_change:+.2%}",
            delta_color="normal"
        )
    
    with col2:
        roc_change = latest['roc_auc'] - baseline['roc_auc']
        st.metric(
            "Current ROC-AUC",
            f"{latest['roc_auc']:.3f}",
            f"{roc_change:+.3f}",
            delta_color="normal"
        )
    
    with col3:
        f1_change = latest['f1_score'] - baseline['f1_score']
        st.metric(
            "Current F1 Score",
            f"{latest['f1_score']:.3f}",
            f"{f1_change:+.3f}",
            delta_color="normal"
        )
    
    with col4:
        days_old = (datetime.now() - latest['timestamp']).days
        st.metric(
            "Model Age",
            f"{days_old} days",
            "Fresh" if days_old < 30 else "Aging",
            delta_color="inverse" if days_old >= 30 else "normal"
        )
    
    st.markdown("---")
    
    # Performance trends
    st.markdown("### Performance Trends")
    
    tab1, tab2, tab3 = st.tabs(["📈 Metrics Over Time", "📊 Distribution", "🔄 Retraining Events"])
    
    with tab1:
        # Plot all metrics over time
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['accuracy'],
            name='Accuracy',
            line=dict(color='#10b981', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['roc_auc'],
            name='ROC-AUC',
            line=dict(color='#3b82f6', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['f1_score'],
            name='F1 Score',
            line=dict(color='#f59e0b', width=2)
        ))
        
        # Add threshold lines
        fig.add_hline(y=0.75, line_dash="dash", line_color="red", 
                     annotation_text="Min Accuracy Threshold")
        fig.add_hline(y=0.70, line_dash="dash", line_color="red",
                     annotation_text="Min ROC-AUC Threshold")
        
        fig.update_layout(
            title="Model Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Score",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Metric distribution
        col_a, col_b = st.columns(2)
        
        with col_a:
            fig2 = go.Figure(data=[
                go.Bar(
                    x=['Accuracy', 'ROC-AUC', 'F1 Score', 'Precision', 'Recall'],
                    y=[
                        latest['accuracy'],
                        latest['roc_auc'],
                        latest['f1_score'],
                        latest['precision'],
                        latest['recall']
                    ],
                    marker_color=['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899']
                )
            ])
            
            fig2.update_layout(
                title="Current Metrics",
                yaxis_title="Score",
                template='plotly_white',
                height=350
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        with col_b:
            # Dataset size trend
            fig3 = go.Figure(data=[
                go.Scatter(
                    x=df['timestamp'],
                    y=df['dataset_size'],
                    mode='lines+markers',
                    line=dict(color='#6366f1', width=2),
                    marker=dict(size=8)
                )
            ])
            
            fig3.update_layout(
                title="Training Dataset Size",
                xaxis_title="Date",
                yaxis_title="Number of Trials",
                template='plotly_white',
                height=350
            )
            
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        # Retraining events
        if 'retraining_events' in history and history['retraining_events']:
            st.markdown("### Recent Retraining Events")
            
            for event in reversed(history['retraining_events'][-10:]):
                with st.expander(
                    f"🔄 {datetime.fromisoformat(event['timestamp']).strftime('%Y-%m-%d %H:%M')} - {event['model_name']}"
                ):
                    st.markdown(f"**Reason**: {event['reason']}")
                    
                    col_x, col_y = st.columns(2)
                    
                    with col_x:
                        st.markdown("**Before Retraining**")
                        for metric, value in event['old_metrics'].items():
                            st.text(f"{metric}: {value:.4f}")
                    
                    with col_y:
                        st.markdown("**After Retraining**")
                        for metric, value in event['new_metrics'].items():
                            improvement = event['improvement'].get(metric, 0)
                            color = "🟢" if improvement > 0 else "🔴"
                            st.text(f"{metric}: {value:.4f} {color}")
        else:
            st.info("No retraining events recorded yet")
    
    st.markdown("---")
    
    # Health status
    st.markdown("### 🏥 Model Health Status")
    
    # Calculate health score
    accuracy_ok = latest['accuracy'] >= 0.75
    roc_auc_ok = latest['roc_auc'] >= 0.70
    not_degraded = (latest['accuracy'] - baseline['accuracy']) > -0.05
    not_too_old = days_old < 90
    
    health_score = sum([accuracy_ok, roc_auc_ok, not_degraded, not_too_old])
    
    col_h1, col_h2 = st.columns([1, 2])
    
    with col_h1:
        if health_score == 4:
            st.success("✅ Excellent")
            st.markdown("All metrics healthy")
        elif health_score == 3:
            st.success("✅ Good")
            st.markdown("Minor issues detected")
        elif health_score == 2:
            st.warning("⚠️ Fair")
            st.markdown("Retraining recommended")
        else:
            st.error("❌ Poor")
            st.markdown("Immediate retraining needed")
    
    with col_h2:
        checks = {
            "Accuracy above 75%": accuracy_ok,
            "ROC-AUC above 70%": roc_auc_ok,
            "No significant degradation": not_degraded,
            "Model age < 90 days": not_too_old
        }
        
        for check, status in checks.items():
            icon = "✅" if status else "❌"
            st.markdown(f"{icon} {check}")


if __name__ == "__main__":
    # For testing
    render_model_health_dashboard()
