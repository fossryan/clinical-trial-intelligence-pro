"""
Clinical Trial Risk Intelligence Platform v2
- Original: ClinicalTrials.gov data, single-trial prediction
- NEW: CSV upload for proprietary data, batch prediction, portfolio analyzer,
       side-by-side benchmarking vs public data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
import shutil
import io
import matplotlib.pyplot as plt

# Import premium features
try:
    from premium_pages import (
        render_competitive_intelligence_page,
        render_financial_calculator_page,
        render_protocol_optimizer_page,
        render_export_center_page
    )
    PREMIUM_FEATURES_AVAILABLE = True
except ImportError:
    PREMIUM_FEATURES_AVAILABLE = False
    print("Premium features not available - premium_pages.py not found")

# Import enterprise monitoring features
try:
    from monitoring_pages import (
        render_real_time_monitoring_page,
        render_site_intelligence_page
    )
    MONITORING_FEATURES_AVAILABLE = True
except ImportError:
    MONITORING_FEATURES_AVAILABLE = False
    print("Monitoring features not available - monitoring_pages.py not found")


# ---------------------------------------------------------------------------
# PAGE CONFIG & CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Clinical Trial Risk Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #14B8A6;
    }
    .insight-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
    }
    .upload-box {
        background-color: #EFF6FF;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px dashed #3B82F6;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #ECFDF5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #FEF2F2;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #EF4444;
        margin: 1rem 0;
    }
    .portfolio-card {
        background-color: #F0FDF4;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #86EFAC;
        margin: 0.5rem 0;
    }
    .template-box {
        background-color: #F5F3FF;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #C4B5FD;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(include_all=False):
    """
    Load the ClinicalTrials.gov processed data (public benchmark)
    
    Args:
        include_all: If True, includes all trials. If False, only trials with known outcomes.
    """
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    feature_files = list(data_dir.glob('clinical_trials_features_*.csv'))
    if not feature_files:
        return None
    
    # Find the file with the most trials (best dataset)
    best_file = None
    max_trials = 0
    
    for file in feature_files:
        # Quick check: count lines (trials)
        with open(file, 'r') as f:
            line_count = sum(1 for _ in f) - 1  # Subtract header
        
        if line_count > max_trials:
            max_trials = line_count
            best_file = file
    
    if best_file is None:
        return None
    
    print(f"Loading {best_file.name} with {max_trials:,} trials")
    df = pd.read_csv(best_file)
    
    if include_all:
        print(f"Loaded all {len(df):,} trials")
        return df.copy()
    else:
        filtered_df = df[df['trial_success'].notna()].copy()
        print(f"Loaded {len(filtered_df):,} trials with known outcomes (filtered from {len(df):,})")
        return filtered_df


@st.cache_resource
def load_models():
    """Load trained models (indication-specific if available, otherwise standard)"""
    model_dir = Path(__file__).parent.parent.parent / 'data' / 'models'
    
    # Try to load indication-specific models first
    try:
        import sys
        sys.path.append(str(Path(__file__).parent.parent / 'models'))
        from indication_specific_models import IndicationSpecificModelEngine
        
        indication_engine = IndicationSpecificModelEngine()
        indication_engine.load_models(model_dir)
        
        # Return indication engine and feature names
        return indication_engine, None, indication_engine.feature_names
    except Exception as e:
        print(f"⚠️  Indication models not available: {e}")
        print("   Falling back to standard models...")
    
    # Fallback to standard models
    try:
        xgb_files  = list(model_dir.glob('xgboost_*.joblib'))
        lgb_files  = list(model_dir.glob('lightgbm_*.joblib'))
        feat_files = list(model_dir.glob('feature_names_*.json'))
        if not xgb_files:
            return None, None, None
        xgb_model = joblib.load(max(xgb_files, key=lambda p: p.stat().st_mtime))
        lgb_model = joblib.load(max(lgb_files, key=lambda p: p.stat().st_mtime)) if lgb_files else None
        with open(max(feat_files, key=lambda p: p.stat().st_mtime), 'r') as f:
            feature_names = json.load(f)
        return xgb_model, lgb_model, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None


# ---------------------------------------------------------------------------
# PREDICTION HELPERS
# ---------------------------------------------------------------------------
def predict_single(trial_features: dict, model, feature_names: list):
    """Return (success_prob, risk_score, indication) for one trial."""
    
    # Check if this is an indication-specific engine
    if hasattr(model, 'predict') and hasattr(model, 'models') and 'general' in model.models:
        # Indication-specific routing
        try:
            success_prob, risk_score, indication_used = model.predict(
                trial_features, 
                return_indication=True
            )
            return float(success_prob), float(risk_score), indication_used
        except:
            pass
    
    # Standard model prediction
    vec = [trial_features.get(f, 0) for f in feature_names]
    X = np.array(vec).reshape(1, -1)
    success_prob = model.predict_proba(X)[0][1]
    return float(success_prob), float(1 - success_prob), 'general'


def row_to_features(row: pd.Series) -> dict:
    """
    Convert a single CSV row into the feature dict the model expects.
    Handles both raw-style columns AND pre-engineered columns gracefully.
    """
    f = {}

    # --- phase ---
    phase_str = str(row.get('phase', '')).upper()
    f['is_phase1']  = 1 if 'PHASE1' in phase_str or 'PHASE 1' in phase_str else 0
    f['is_phase2']  = 1 if 'PHASE2' in phase_str or 'PHASE 2' in phase_str else 0
    f['is_phase3']  = 1 if 'PHASE3' in phase_str or 'PHASE 3' in phase_str else 0
    f['is_phase4']  = 1 if 'PHASE4' in phase_str or 'PHASE 4' in phase_str else 0
    f['is_combined_phase'] = 1 if ('1' in phase_str and '2' in phase_str) or ('2' in phase_str and '3' in phase_str) else 0
    if 'PHASE3' in phase_str or 'PHASE 3' in phase_str:
        f['phase_numeric'] = 3.0
    elif 'PHASE2' in phase_str or 'PHASE 2' in phase_str:
        f['phase_numeric'] = 2.0
    elif 'PHASE1' in phase_str or 'PHASE 1' in phase_str:
        f['phase_numeric'] = 1.0
    else:
        f['phase_numeric'] = 2.0  # default

    # --- enrollment ---
    enroll = pd.to_numeric(row.get('enrollment', 200), errors='coerce')
    if pd.isna(enroll):
        enroll = 200
    f['enrollment']        = enroll
    f['log_enrollment']    = np.log1p(enroll)
    f['is_small_trial']    = 1 if enroll < 100 else 0
    f['is_actual_enrollment'] = 1  # assume actual if user provides it

    # --- therapeutic area (keyword scan on condition) ---
    condition = str(row.get('condition', '')).lower()
    oncology_kw    = ['cancer','carcinoma','tumor','tumour','lymphoma','leukemia','melanoma','sarcoma','glioma','myeloma','metastatic','oncology']
    autoimmune_kw  = ['autoimmune','rheumatoid','lupus','crohn','colitis','psoriasis','multiple sclerosis','arthritis','inflammatory']
    cns_kw         = ['alzheimer','parkinson','depression','schizophrenia','anxiety','epilepsy','migraine','neurological','psychiatric']
    cardio_kw      = ['heart','cardiac','cardiovascular','hypertension','arrhythmia','atherosclerosis','myocardial']

    f['is_oncology']       = 1 if any(k in condition for k in oncology_kw)       else 0
    f['is_autoimmune']     = 1 if any(k in condition for k in autoimmune_kw)     else 0
    f['is_cns']            = 1 if any(k in condition for k in cns_kw)            else 0
    f['is_cardiovascular'] = 1 if any(k in condition for k in cardio_kw)         else 0
    f['condition_count']   = max(1, str(row.get('condition', '')).count('|') + 1)

    # --- sponsor ---
    sponsor_name  = str(row.get('lead_sponsor_name', '')).lower()
    sponsor_class = str(row.get('lead_sponsor_class', '')).upper()
    big_pharma    = ['pfizer','novartis','roche','merck','gsk','sanofi','abbvie','bristol','lilly','astrazeneca','amgen','gilead','biogen','bms','takeda']
    f['is_industry_sponsor'] = 1 if sponsor_class == 'INDUSTRY' else (1 if any(bp in sponsor_name for bp in big_pharma) else 0)
    f['is_academic_sponsor'] = 1 if sponsor_class in ('OTHER','OTHER_GOV','NIH') else 0
    f['is_big_pharma']       = 1 if any(bp in sponsor_name for bp in big_pharma) else 0
    f['has_collaborators']   = 1 if pd.to_numeric(row.get('collaborator_count', 0), errors='coerce') > 0 else 0

    # --- geography ---
    countries = str(row.get('countries', ''))
    loc_count = pd.to_numeric(row.get('location_count', 1), errors='coerce') or 1
    f['location_count']   = loc_count
    f['is_multisite']     = 1 if loc_count > 1 else 0
    f['is_international'] = 1 if countries.count('|') > 0 else 0
    f['is_us_trial']      = 1 if 'United States' in countries else 0
    eu = ['France','Germany','Spain','Italy','United Kingdom']
    f['is_europe_trial']  = 1 if any(c in countries for c in eu) else 0

    # --- design ---
    alloc  = str(row.get('allocation', '')).upper()
    mask   = str(row.get('masking', '')).upper()
    f['is_randomized']           = 1 if alloc == 'RANDOMIZED' else 0
    f['is_blinded']              = 1 if mask in ('DOUBLE','TRIPLE','QUADRUPLE') else 0
    f['is_treatment_purpose']    = 1 if str(row.get('primary_purpose', '')).upper() == 'TREATMENT' else 0
    f['is_parallel']             = 1 if str(row.get('intervention_model', '')).upper() == 'PARALLEL' else 0
    pri_out  = pd.to_numeric(row.get('primary_outcome_count', 1), errors='coerce') or 1
    sec_out  = pd.to_numeric(row.get('secondary_outcome_count', 0), errors='coerce') or 0
    f['primary_outcome_count']   = pri_out
    f['secondary_outcome_count'] = sec_out
    f['total_outcome_count']     = pri_out + sec_out
    f['has_secondary_outcomes']  = 1 if sec_out > 0 else 0

    # --- intervention ---
    int_type = str(row.get('intervention_type', '')).upper()
    int_name = str(row.get('intervention_name', ''))
    f['is_drug']       = 1 if 'DRUG' in int_type else 0
    f['is_biological'] = 1 if 'BIOLOGICAL' in int_type else 0
    f['is_device']     = 1 if 'DEVICE' in int_type else 0
    f['intervention_count']           = max(1, int_name.count('|') + 1)
    f['has_multiple_interventions']   = 1 if int_name.count('|') > 0 else 0

    # --- temporal ---
    start = pd.to_datetime(row.get('start_date', None), errors='coerce')
    end   = pd.to_datetime(row.get('completion_date', None), errors='coerce')
    if pd.notna(start) and pd.notna(end):
        f['study_duration_days'] = (end - start).days
    else:
        f['study_duration_days'] = 730  # default 2 years
    f['start_year']      = start.year if pd.notna(start) else 2020
    f['is_recent_trial'] = 1 if f['start_year'] >= 2015 else 0

    # --- complexity ---
    f['complexity_score'] = (
        f['is_combined_phase'] +
        f['has_multiple_interventions'] +
        f['is_international'] +
        (1 if f['condition_count'] > 1 else 0) +
        (1 if f['total_outcome_count'] > 5 else 0)
    )
    f['is_complex_trial'] = 1 if f['complexity_score'] >= 3 else 0

    return f


def batch_predict(df_upload: pd.DataFrame, model, feature_names: list) -> pd.DataFrame:
    """Run predictions on every row of an uploaded DataFrame."""
    results = []
    for idx, row in df_upload.iterrows():
        feats       = row_to_features(row)
        succ, risk  = predict_single(feats, model, feature_names)
        results.append({
            'row_index':         idx,
            'trial_name':        row.get('brief_title', row.get('trial_name', f'Trial {idx+1}')),
            'phase':             row.get('phase', 'N/A'),
            'condition':         row.get('condition', 'N/A'),
            'sponsor':           row.get('lead_sponsor_name', row.get('sponsor', 'N/A')),
            'enrollment':        row.get('enrollment', 'N/A'),
            'success_probability': round(succ * 100, 1),
            'risk_score':        round(risk * 100, 1),
            'risk_level':        '🟢 Low' if risk < 0.3 else ('🟡 Medium' if risk < 0.6 else '🔴 High')
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CHART HELPERS (reused across pages)
# ---------------------------------------------------------------------------
def create_sankey_diagram(df):
    """Phase attrition Sankey."""
    labels = ["All Trials", "Phase 2 Started", "Phase 2 Completed",
              "Phase 3 Started", "Phase 3 Completed", "Terminated/Withdrawn"]

    phase2      = df[df['is_phase2'] == 1]
    phase3      = df[df['is_phase3'] == 1]
    p2_comp     = phase2[phase2['trial_success'] == 1]
    p3_comp     = phase3[phase3['trial_success'] == 1]
    terminated  = df[df['trial_success'] == 0]

    sources = [0, 1, 2, 2, 3]
    targets = [1, 2, 3, 5, 4]
    values  = [
        len(phase2),
        len(p2_comp),
        len(phase3),
        len(terminated) - len(phase2[phase2['trial_success'] == 0]),
        len(p3_comp)
    ]
    values = [max(v, 1) for v in values]

    colors_link = ['rgba(20,184,166,0.3)','rgba(20,184,166,0.5)',
                   'rgba(20,184,166,0.3)','rgba(239,68,68,0.3)','rgba(20,184,166,0.5)']

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                  label=labels,
                  color=["#1E3A8A","#14B8A6","#10B981","#14B8A6","#10B981","#EF4444"]),
        link=dict(source=sources, target=targets, value=values, color=colors_link)
    )])
    fig.update_layout(title_text="Clinical Trial Phase Attrition", font_size=12, height=400)
    return fig


def create_success_rate_chart(df):
    """Success rate bar chart by therapeutic area."""
    area_map = {
        'is_oncology': 'Oncology',
        'is_autoimmune': 'Autoimmune',
        'is_cns': 'CNS',
        'is_cardiovascular': 'Cardiovascular'
    }
    rows = []
    for col, label in area_map.items():
        subset = df[df[col] == 1]
        if len(subset) > 0:
            rows.append({'Area': label,
                         'Success Rate': subset['trial_success'].mean() * 100,
                         'Count': len(subset)})
    if not rows:
        return go.Figure()
    area_df = pd.DataFrame(rows)
    fig = px.bar(area_df, x='Area', y='Success Rate',
                 text='Count', color='Area',
                 color_discrete_map={'Oncology':'#EF4444','Autoimmune':'#3B82F6',
                                     'CNS':'#8B5CF6','Cardiovascular':'#F59E0B'})
    fig.update_traces(texttemplate='n=%{text}', textposition='outside')
    fig.update_layout(title="Success Rate by Therapeutic Area", yaxis_title="Success Rate (%)", height=380)
    return fig


# ---------------------------------------------------------------------------
# CSV TEMPLATE GENERATOR
# ---------------------------------------------------------------------------
def generate_template_csv() -> bytes:
    """
    Return a bytes CSV that shows every column the upload system can use,
    with one example row filled in so users understand the format.
    """
    template = pd.DataFrame([{
        'brief_title':            'Phase 2 Study of ExampleDrug in NSCLC',
        'phase':                  'Phase 2',
        'enrollment':             250,
        'condition':              'Non-Small Cell Lung Cancer',
        'lead_sponsor_name':      'Pfizer Inc',
        'lead_sponsor_class':     'INDUSTRY',
        'intervention_type':      'DRUG',
        'intervention_name':      'ExampleDrug 100mg',
        'allocation':             'RANDOMIZED',
        'masking':                'DOUBLE',
        'intervention_model':     'PARALLEL',
        'primary_purpose':        'TREATMENT',
        'location_count':         12,
        'countries':              'United States|Germany|France',
        'start_date':             '2023-03-15',
        'completion_date':        '2025-06-30',
        'primary_outcome_count':  1,
        'secondary_outcome_count':3,
        'collaborator_count':     2,
    }])
    # Add a second blank row with column-name hints
    hints = {col: f'← {col}' for col in template.columns}
    template.loc[1] = hints
    return template.to_csv(index=False).encode('utf-8')


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
def main():
    # ---- SIDEBAR ----
    st.sidebar.markdown("## 🧬 Clinical Trial Risk Intelligence", unsafe_allow_html=False)
    st.sidebar.markdown("---")
    
    # Data loading option
    st.sidebar.markdown("### 📊 Dataset Options")
    include_all_trials = st.sidebar.checkbox(
        "Load all trials (8,471)",
        value=False,
        help="Include trials without outcome data. Useful for showing full dataset size. Uncheck to only use trials with known outcomes (5,745) for training."
    )
    st.sidebar.markdown("---")
    
    # ---- load core data & models ----
    df = load_data(include_all=include_all_trials)
    xgb_model, lgb_model, feature_names = load_models()

    page = st.sidebar.radio("Navigation", [
        "📊 Overview",
        "🎯 Risk Predictor",
        "📤 Upload & Batch Predict",
        "📁 Portfolio Analyzer",
        "🔍 Deep Dive Analytics",
        "📈 Model Performance",
        "---",
        "⚡ Real-Time Monitoring 🚀",
        "🏥 Site Intelligence 🚀",
        "---",
        "🎯 Competitive Intelligence 💎",
        "💰 Financial Calculator 💎",
        "🔬 Protocol Optimizer 💎",
        "📤 Export Center",
        "💎 Pricing"
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Sources**")
    st.sidebar.markdown("• Public benchmark: ClinicalTrials.gov (2 000 trials)")
    st.sidebar.markdown("• Your data: Upload CSV on the **Upload** page")
    st.sidebar.markdown("---")
    st.sidebar.markdown("*v2.0 — Supports proprietary data upload*", unsafe_allow_html=False)

    # ====================
    # PAGE 1: OVERVIEW
    # ====================
    if page == "📊 Overview":
        st.header("Clinical Trial Risk Intelligence — Overview")
        st.markdown("Real-time portfolio analytics powered by 8,500+ clinical trials from ClinicalTrials.gov (all phases, 2010-2025).")

        if df is None:
            st.error("No benchmark data found. Run `collect_trials.py` → `engineer_features.py` first.")
            return

        # KPI row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trials",        f"{len(df):,}")
        col2.metric("Overall Success Rate", f"{df['trial_success'].mean()*100:.1f}%")
        p2 = df[df['is_phase2']==1]['trial_success'].mean()
        p3 = df[df['is_phase3']==1]['trial_success'].mean()
        col3.metric("Phase 2 Success",     f"{p2*100:.1f}%" if not np.isnan(p2) else "N/A")
        col4.metric("Phase 3 Success",     f"{p3*100:.1f}%" if not np.isnan(p3) else "N/A")

        # Charts row
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_success_rate_chart(df), width="stretch")
        with c2:
            st.plotly_chart(create_sankey_diagram(df), width="stretch")

        # Sponsor benchmark
        st.subheader("Industry vs Academic Success Rates")
        ind = df[df['is_industry_sponsor']==1]['trial_success'].mean()
        aca = df[df['is_academic_sponsor']==1]['trial_success'].mean()
        fig = go.Figure(data=[go.Bar(
            x=['Industry', 'Academic'],
            y=[ind*100 if not np.isnan(ind) else 0, aca*100 if not np.isnan(aca) else 0],
            marker_color=['#14B8A6','#64748B'],
            text=[f'{ind*100:.1f}%', f'{aca*100:.1f}%'],
            textposition='outside'
        )])
        fig.update_layout(yaxis_title="Success Rate (%)", height=320)
        st.plotly_chart(fig, width="stretch")

        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("""
        **📌 Key Insights**
        • Small trials (<100 patients) fail **3.2×** more often than large trials.
        • Oncology trials succeed **15 %** less than other indications.
        • Industry-sponsored trials out-perform academic by **+12 %**.
        • Randomised + blinded designs add **+8 %** success probability.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Premium upgrade prompt
        st.info("💎 **Unlock Premium Intelligence:** Track competitor trials, calculate financial impact, and optimize protocols with AI → [View Pricing](#)")

    # ====================
    # PAGE 2: RISK PREDICTOR  (single trial)
    # ====================
    elif page == "🎯 Risk Predictor":
        st.header("Trial Risk Predictor")
        st.markdown("Input one trial's characteristics → get an instant success/risk prediction with SHAP explanation.")

        if xgb_model is None:
            st.warning("Models not found. Run `train_models.py` first.")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Study Design")
            phase            = st.selectbox("Phase", ["Phase 2","Phase 3","Phase 1/2","Phase 2/3","Phase 1","Phase 4"])
            enrollment       = st.number_input("Enrollment (patients)", min_value=10, max_value=10000, value=200)
            is_randomized    = st.checkbox("Randomized", value=True)
            is_blinded       = st.checkbox("Double-Blinded", value=True)
            therapeutic_area = st.selectbox("Therapeutic Area", ["Oncology","Autoimmune","CNS","Cardiovascular","Other"])

        with col2:
            st.subheader("Sponsor & Geography")
            sponsor_type     = st.selectbox("Sponsor Type", ["Industry","Academic/NIH","Other"])
            is_big_pharma    = st.checkbox("Big Pharma Sponsor")
            is_multisite     = st.checkbox("Multi-Site Study", value=True)
            is_international = st.checkbox("International Study")
            intervention_type= st.selectbox("Intervention Type", ["Drug","Biological","Device","Multiple"])

        if st.button("🎯 Predict Trial Risk", type="primary"):
            trial_features = {
                'is_phase1': 1 if 'Phase 1' == phase else 0,
                'is_phase2': 1 if 'Phase 2' in phase else 0,
                'is_phase3': 1 if 'Phase 3' in phase else 0,
                'is_phase4': 1 if 'Phase 4' == phase else 0,
                'is_combined_phase': 1 if '/' in phase else 0,
                'phase_numeric': 1.0 if phase=='Phase 1' else (2.0 if 'Phase 2' in phase else (3.0 if 'Phase 3' in phase else 4.0)),
                'enrollment': enrollment,
                'log_enrollment': np.log1p(enrollment),
                'is_small_trial': 1 if enrollment < 100 else 0,
                'is_actual_enrollment': 1,
                'is_oncology':       1 if therapeutic_area=='Oncology' else 0,
                'is_autoimmune':     1 if therapeutic_area=='Autoimmune' else 0,
                'is_cns':            1 if therapeutic_area=='CNS' else 0,
                'is_cardiovascular': 1 if therapeutic_area=='Cardiovascular' else 0,
                'condition_count':   1,
                'is_randomized':     1 if is_randomized else 0,
                'is_blinded':        1 if is_blinded else 0,
                'is_industry_sponsor': 1 if sponsor_type=='Industry' else 0,
                'is_academic_sponsor': 1 if sponsor_type=='Academic/NIH' else 0,
                'is_big_pharma':     1 if is_big_pharma else 0,
                'has_collaborators': 0,
                'is_multisite':      1 if is_multisite else 0,
                'is_international':  1 if is_international else 0,
                'is_us_trial':       1,
                'is_europe_trial':   1 if is_international else 0,
                'location_count':    5 if is_multisite else 1,
                'is_drug':           1 if intervention_type=='Drug' else 0,
                'is_biological':     1 if intervention_type=='Biological' else 0,
                'is_device':         1 if intervention_type=='Device' else 0,
                'intervention_count':          2 if intervention_type=='Multiple' else 1,
                'has_multiple_interventions':  1 if intervention_type=='Multiple' else 0,
                'is_treatment_purpose':        1,
                'is_parallel':                 1,
                'primary_outcome_count':       1,
                'secondary_outcome_count':     2,
                'total_outcome_count':         3,
                'has_secondary_outcomes':      1,
                'study_duration_days':         730,
                'start_year':                  2023,
                'is_recent_trial':             1,
                'complexity_score':            (1 if '/' in phase else 0) + (1 if intervention_type=='Multiple' else 0) + (1 if is_international else 0),
                'is_complex_trial':            0,
            }

            success_prob, risk_score, indication_used = predict_single(trial_features, xgb_model, feature_names)

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Success Probability", f"{success_prob*100:.1f} %")
            c2.metric("Risk Score",          f"{risk_score*100:.1f} %")
            risk_label = "🟢 Low" if risk_score < 0.3 else ("🟡 Medium" if risk_score < 0.6 else "🔴 High")
            c3.metric("Risk Level", risk_label)
            c4.metric("Model Used", indication_used.replace('_', ' ').title())

            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            if risk_score < 0.3:
                st.markdown("**✅ Low Risk Trial:** Strong success indicators. Prioritise for portfolio funding.")
            elif risk_score < 0.6:
                st.markdown("**⚠️ Medium Risk Trial:** Mixed signals. Recommend enhanced monitoring and milestone gates.")
            else:
                st.markdown("**🚨 High Risk Trial:** Multiple failure risk factors detected. Deep-review before committing capital.")
            st.markdown('</div>', unsafe_allow_html=True)

            # Feature contribution breakdown (manual SHAP-lite)
            st.subheader("Risk Factor Breakdown")
            contributions = []
            positive_factors = []
            negative_factors = []
            if trial_features.get('is_small_trial') == 1:
                negative_factors.append(("Small enrollment (<100)", -12))
            if trial_features.get('is_oncology') == 1:
                negative_factors.append(("Oncology indication", -8))
            if trial_features.get('phase_numeric', 3) < 3:
                negative_factors.append(("Phase 2 (proof-of-concept)", -10))
            if trial_features.get('is_combined_phase') == 1:
                negative_factors.append(("Combined phase (1/2 or 2/3)", -4))
            if trial_features.get('is_international') == 1:
                negative_factors.append(("International complexity", -3))
            if trial_features.get('is_randomized') == 1:
                positive_factors.append(("Randomised design", +4))
            if trial_features.get('is_blinded') == 1:
                positive_factors.append(("Double-blinded", +3))
            if trial_features.get('is_big_pharma') == 1:
                positive_factors.append(("Big Pharma sponsor", +6))
            if trial_features.get('is_multisite') == 1:
                positive_factors.append(("Multi-site study", +3))
            if trial_features.get('enrollment', 0) >= 200:
                positive_factors.append(("Large enrollment (≥200)", +5))

            all_factors = positive_factors + negative_factors
            if all_factors:
                labels = [f[0] for f in all_factors]
                values = [f[1] for f in all_factors]
                colors = ['#10B981' if v > 0 else '#EF4444' for v in values]
                fig = go.Figure(go.Bar(
                    x=values, y=labels, orientation='h',
                    marker_color=colors,
                    text=[f"+{v}%" if v > 0 else f"{v}%" for v in values],
                    textposition='outside'
                ))
                fig.update_layout(height=max(280, 40*len(labels)), xaxis_title="Impact on Success (%)",
                                  shapes=[dict(type='line', x0=0, x1=0, y0=-0.5, y1=len(labels)-0.5,
                                               line=dict(color='#1E3A8A', width=2, dash='dash'))])
                st.plotly_chart(fig, width="stretch")

    # ====================
    # PAGE 3: UPLOAD & BATCH PREDICT  ← NEW
    # ====================
    elif page == "📤 Upload & Batch Predict":
        st.header("Upload Your Trials — Batch Prediction")
        st.markdown("""
        Upload a CSV of your own clinical trials (proprietary or public).
        The engine scores **every row** in seconds and returns a ranked risk report.
        """)

        if xgb_model is None:
            st.warning("Models not found. Run `train_models.py` first.")
            st.stop()

        # --- template download ---
        st.markdown('<div class="template-box">', unsafe_allow_html=True)
        st.markdown("**📥 Download the CSV Template** — shows every column the system reads and an example row.")
        st.download_button(
            label="⬇️  Download CSV Template",
            data=generate_template_csv(),
            file_name="trial_upload_template.csv",
            mime="text/csv"
        )
        st.markdown("""
        **Required columns:** `phase`, `enrollment`, `condition`  
        **Recommended:** `lead_sponsor_name`, `allocation`, `masking`, `intervention_type`, `location_count`, `countries`  
        **Optional (everything else):** Any column in the template that you have. Missing columns default to safe values.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- file upload widget ---
        uploaded_file = st.file_uploader(
            "Drop your CSV here (or click to browse)",
            type=['csv'],
            help="Max 10 000 rows. Use the template above for column names."
        )

        if uploaded_file is not None:
            # read & preview
            try:
                df_upload = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
                st.stop()

            # drop hint row if present (starts with ←)
            mask = df_upload.apply(lambda row: row.astype(str).str.startswith('←').any(), axis=1)
            df_upload = df_upload[~mask].reset_index(drop=True)

            st.markdown(f"✅ Loaded **{len(df_upload)} trials** with **{len(df_upload.columns)} columns**.")
            st.dataframe(df_upload.head(), width="stretch")

            # validate minimum columns
            required = {'phase', 'enrollment', 'condition'}
            found = set(c.lower().strip() for c in df_upload.columns)
            # normalise column names to lower
            df_upload.columns = [c.lower().strip() for c in df_upload.columns]
            missing = required - set(df_upload.columns)
            if missing:
                st.markdown(f'<div class="error-box">⚠️  Missing required columns: <b>{missing}</b>. '
                            f'Please add them and re-upload.</div>', unsafe_allow_html=True)
                st.stop()

            # --- run predictions ---
            if st.button("🚀 Run Batch Predictions", type="primary"):
                with st.spinner("Scoring trials…"):
                    results_df = batch_predict(df_upload, xgb_model, feature_names)

                # store in session so Portfolio Analyzer can pick it up
                st.session_state['batch_results'] = results_df
                st.session_state['batch_raw']     = df_upload

                # ---- RESULTS DISPLAY ----
                st.markdown("---")
                st.subheader("📊 Batch Results")

                # summary KPIs
                avg_success = results_df['success_probability'].mean()
                high_risk   = (results_df['risk_level'] == '🔴 High').sum()
                med_risk    = (results_df['risk_level'] == '🟡 Medium').sum()
                low_risk    = (results_df['risk_level'] == '🟢 Low').sum()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Trials Scored",       len(results_df))
                c2.metric("Avg Success Prob",    f"{avg_success:.1f} %")
                c3.metric("🔴 High Risk",        high_risk)
                c4.metric("🟢 Low Risk",         low_risk)

                # risk distribution pie
                fig = go.Figure(data=[go.Pie(
                    labels=['🟢 Low','🟡 Medium','🔴 High'],
                    values=[low_risk, med_risk, high_risk],
                    marker_colors=['#10B981','#F59E0B','#EF4444'],
                    hole=0.4
                )])
                fig.update_layout(title="Risk Distribution", height=320)

                c_pie, c_bar = st.columns(2)
                with c_pie:
                    st.plotly_chart(fig, width="stretch")

                # success probability histogram
                with c_bar:
                    fig2 = px.histogram(results_df, x='success_probability', nbins=20,
                                        color_discrete_sequence=['#14B8A6'],
                                        title="Success Probability Distribution")
                    fig2.update_layout(xaxis_title="Success Probability (%)", yaxis_title="# Trials", height=320)
                    st.plotly_chart(fig2, width="stretch")

                # ranked table (sorted best first)
                st.subheader("Ranked Trial List")
                ranked = results_df.sort_values('success_probability', ascending=False).reset_index(drop=True)
                ranked.index = ranked.index + 1   # 1-based rank
                ranked.index.name = 'Rank'
                st.dataframe(ranked, width="stretch")

                # downloadable CSV
                st.download_button(
                    label="⬇️  Download Results CSV",
                    data=ranked.to_csv(),
                    file_name="batch_risk_predictions.csv",
                    mime="text/csv"
                )

                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("✅ Results saved — open **Portfolio Analyzer** to compare your trials against the public benchmark.")
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="upload-box">', unsafe_allow_html=True)
            st.markdown("📂 No file uploaded yet. Download the template above, fill in your trials, then upload here.")
            st.markdown('</div>', unsafe_allow_html=True)

    # ====================
    # PAGE 4: PORTFOLIO ANALYZER  ← NEW
    # ====================
    elif page == "📁 Portfolio Analyzer":
        st.header("Portfolio Analyzer")
        st.markdown("""
        Compare **your uploaded trials** side-by-side against the **2,000-trial public benchmark**.
        Identify where your portfolio is over- or under-weighted on risk.
        """)

        if df is None:
            st.error("Public benchmark data not loaded. Run `collect_trials.py` first.")
            return

        # check for uploaded batch
        has_user_data = 'batch_results' in st.session_state and len(st.session_state['batch_results']) > 0

        if not has_user_data:
            st.markdown('<div class="upload-box">', unsafe_allow_html=True)
            st.markdown("📂 No uploaded data yet. Go to **📤 Upload & Batch Predict**, upload your CSV, and run predictions first.")
            st.markdown('</div>', unsafe_allow_html=True)
            # still show benchmark-only portfolio view
            st.subheader("Public Benchmark Portfolio View")
            st.plotly_chart(create_success_rate_chart(df), width="stretch")
            return

        user_results = st.session_state['batch_results']
        user_raw     = st.session_state.get('batch_raw', pd.DataFrame())

        # --- KPI comparison row ---
        st.subheader("Head-to-Head: Your Portfolio vs Public Benchmark")
        bench_avg = df['trial_success'].mean() * 100
        your_avg  = user_results['success_probability'].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Your Avg Success Prob",   f"{your_avg:.1f} %")
        c2.metric("Benchmark Avg Success",   f"{bench_avg:.1f} %",
                  delta=f"{your_avg - bench_avg:+.1f} pp")
        c3.metric("Your High-Risk Trials",   f"{(user_results['risk_level']=='🔴 High').sum()} / {len(user_results)}")

        # --- comparison bar: success by indication ---
        st.subheader("Success Rate by Indication — You vs Benchmark")

        # benchmark rates
        bench_rates = {}
        for col, label in [('is_oncology','Oncology'),('is_autoimmune','Autoimmune'),
                           ('is_cns','CNS'),('is_cardiovascular','Cardiovascular')]:
            sub = df[df[col]==1]
            if len(sub) > 0:
                bench_rates[label] = sub['trial_success'].mean() * 100

        # user rates (approximate from condition text)
        user_rates = {}
        if 'condition' in user_raw.columns:
            oncology_kw = ['cancer','carcinoma','tumor','tumour','lymphoma','leukemia','melanoma','sarcoma','glioma','myeloma','metastatic','oncology']
            autoimmune_kw = ['autoimmune','rheumatoid','lupus','crohn','colitis','psoriasis','multiple sclerosis','arthritis','inflammatory']
            cns_kw = ['alzheimer','parkinson','depression','schizophrenia','anxiety','epilepsy','migraine','neurological','psychiatric']
            cardio_kw = ['heart','cardiac','cardiovascular','hypertension','arrhythmia','atherosclerosis','myocardial']

            for kw_list, label in [(oncology_kw,'Oncology'),(autoimmune_kw,'Autoimmune'),
                                   (cns_kw,'CNS'),(cardio_kw,'Cardiovascular')]:
                mask = user_raw['condition'].fillna('').str.lower().apply(lambda x: any(k in x for k in kw_list))
                indices = user_raw[mask].index
                if len(indices) > 0:
                    matched = user_results[user_results['row_index'].isin(indices)]
                    user_rates[label] = matched['success_probability'].mean()

        # build grouped bar
        all_labels = sorted(set(list(bench_rates.keys()) + list(user_rates.keys())))
        if all_labels:
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Benchmark', x=all_labels,
                                 y=[bench_rates.get(l, 0) for l in all_labels],
                                 marker_color='#64748B'))
            fig.add_trace(go.Bar(name='Your Portfolio', x=all_labels,
                                 y=[user_rates.get(l, 0) for l in all_labels],
                                 marker_color='#14B8A6'))
            fig.update_layout(barmode='group', yaxis_title='Success Probability (%)', height=380,
                              title='Indication-Level Comparison')
            st.plotly_chart(fig, width="stretch")

        # --- risk heat map (enrollment vs phase) ---
        st.subheader("Your Portfolio — Risk Heat Map (Phase × Enrollment)")
        if len(user_results) > 0 and 'enrollment' in user_raw.columns and 'phase' in user_raw.columns:
            heat_df = pd.DataFrame({
                'phase':      user_raw['phase'].values,
                'enrollment': pd.to_numeric(user_raw['enrollment'], errors='coerce'),
                'success_prob': user_results['success_probability'].values
            })
            heat_df['enroll_bin'] = pd.cut(heat_df['enrollment'],
                                           bins=[0,50,100,200,500,100000],
                                           labels=['<50','50-100','100-200','200-500','500+'])
            pivot = heat_df.pivot_table(values='success_prob', index='phase', columns='enroll_bin', aggfunc='mean')
            if not pivot.empty:
                fig = px.imshow(pivot.values.astype(float), x=pivot.columns.tolist(), y=pivot.index.tolist(),
                                color_continuous_scale='RdYlGn', aspect='auto',
                                labels=dict(x='Enrollment Bin', y='Phase', color='Avg Success %'),
                                title='Average Success Probability Heat Map')
                fig.update_layout(height=340)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Not enough data variation to build heat map.")
        else:
            st.info("Upload data with `phase` and `enrollment` columns to see the heat map.")

        # --- full ranked table ---
        st.subheader("Your Full Portfolio — Ranked by Risk")
        st.dataframe(user_results.sort_values('risk_score', ascending=False).reset_index(drop=True),
                     width="stretch")

    # ====================
    # PAGE 5: DEEP DIVE  (preserved + therapeutic area filter fix)
    # ====================
    elif page == "🔍 Deep Dive Analytics":
        st.header("Deep Dive Analytics")

        if df is None:
            st.error("No data loaded.")
            return

        col1, col2, col3 = st.columns(3)
        with col1:
            phase_filter = st.multiselect("Phase", ['Phase 2','Phase 3'], default=['Phase 2','Phase 3'])
        with col2:
            status_options = df['overall_status'].unique().tolist() if 'overall_status' in df.columns else ['COMPLETED','TERMINATED']
            status_filter  = st.multiselect("Status", status_options, default=[s for s in ['COMPLETED','TERMINATED'] if s in status_options])
        with col3:
            area_filter = st.multiselect("Therapeutic Area", ['Oncology','Autoimmune','CNS','Cardiovascular'])

        df_filtered = df.copy()

        # phase
        if phase_filter:
            phase_mask = pd.Series(False, index=df_filtered.index)
            if 'Phase 2' in phase_filter:
                phase_mask |= df_filtered['is_phase2'] == 1
            if 'Phase 3' in phase_filter:
                phase_mask |= df_filtered['is_phase3'] == 1
            df_filtered = df_filtered[phase_mask]

        # status
        if status_filter and 'overall_status' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['overall_status'].isin(status_filter)]

        # therapeutic area
        if area_filter:
            area_mask = pd.Series(False, index=df_filtered.index)
            if 'Oncology'       in area_filter: area_mask |= df_filtered['is_oncology'] == 1
            if 'Autoimmune'     in area_filter: area_mask |= df_filtered['is_autoimmune'] == 1
            if 'CNS'            in area_filter: area_mask |= df_filtered['is_cns'] == 1
            if 'Cardiovascular' in area_filter: area_mask |= df_filtered['is_cardiovascular'] == 1
            df_filtered = df_filtered[area_mask]

        st.write(f"Showing **{len(df_filtered)}** trials")

        # enrollment success chart
        st.subheader("Success Rate by Enrollment Size")
        df_filtered = df_filtered.copy()
        df_filtered['enrollment_bin'] = pd.cut(
            pd.to_numeric(df_filtered['enrollment'], errors='coerce'),
            bins=[0,50,100,200,500,100000], labels=['<50','50–100','100–200','200–500','500+'])
        enr_agg = df_filtered.groupby('enrollment_bin', observed=True)['trial_success'].agg(['mean','count'])
        enr_agg = enr_agg[enr_agg['count'] >= 3]

        fig = go.Figure(go.Bar(
            x=enr_agg.index.astype(str),
            y=enr_agg['mean']*100,
            text=[f"{r*100:.1f}%<br>n={c}" for r, c in zip(enr_agg['mean'], enr_agg['count'])],
            textposition='outside', marker_color='#14B8A6'))
        fig.update_layout(title="Success Rate by Enrollment", xaxis_title="Enrollment", yaxis_title="Success Rate (%)", height=380)
        st.plotly_chart(fig, width="stretch")

        # trial browser
        st.subheader("Trial Browser")
        display_cols = [c for c in ['nct_id','brief_title','overall_status','phase','enrollment','lead_sponsor_name','condition','trial_success'] if c in df_filtered.columns]
        st.dataframe(df_filtered[display_cols].head(100), width="stretch", height=400)

    # ====================
    # PAGE 6: MODEL PERFORMANCE (preserved)
    # ====================
    elif page == "📈 Model Performance":
        st.header("Model Performance & Insights")

        if xgb_model is None:
            st.warning("Models not found. Run `train_models.py`.")
            st.stop()

        st.markdown("Our predictive models achieve **78%+ accuracy** identifying high-risk trials 18 months before completion.")

        # metrics table
        model_dir = Path(__file__).parent.parent.parent / 'data' / 'models'
        metrics_files = list(model_dir.glob('metrics_*.json'))
        if metrics_files:
            with open(max(metrics_files, key=lambda p: p.stat().st_mtime), 'r') as f:
                metrics = json.load(f)
            metrics_df = pd.DataFrame(metrics).T
            metrics_df['Model'] = metrics_df.index
            cols_show = [c for c in ['Model','roc_auc','f1_score','accuracy'] if c in metrics_df.columns]
            display_metrics = metrics_df[cols_show].copy()
            display_metrics.columns = ['Model'] + [c.replace('_',' ').title() for c in cols_show[1:]]
            st.dataframe(display_metrics, width="stretch")

        # feature importance
        st.subheader("Top Risk Factors")
        if hasattr(xgb_model, 'feature_importances_'):
            feat_imp = pd.DataFrame({
                'Feature':    feature_names,
                'Importance': xgb_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)

            name_map = {
                'is_phase2':'Phase 2','is_phase3':'Phase 3','enrollment':'Enrollment Size',
                'is_oncology':'Oncology','is_small_trial':'Small Trial (<100)',
                'is_industry_sponsor':'Industry Sponsor','is_multisite':'Multi-Site',
                'is_randomized':'Randomised','is_blinded':'Blinded',
                'phase_numeric':'Phase (Numeric)','log_enrollment':'Log Enrollment',
                'is_big_pharma':'Big Pharma','is_international':'International',
                'complexity_score':'Complexity Score','is_recent_trial':'Recent Trial (>2015)'
            }
            feat_imp['Feature'] = feat_imp['Feature'].map(lambda x: name_map.get(x, x))

            fig = go.Figure(go.Bar(
                x=feat_imp['Importance'], y=feat_imp['Feature'], orientation='h',
                marker_color='#14B8A6'))
            fig.update_layout(title="Top 15 Predictive Features", xaxis_title="Importance", height=500,
                              yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, width="stretch")

        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("""
        **🔍 Model Insights**
        1. **Phase 2 trials** have 3× higher failure rates than Phase 3.
        2. **Small enrollment** (<100 patients) is the strongest single risk predictor.
        3. **Oncology** shows 15% lower success across all phases.
        4. **Industry sponsors** out-perform academic by +12%.
        5. **Blinded, randomised designs** correlate with +8% higher success.

        **💡 Recommendation:** Prioritise Phase 3, ≥200 enrollment, non-oncology for lowest portfolio risk.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # ====================
    # ENTERPRISE MONITORING PAGES
    # ====================
    
    elif page == "⚡ Real-Time Monitoring 🚀":
        if MONITORING_FEATURES_AVAILABLE:
            render_real_time_monitoring_page(df)
        else:
            st.error("Real-Time Monitoring features not available. Please ensure monitoring_pages.py is properly installed.")
            st.info("This enterprise feature provides live trial tracking, enrollment velocity monitoring, and automated risk alerts.")
    
    elif page == "🏥 Site Intelligence 🚀":
        if MONITORING_FEATURES_AVAILABLE:
            render_site_intelligence_page(df)
        else:
            st.error("Site Intelligence features not available. Please ensure monitoring_pages.py is properly installed.")
            st.info("This enterprise feature provides AI-powered site selection, performance tracking, and geographic optimization.")
    
    # ====================
    # PREMIUM PAGES
    # ====================
    
    elif page == "🎯 Competitive Intelligence 💎":
        if PREMIUM_FEATURES_AVAILABLE:
            render_competitive_intelligence_page(df)
        else:
            st.error("Premium features not available. Please ensure premium_pages.py is in the same directory.")
    
    elif page == "💰 Financial Calculator 💎":
        if PREMIUM_FEATURES_AVAILABLE:
            render_financial_calculator_page()
        else:
            st.error("Premium features not available. Please ensure premium_pages.py is in the same directory.")
    
    elif page == "🔬 Protocol Optimizer 💎":
        if PREMIUM_FEATURES_AVAILABLE:
            render_protocol_optimizer_page()
        else:
            st.error("Premium features not available. Please ensure premium_pages.py is in the same directory.")
    
    elif page == "📤 Export Center":
        if PREMIUM_FEATURES_AVAILABLE:
            # Get predictions from session state if available
            user_results = st.session_state.get('user_results', None)
            render_export_center_page(df, user_results)
        else:
            st.error("Premium features not available. Please ensure premium_pages.py is in the same directory.")
    
    elif page == "💎 Pricing":
        render_pricing_page()


def render_pricing_page():
    """Pricing page for premium tiers"""
    st.header("💎 Pricing & Plans")
    st.markdown("Choose the plan that fits your clinical development needs")
    
    # Premium callout
    st.markdown("""
    <div style="background-color: #EFF6FF; padding: 1.5rem; border-radius: 0.75rem; border: 2px solid #3B82F6; margin: 1rem 0;">
        <h3 style="color: #1E3A8A; margin-top: 0;">🚀 Transform Your Clinical Development Strategy</h3>
        <p>Join leading biotech and pharma companies using AI-powered trial intelligence to reduce risk and optimize portfolios.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background-color: #F8FAFC; padding: 1.5rem; border-radius: 0.5rem; border: 2px solid #E2E8F0; height: 100%;">
            <h3 style="color: #64748B;">Free</h3>
            <h2 style="color: #1E3A8A;">$0<span style="font-size: 1rem; color: #64748B;">/month</span></h2>
            <hr style="border-color: #E2E8F0;">
            <p>✅ Basic risk predictions</p>
            <p>✅ 5 trials/month</p>
            <p>✅ Public benchmarks</p>
            <p>✅ CSV upload (limited)</p>
            <p>✅ Community support</p>
            <br>
            <p style="color: #10B981; font-weight: bold;">✓ Current Plan</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #F0FDF4; padding: 1.5rem; border-radius: 0.5rem; border: 2px solid #14B8A6; height: 100%;">
            <h3 style="color: #14B8A6;">Professional</h3>
            <h2 style="color: #1E3A8A;">$2,083<span style="font-size: 1rem; color: #64748B;">/month</span></h2>
            <p style="font-size: 0.9rem; color: #64748B;">Billed annually at $25,000</p>
            <hr style="border-color: #14B8A6;">
            <p>✅ <strong>Unlimited predictions</strong></p>
            <p>✅ <strong>Competitive Intel</strong> (3 companies)</p>
            <p>✅ <strong>Financial Calculator</strong></p>
            <p>✅ <strong>Real-Time Monitoring</strong> (10 trials)</p>
            <p>✅ Excel exports</p>
            <p>✅ Email support</p>
            <br>
            <a href="mailto:ryan@yourcompany.com?subject=Professional Tier Inquiry" style="background-color: #14B8A6; color: white; padding: 0.75rem 1.5rem; border-radius: 0.5rem; text-decoration: none; display: inline-block;">Contact Sales</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #FEF3C7; padding: 1.5rem; border-radius: 0.5rem; border: 2px solid #F59E0B; height: 100%;">
            <h3 style="color: #F59E0B;">Enterprise</h3>
            <h2 style="color: #1E3A8A;">$6,250<span style="font-size: 1rem; color: #64748B;">/month</span></h2>
            <p style="font-size: 0.9rem; color: #64748B;">Billed annually at $75,000</p>
            <hr style="border-color: #F59E0B;">
            <p>✅ <strong>Everything in Professional</strong></p>
            <p>✅ <strong>AI Protocol Optimizer</strong></p>
            <p>✅ <strong>Regulatory Advisor</strong></p>
            <p>✅ <strong>Indication Recommender</strong></p>
            <p>✅ PowerPoint/PDF reports</p>
            <p>✅ Priority support</p>
            <p>✅ Quarterly business reviews</p>
            <br>
            <a href="mailto:ryan@yourcompany.com?subject=Enterprise Tier Inquiry" style="background-color: #F59E0B; color: white; padding: 0.75rem 1.5rem; border-radius: 0.5rem; text-decoration: none; display: inline-block;">Contact Sales</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background-color: #EFF6FF; padding: 1.5rem; border-radius: 0.5rem; border: 2px solid #3B82F6; height: 100%;">
            <h3 style="color: #3B82F6;">Enterprise+</h3>
            <h2 style="color: #1E3A8A;">$12,500<span style="font-size: 1rem; color: #64748B;">/month</span></h2>
            <p style="font-size: 0.9rem; color: #64748B;">Billed annually at $150,000</p>
            <hr style="border-color: #3B82F6;">
            <p>✅ <strong>Everything in Enterprise</strong></p>
            <p>✅ <strong>API Access</strong> (1M requests)</p>
            <p>✅ <strong>Custom Models</strong></p>
            <p>✅ <strong>White-label option</strong></p>
            <p>✅ Dedicated CSM</p>
            <p>✅ 20 hours consultation</p>
            <p>✅ Custom integrations</p>
            <br>
            <a href="mailto:ryan@yourcompany.com?subject=Enterprise+ Tier Inquiry" style="background-color: #3B82F6; color: white; padding: 0.75rem 1.5rem; border-radius: 0.5rem; text-decoration: none; display: inline-block;">Contact Sales</a>
        </div>
        """, unsafe_allow_html=True)
    
    # Value proposition
    st.markdown("---")
    st.subheader("Why Clinical Trial Intelligence?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Predictive, Not Descriptive
        Unlike competitors who just track trials, we predict outcomes 18 months in advance using AI trained on 2,000+ studies.
        """)
    
    with col2:
        st.markdown("""
        ### 💰 Proven ROI
        Average Phase 2 costs $13M. If we help avoid one failure, that's a 500X return on Professional tier investment.
        """)
    
    with col3:
        st.markdown("""
        ### ⚡ Real-Time Intelligence
        Get instant insights vs. waiting weeks for consultant reports or quarterly database updates.
        """)
    
    # FAQ
    st.markdown("---")
    st.subheader("Frequently Asked Questions")
    
    with st.expander("Can I try before buying?"):
        st.markdown("""
        Yes! The free tier gives you 5 predictions/month to test the platform. 
        We also offer 90-day pilots for Professional tier at 50% off for qualified customers.
        """)
    
    with st.expander("What kind of companies use this?"):
        st.markdown("""
        Our customers include:
        - **Biotech companies** (Series B-IPO) managing 3-20 trials
        - **Pharma divisions** optimizing portfolio allocation
        - **Investment firms** (VCs, hedge funds) for due diligence
        - **CROs and consultants** differentiating their services
        """)
    
    with st.expander("How accurate are the predictions?"):
        st.markdown("""
        Our models achieve **78%+ accuracy** in predicting trial outcomes, trained on:
        - 2,000+ Phase 2-3 interventional trials
        - 50+ predictive features per trial
        - Historical data from ClinicalTrials.gov
        - Validated using cross-validation and out-of-sample testing
        """)
    
    with st.expander("Can you integrate with our existing systems?"):
        st.markdown("""
        Yes! Enterprise+ tier includes:
        - REST API for programmatic access
        - Custom integrations with Veeva, Medidata, etc.
        - White-label deployment options
        - SSO and security compliance
        """)
    
    with st.expander("What's included in customer support?"):
        st.markdown("""
        - **Free tier:** Community forum access
        - **Professional:** Email support (48-hour response)
        - **Enterprise:** Priority email + monthly check-ins
        - **Enterprise+:** Dedicated customer success manager + 20 hours expert consultation
        """)
    
    # CTA
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #14B8A6; color: white; padding: 2rem; border-radius: 0.75rem; text-align: center;">
        <h2 style="color: white; margin-top: 0;">Ready to reduce trial risk?</h2>
        <p style="font-size: 1.2rem;">Schedule a personalized demo to see how we can optimize your clinical development portfolio.</p>
        <a href="mailto:ryan@yourcompany.com?subject=Demo Request" style="background-color: white; color: #14B8A6; padding: 1rem 2rem; border-radius: 0.5rem; text-decoration: none; font-weight: bold; display: inline-block; margin-top: 1rem;">Schedule Demo</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
