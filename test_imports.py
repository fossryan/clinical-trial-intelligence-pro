"""
Simple test app to verify all imports work
Use this to debug deployment issues
"""

import streamlit as st

st.title("Import Test")

# Test each import individually
imports_status = {}

try:
    import pandas as pd
    imports_status['pandas'] = '✅ OK'
except Exception as e:
    imports_status['pandas'] = f'❌ {str(e)}'

try:
    import numpy as np
    imports_status['numpy'] = '✅ OK'
except Exception as e:
    imports_status['numpy'] = f'❌ {str(e)}'

try:
    import plotly
    imports_status['plotly'] = '✅ OK'
except Exception as e:
    imports_status['plotly'] = f'❌ {str(e)}'

try:
    import plotly.express as px
    imports_status['plotly.express'] = '✅ OK'
except Exception as e:
    imports_status['plotly.express'] = f'❌ {str(e)}'

try:
    import plotly.graph_objects as go
    imports_status['plotly.graph_objects'] = '✅ OK'
except Exception as e:
    imports_status['plotly.graph_objects'] = f'❌ {str(e)}'

try:
    import joblib
    imports_status['joblib'] = '✅ OK'
except Exception as e:
    imports_status['joblib'] = f'❌ {str(e)}'

try:
    import sklearn
    imports_status['sklearn'] = '✅ OK'
except Exception as e:
    imports_status['sklearn'] = f'❌ {str(e)}'

try:
    import xgboost
    imports_status['xgboost'] = '✅ OK'
except Exception as e:
    imports_status['xgboost'] = f'❌ {str(e)}'

try:
    import lightgbm
    imports_status['lightgbm'] = '✅ OK'
except Exception as e:
    imports_status['lightgbm'] = f'❌ {str(e)}'

# Display results
st.header("Import Status")
for package, status in imports_status.items():
    st.write(f"**{package}**: {status}")

# Show Python version
import sys
st.write(f"\n**Python Version**: {sys.version}")

# Show installed packages
try:
    import pkg_resources
    st.header("Installed Packages")
    packages = sorted([f"{d.project_name}=={d.version}" for d in pkg_resources.working_set])
    st.text("\n".join(packages))
except:
    pass
