#!/bin/bash
# Quick Fix - Add Dataset Toggle to Sidebar

echo "🔧 Adding Dataset Size Toggle to Clinical Trial Intelligence Pro"
echo ""

# Check if we're in the right directory
if [ ! -f "src/app/streamlit_app.py" ]; then
    echo "❌ Error: Must run this from the clinical-trial-intelligence-pro-main directory"
    echo "   Current directory: $(pwd)"
    exit 1
fi

# Backup original
cp src/app/streamlit_app.py src/app/streamlit_app.py.backup
echo "✅ Backed up original streamlit_app.py"

# Check if checkbox already exists
if grep -q "Load all trials" src/app/streamlit_app.py; then
    echo "✅ Checkbox code already present in file"
    echo ""
    echo "The code is correct. Issue might be:"
    echo "1. You're running an old version of the file"
    echo "2. Streamlit cache needs clearing"
    echo "3. You need to restart the Streamlit app"
    echo ""
    echo "Try this:"
    echo "  1. Stop your Streamlit app (Ctrl+C)"
    echo "  2. Clear Streamlit cache: rm -rf ~/.streamlit/cache"
    echo "  3. Restart: streamlit run src/app/streamlit_app.py"
else
    echo "❌ Checkbox code NOT found - applying fix..."
    
    # This should not happen if using the fixed file, but just in case:
    python3 << 'PYTHON_SCRIPT'
import re

with open('src/app/streamlit_app.py', 'r') as f:
    content = f.read()

# Find the main() function and insert the checkbox before df = load_data()
pattern = r'(def main\(\):.*?st\.sidebar\.markdown\("---"\))'
replacement = r'''\1
    
    # Data loading option
    st.sidebar.markdown("### 📊 Dataset Options")
    include_all_trials = st.sidebar.checkbox(
        "Load all trials (8,471)",
        value=False,
        help="Include trials without outcome data. Useful for showing full dataset size. Uncheck to only use trials with known outcomes (5,745) for training."
    )
    st.sidebar.markdown("---")
    '''

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Update the load_data call
content = content.replace('df = load_data()', 'df = load_data(include_all=include_all_trials)')

with open('src/app/streamlit_app.py', 'w') as f:
    f.write(content)

print("✅ Applied checkbox fix")
PYTHON_SCRIPT

fi

echo ""
echo "🧪 Verifying installation..."
if grep -q "include_all_trials = st.sidebar.checkbox" src/app/streamlit_app.py; then
    echo "✅ Checkbox code verified"
else
    echo "❌ Verification failed"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ FIX APPLIED SUCCESSFULLY!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📌 IMPORTANT: You MUST restart Streamlit to see changes"
echo ""
echo "Steps:"
echo "  1. Stop Streamlit app (press Ctrl+C in terminal)"
echo "  2. Restart: streamlit run src/app/streamlit_app.py"
echo ""
echo "You should now see in the sidebar:"
echo "  📊 Dataset Options"
echo "  ☐ Load all trials (8,471)"
echo ""
echo "✅ Check the box → Shows 8,471 trials"
echo "☐ Uncheck the box → Shows 5,745 trials"
echo ""
