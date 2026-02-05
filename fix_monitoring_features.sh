#!/bin/bash
# Fix Monitoring Features - Clinical Trial Intelligence Pro
# Fixes import issues for Real-Time Monitoring and Site Intelligence

set -e

echo "🔧 Fixing Clinical Trial Intelligence Pro monitoring features..."
echo ""

# Fix 1: Rename directory with space
echo "📁 Step 1: Renaming 'site intelligence' to 'site_intelligence'..."
if [ -d "src/site intelligence" ]; then
    mv "src/site intelligence" "src/site_intelligence"
    echo "   ✓ Directory renamed"
else
    echo "   ℹ Directory already renamed or doesn't exist"
fi
echo ""

# Fix 2: Create __init__.py files
echo "📄 Step 2: Creating package initialization files..."
touch src/__init__.py
touch src/monitoring/__init__.py
touch src/site_intelligence/__init__.py
touch src/app/__init__.py
touch src/data_collection/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
echo "   ✓ All __init__.py files created"
echo ""

# Fix 3: Verify imports work
echo "🧪 Step 3: Testing imports..."
python3 << EOF
import sys
from pathlib import Path
sys.path.insert(0, 'src')

try:
    from monitoring.real_time_monitor import RealTimeTrialMonitor
    print('   ✅ RealTimeTrialMonitor imported successfully')
except ImportError as e:
    print(f'   ❌ Failed to import RealTimeTrialMonitor: {e}')
    exit(1)

try:
    from site_intelligence.site_engine import SiteIntelligenceEngine
    print('   ✅ SiteIntelligenceEngine imported successfully')
except ImportError as e:
    print(f'   ❌ Failed to import SiteIntelligenceEngine: {e}')
    exit(1)

print('   ✅ All imports successful!')
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 SUCCESS! Monitoring features are now fixed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Fixed Issues:"
echo "   • Renamed 'site intelligence' directory to 'site_intelligence'"
echo "   • Created all required __init__.py package files"
echo "   • Verified imports are working correctly"
echo ""
echo "🚀 Next Steps:"
echo "   1. Start the application:"
echo "      $ streamlit run src/app/streamlit_app.py"
echo ""
echo "   2. Navigate to these pages to verify functionality:"
echo "      • 🔴 Real-Time Trial Monitoring"
echo "      • 🏥 Site Intelligence"
echo ""
echo "   Both features should now display their interfaces instead of"
echo "   error messages."
echo ""
