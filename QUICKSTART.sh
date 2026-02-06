#!/bin/bash

# Clinical Trial Intelligence - One-Click Setup
# This script does EVERYTHING for you

set -e  # Exit on any error

echo ""
echo "🧬 Clinical Trial Risk Intelligence Platform"
echo "==========================================="
echo ""
echo "This script will:"
echo "  1. Create virtual environment"
echo "  2. Install all dependencies"
echo "  3. Collect clinical trial data (5-10 min)"
echo "  4. Engineer features"
echo "  5. Train ML models"
echo "  6. Launch dashboard"
echo ""
read -p "Ready to start? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# Check Python version
echo ""
echo "📋 Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "🔧 Creating virtual environment..."
python3 -m venv venv
echo "✅ Virtual environment created"

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --quiet --upgrade pip

# Install dependencies
echo ""
echo "📦 Installing dependencies (this may take 2-3 minutes)..."
pip install --quiet -r requirements.txt
echo "✅ Dependencies installed"

# Create data directories
echo ""
echo "📁 Creating data directories..."
mkdir -p data/raw data/processed data/models
echo "✅ Directories created"

# Collect data
echo ""
echo "🌐 Collecting clinical trial data from ClinicalTrials.gov..."
echo "This will take 5-10 minutes (collecting 2,000 trials)..."
python src/data_collection/collect_trials.py

# Engineer features
echo ""
echo "🔨 Engineering features..."
python src/features/engineer_features.py

# Train models
echo ""
echo "🤖 Training machine learning models..."
echo "This will take 3-5 minutes..."
python src/models/train_models.py

# Success message
echo ""
echo "=============================================="
echo "✅ Setup Complete!"
echo "=============================================="
echo ""
echo "🚀 To launch the dashboard, run:"
echo ""
echo "   source venv/bin/activate"
echo "   streamlit run src/app/streamlit_app.py"
echo ""
echo "The dashboard will open at: http://localhost:8501"
echo ""
echo "=============================================="
