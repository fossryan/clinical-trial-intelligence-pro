#!/bin/bash

# Clinical Trial Intelligence Platform - Setup Script
# Automates the entire data collection → model training → deployment pipeline

set -e  # Exit on error

echo "🧬 Clinical Trial Intelligence Platform - Setup"
echo "=============================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Create directory structure
echo ""
echo "📁 Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/models
mkdir -p docs/screenshots
mkdir -p notebooks
mkdir -p tests
echo "✅ Directories created"

# Run data collection
echo ""
echo "🌐 Collecting clinical trial data..."
echo "This may take 5-10 minutes..."
python src/data_collection/collect_trials.py

# Engineer features
echo ""
echo "🔨 Engineering features..."
python src/features/engineer_features.py

# Train models
echo ""
echo "🤖 Training machine learning models..."
echo "This may take 2-5 minutes..."
python src/models/train_models.py

echo ""
echo "=============================================="
echo "✅ Setup complete!"
echo ""
echo "🚀 To launch the dashboard, run:"
echo "   streamlit run src/app/streamlit_app.py"
echo ""
echo "📊 The dashboard will open at: http://localhost:8501"
echo "=============================================="
