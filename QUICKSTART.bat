@echo off
REM Clinical Trial Intelligence - Quick Start (Windows)
REM This script runs the complete pipeline and launches the app

echo.
echo ================================================================================
echo CLINICAL TRIAL INTELLIGENCE - QUICK START
echo ================================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

echo [1/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [2/4] Running data pipeline (this may take 15-20 minutes)...
echo       - Collecting clinical trial data
echo       - Engineering features
echo       - Training ML models
python run_pipeline.py
if errorlevel 1 (
    echo ERROR: Pipeline failed
    pause
    exit /b 1
)

echo.
echo [3/4] Pipeline complete! Generated files:
dir /s /b data\models\*.joblib
dir /s /b data\processed\*.csv

echo.
echo [4/4] Launching Streamlit app...
echo.
echo ================================================================================
echo App will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo ================================================================================
echo.

streamlit run src\app\streamlit_app.py

pause
