# 🚀 Clinical Trial Intelligence - Installation Walkthrough

**Complete step-by-step guide to get your dashboard running in 30 minutes.**

---

## Before You Start

**What you need:**
- Python 3.10 or higher ([Download here](https://www.python.org/downloads/))
- 2GB free disk space
- Internet connection
- Terminal/Command Prompt

**How long will this take?**
- Setup: 5 minutes
- Data collection: 5-10 minutes  
- Model training: 3-5 minutes
- **Total: ~20-30 minutes**

---

## OPTION 1: One-Click Setup (Easiest) ⭐

### On Mac/Linux:
```bash
# 1. Extract the download
cd ~/Downloads
tar -xzf clinical-trial-intelligence-SOLO.tar.gz
cd clinical-trial-intelligence

# 2. Run the magic script
chmod +x QUICKSTART.sh
./QUICKSTART.sh

# 3. When it's done, launch dashboard:
source venv/bin/activate
streamlit run src/app/streamlit_app.py
```

### On Windows:
```bash
# 1. Extract the download (right-click → Extract All)
# 2. Open Command Prompt in that folder
# 3. Run:
QUICKSTART.bat

# 4. When it's done, launch dashboard:
venv\Scripts\activate
streamlit run src\app\streamlit_app.py
```

**That's it!** The script does everything automatically.

---

## OPTION 2: Manual Step-by-Step (If Script Fails)

### Step 1: Extract Files (1 minute)

**Mac/Linux:**
```bash
cd ~/Downloads
tar -xzf clinical-trial-intelligence-SOLO.tar.gz
cd clinical-trial-intelligence
```

**Windows:**
- Right-click `clinical-trial-intelligence-SOLO.tar.gz`
- Click "Extract All"
- Open folder in Command Prompt

---

### Step 2: Check Python (1 minute)

```bash
python3 --version
# or on Windows:
python --version
```

**You should see:** `Python 3.10.x` or higher

**If not installed:**
1. Download from [python.org](https://www.python.org/downloads/)
2. Install (check "Add to PATH" on Windows)
3. Restart terminal
4. Try again

---

### Step 3: Create Virtual Environment (2 minutes)

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**You should see `(venv)` appear in your terminal.**

---

### Step 4: Install Dependencies (3 minutes)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**This installs:**
- pandas, numpy (data processing)
- scikit-learn, xgboost (machine learning)
- streamlit, plotly (dashboard)
- requests (API calls)

**If you see errors about Visual C++** (Windows):
- Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

---

### Step 5: Create Data Directories (30 seconds)

**Mac/Linux:**
```bash
mkdir -p data/raw data/processed data/models
```

**Windows:**
```bash
mkdir data\raw
mkdir data\processed
mkdir data\models
```

---

### Step 6: Collect Clinical Trial Data (5-10 minutes)

```bash
python src/data_collection/collect_trials.py
```

**What this does:**
- Connects to ClinicalTrials.gov API
- Downloads 2,000 Phase 2-3 trials
- Saves to `data/raw/clinical_trials_raw_[timestamp].csv`

**You'll see:**
```
Starting data collection (max: 2000 trials)...
Retrieved 100 trials...
Retrieved 200 trials...
...
Total trials collected: 2000
Data saved to: data/raw/clinical_trials_raw_20260202_143022.csv
```

**If it fails:**
- Check internet connection
- ClinicalTrials.gov might be slow (try again)
- Reduce `max_studies=2000` to `1000` in the code

---

### Step 7: Engineer Features (2 minutes)

```bash
python src/features/engineer_features.py
```

**What this does:**
- Loads raw data
- Creates 40+ predictive features
- Saves to `data/processed/clinical_trials_features_[timestamp].csv`

**You'll see:**
```
Loading data from: data/raw/clinical_trials_raw_20260202_143022.csv
Loaded 2000 trials
Engineering features...
Feature engineering complete. Shape: (2000, 45)
Features saved to: data/processed/clinical_trials_features_20260202_143525.csv
```

---

### Step 8: Train Machine Learning Models (3-5 minutes)

```bash
python src/models/train_models.py
```

**What this does:**
- Trains 3 models (Logistic Regression, XGBoost, LightGBM)
- Generates SHAP explanations
- Saves models to `data/models/`

**You'll see:**
```
==================================================
Training Logistic Regression (Baseline)
==================================================
Logistic Regression Performance:
  Accuracy:  0.7124
  F1 Score:  0.6845
  ROC AUC:   0.7213

==================================================
Training XGBoost
==================================================
XGBoost Performance:
  Accuracy:  0.7634
  F1 Score:  0.7421
  ROC AUC:   0.7812

==================================================
Training LightGBM
==================================================
LightGBM Performance:
  Accuracy:  0.7589
  F1 Score:  0.7389
  ROC AUC:   0.7791

Saved xgboost to data/models/xgboost_20260202_144012.joblib
```

**If training is slow:**
- This is normal (processing 2,000 trials)
- Get coffee ☕
- Should finish in 3-5 minutes

---

### Step 9: Launch Dashboard! 🚀

```bash
streamlit run src/app/streamlit_app.py
```

**You'll see:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.1.x:8501
```

**Your browser should automatically open to the dashboard!**

If not, manually go to: `http://localhost:8501`

---

## 🎉 Success! What You'll See

Your dashboard has 4 pages:

1. **📊 Overview** - Portfolio success rates, phase attrition
2. **🎯 Risk Predictor** - Input trial → Get success probability
3. **🔍 Deep Dive** - Filter and explore trials
4. **📈 Model Performance** - Feature importance, SHAP

**Try this:**
1. Go to "🎯 Risk Predictor"
2. Input: Phase 2, Oncology, 80 patients
3. Click "Predict Trial Risk"
4. See: ~40% success probability (high risk!)

---

## 📸 Take Screenshots for Your Portfolio

**Important pages to capture:**
1. Overview page (shows your data scale)
2. Risk Predictor with a prediction
3. UMAP or Sankey diagram (looks impressive)
4. Feature importance chart

**Use these in:**
- Resume
- LinkedIn post
- Cover letters
- Interview presentations

---

## 🛑 Troubleshooting

### "Module not found" error
```bash
# Make sure virtual environment is activated:
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies:
pip install -r requirements.txt
```

### "No module named 'streamlit'"
```bash
# Install streamlit specifically:
pip install streamlit
```

### "Port 8501 already in use"
```bash
# Kill existing Streamlit:
# Mac/Linux: pkill -f streamlit
# Windows: taskkill /F /IM streamlit.exe

# Or use different port:
streamlit run src/app/streamlit_app.py --server.port 8502
```

### Dashboard is blank/not loading
```bash
# Clear Streamlit cache:
streamlit cache clear

# Restart:
streamlit run src/app/streamlit_app.py
```

### "No data found" errors
- Make sure you ran steps 6-8 first
- Check `data/raw/` and `data/processed/` have CSV files
- Re-run data collection if needed

---

## ✅ Verification Checklist

You're successful if:
- [ ] Dashboard opens in browser
- [ ] You see "Clinical Trial Risk Intelligence Platform" header
- [ ] Overview page shows statistics (trial counts, success rates)
- [ ] You can navigate between tabs
- [ ] Risk Predictor accepts input
- [ ] Charts/visualizations display

---

## 🎯 Next Steps

Now that it's working:

1. **Play with it** - Try different inputs in Risk Predictor
2. **Take screenshots** - You'll need these for portfolio
3. **Read the code** - Understand what it does
4. **Prepare to demo** - Practice 5-minute walkthrough

**Tomorrow:**
- Create GitHub repo
- Deploy to Streamlit Cloud
- Add to resume

---

## 💬 Questions?

If stuck:
1. Check error message carefully
2. Google the specific error
3. Verify Python version (3.10+)
4. Make sure virtual environment activated
5. Try QUICKSTART script instead

**Common issue:** "It worked but dashboard looks weird"
- This is normal! Some styling may not load perfectly locally
- Will look better once deployed to Streamlit Cloud

---

## 🚀 You Did It!

**You now have a working, production-quality data science project!**

This is your portfolio centerpiece. It shows:
- ✅ API integration (ClinicalTrials.gov)
- ✅ Feature engineering (40+ features)
- ✅ Machine learning (XGBoost, SHAP)
- ✅ Visualization (Interactive dashboard)
- ✅ Domain knowledge (Clinical trials)

**This alone can get you interviews at Pfizer, BMS, Gilead.**

---

Ready to deploy it? See `DEPLOYMENT_GUIDE.md` next!
