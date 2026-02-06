# EMERGENCY FIX: Plotly Import Error on Streamlit Cloud

## The Error
```
ModuleNotFoundError: import plotly.express as px
```

This means Plotly didn't install correctly during deployment.

---

## SOLUTION 1: Complete Requirements File ⭐ TRY THIS FIRST

Replace your `requirements.txt` with this complete version:

```txt
streamlit==1.29.0
pandas==2.1.4
numpy==1.26.2
plotly==5.18.0
scikit-learn==1.3.2
xgboost==2.0.3
lightgbm==4.1.0
joblib==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
altair==5.2.0
```

**AND** create `runtime.txt`:
```txt
python-3.11
```

Then:
```bash
git add requirements.txt runtime.txt
git commit -m "Fix plotly import error"
git push
```

---

## SOLUTION 2: Debug Mode

Temporarily replace your main app path to test imports:

1. **Add `test_imports.py` to your repo** (file provided)
2. **In Streamlit Cloud settings**, change main file to: `test_imports.py`
3. **Deploy and check** which packages are failing
4. **Once verified**, change back to: `src/app/streamlit_app.py`

---

## SOLUTION 3: Clean Deployment

Sometimes the cache gets corrupted. Force a clean rebuild:

1. In Streamlit Cloud, click your app
2. Click "⋮" menu → "Delete app"
3. Create a new app pointing to same repo
4. Set main file: `src/app/streamlit_app.py`
5. Set Python: 3.11 (in Advanced settings)

---

## SOLUTION 4: Minimal Requirements Test

Test with absolute minimum to isolate the issue:

**requirements.txt:**
```txt
streamlit==1.29.0
plotly==5.18.0
pandas==2.1.4
numpy==1.26.2
```

If this works, add packages back one at a time:
```bash
# Add ML packages
streamlit==1.29.0
plotly==5.18.0
pandas==2.1.4
numpy==1.26.2
joblib==1.3.2
scikit-learn==1.3.2

# Then add xgboost/lightgbm
xgboost==2.0.3
lightgbm==4.1.0
```

---

## SOLUTION 5: Use Latest Versions

Switch to newest stable versions that definitely support Python 3.11:

```txt
streamlit==1.40.2
plotly==5.24.1
pandas==2.2.3
numpy==2.0.2
scikit-learn==1.5.2
xgboost==2.1.3
lightgbm==4.5.0
joblib==1.4.2
matplotlib==3.9.3
```

**⚠️ If using this:** Re-run pipeline locally first:
```bash
pip install -r requirements.txt --upgrade
python run_pipeline.py
git add data/
git commit -m "Update models for new versions"
git push
```

---

## SOLUTION 6: Check Streamlit Cloud Settings

Verify these settings in Streamlit Cloud:

1. **Python version:** Should be `3.11` (NOT 3.13)
2. **Main file path:** `src/app/streamlit_app.py` (with forward slashes)
3. **Advanced settings → Secrets:** Empty (unless you added secrets)
4. **Resource limits:** Default is fine

---

## SOLUTION 7: Add System Dependencies

Update `packages.txt` to include C libraries plotly might need:

```txt
build-essential
libgomp1
python3-dev
gcc
g++
```

---

## DETAILED DEBUGGING STEPS

### Step 1: Check Deployment Logs
Look for these lines in Streamlit Cloud logs:

**Good:**
```
✅ Successfully installed plotly-5.18.0
✅ Your app is now deployed
```

**Bad:**
```
❌ ERROR: Failed building wheel for plotly
❌ Could not install packages due to an OSError
```

### Step 2: Look for Specific Errors

**If you see "wheel" errors:**
```txt
# Add to requirements.txt
wheel==0.42.0
setuptools==69.0.0
```

**If you see "gcc" or "compile" errors:**
```txt
# Add to packages.txt
build-essential
python3-dev
```

**If you see "numpy" version conflicts:**
```txt
# Pin numpy first in requirements.txt
numpy==1.26.2
plotly==5.18.0
...
```

---

## QUICK CHECKLIST

Before deploying again, verify:

- [ ] `runtime.txt` exists with `python-3.11`
- [ ] `requirements.txt` has plotly==5.18.0 (or newer)
- [ ] `packages.txt` has build-essential
- [ ] Main file path is `src/app/streamlit_app.py`
- [ ] Python version in settings is 3.11
- [ ] No typos in file names
- [ ] Files are committed to GitHub
- [ ] You're on the correct branch (main)

---

## NUCLEAR OPTION: Start Fresh

If nothing works:

1. **Create new branch:**
   ```bash
   git checkout -b streamlit-fix
   ```

2. **Use this minimal structure:**
   ```
   your-repo/
   ├── streamlit_app.py          # Move app to root
   ├── requirements.txt           # Minimal requirements
   ├── runtime.txt                # python-3.11
   └── data/                      # Your data files
   ```

3. **Move streamlit_app.py to root:**
   ```bash
   cp src/app/streamlit_app.py streamlit_app.py
   ```

4. **Update import paths in streamlit_app.py:**
   ```python
   # Change:
   data_dir = Path(__file__).parent.parent / 'data'
   # To:
   data_dir = Path(__file__).parent / 'data'
   ```

5. **Deploy with main file:** `streamlit_app.py`

---

## TESTING LOCALLY BEFORE DEPLOY

Always test locally first:

```bash
# Create fresh virtual environment
python3.11 -m venv test_env
source test_env/bin/activate  # or test_env\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt

# Test imports
python -c "import plotly.express as px; print('Success!')"

# Run app
streamlit run src/app/streamlit_app.py
```

If it works locally but not on Streamlit Cloud, the issue is environment-specific.

---

## STILL STUCK?

1. **Copy full deployment logs** from Streamlit Cloud
2. **Check if plotly actually installed:**
   - Look for "Successfully installed plotly-X.X.X" in logs
3. **Try the test_imports.py** to see exact error
4. **Contact Streamlit support** with logs if all else fails

---

## FILES PROVIDED

I've created these files for you:

1. **requirements_complete.txt** - Full dependencies ⭐ USE THIS
2. **requirements_debug.txt** - Minimal for testing
3. **runtime.txt** - Python version specification
4. **test_imports.py** - Import verification script
5. **packages.txt** - System dependencies

**Next step:** Replace your requirements.txt with requirements_complete.txt content and add runtime.txt!
