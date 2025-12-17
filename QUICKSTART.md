# Quick Start Guide - MLOps Air Quality Prediction

This guide will help you run the complete pipeline from start to finish.

---

## Prerequisites Check

```powershell
# Check UV is installed
uv --version

# Check Python version
uv run python --version  # Should be 3.11+

# Check Java (required for H2O AutoML)
java -version  # Should be 8+
# If not installed: https://www.java.com/download/
```

---

## Step-by-Step Execution

### 1. Data Preparation (Already Done ✅)

The data cleaning and splitting has already been executed:
- ✅ `data/processed/cleaned_data.csv` created
- ✅ `data/processed/train.csv` created
- ✅ `data/processed/validate.csv` created
- ✅ `data/processed/test.csv` created

If you want to re-run:
```powershell
uv run python scripts/02_clean_data.py
uv run python scripts/03_split_data.py
```

---

### 2. Start MLflow Server

**IMPORTANT:** You must start MLflow server BEFORE running training scripts!

```powershell
# Open a new terminal and keep it running
cd mlops_airquality
mlflow ui --port 5000
```

You should see:
```
[INFO] Starting gunicorn 20.1.0
[INFO] Listening at: http://127.0.0.1:5000
```

Open browser: http://localhost:5000

---

### 3. Train All Three Models

**In a NEW terminal** (keep MLflow server running):

```powershell
cd mlops_airquality

# Train XGBoost
uv run python scripts/05_train_xgboost.py

# Train Random Forest
uv run python scripts/06_train_random_forest.py

# Train Neural Network
uv run python scripts/07_train_neural_network.py
```

**Expected output for each:**
- Training progress
- Metrics summary (RMSE, MAE, R²)
- MLflow logging confirmation
- Model registration confirmation

**Verify in MLflow UI:**
- Go to http://localhost:5000
- Click "Experiments" → "air-quality-prediction"
- You should see 3 runs
- Each run should have metrics and artifacts

---

### 4. Compare Models

In MLflow UI:
1. Select all 3 runs (checkboxes)
2. Click "Compare"
3. View metrics side-by-side
4. Identify champion model (lowest test_rmse)
5. **Take screenshot for documentation**

Create comparison table:
| Model | Test RMSE | Test MAE | Test R² |
|-------|-----------|----------|---------|
| XGBoost | [from UI] | [from UI] | [from UI] |
| Random Forest | [from UI] | [from UI] | [from UI] |
| Neural Network | [from UI] | [from UI] | [from UI] |

---

### 5. Deploy FastAPI

**In a NEW terminal** (keep MLflow server running):

```powershell
cd mlops_airquality
uv run uvicorn api.main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
✓ Loaded air-quality-xgboost
✓ Loaded air-quality-random-forest
✓ Loaded air-quality-neural-network
```

**Verify:**
- Open browser: http://localhost:8000
- You should see API info with "status": "running"
- Open interactive docs: http://localhost:8000/docs

---

### 6. Test API

**In a NEW terminal** (keep API server running):

```powershell
cd mlops_airquality
uv run python api/test_api.py
```

**Expected output:**
- Health check ✓
- Model info ✓
- XGBoost prediction ✓
- Random Forest prediction ✓
- Neural Network prediction ✓
- Batch predictions ✓

**Take screenshots:**
- API test output
- Interactive docs at /docs
- Example prediction response

---

### 7. Run Drift Analysis

```powershell
cd mlops_airquality
uv run python scripts/08_drift_analysis.py
```

**Expected output:**
- Performance comparison table
- Data drift report saved
- Performance drift report saved
- Summary JSON saved

**View reports:**
```powershell
# Open in browser
start reports/drift_analysis/data_drift_report.html
start reports/drift_analysis/performance_drift_report.html
```

**Take screenshots of drift reports**

---

### 8. Optional: H2O AutoML

```powershell
cd mlops_airquality
uv run python scripts/04_h2o_automl.py
```

**Note:** Requires Java. If you get an error, install Java first.

---

## Troubleshooting

### "Connection refused" error when training models
**Problem:** MLflow server not running  
**Solution:** Start MLflow server first: `mlflow ui --port 5000`

### "Models not loaded" in API
**Problem:** Models not trained yet  
**Solution:** Train all three models first (Step 3)

### H2O AutoML fails
**Problem:** Java not installed  
**Solution:** Install Java 8+ from https://www.java.com/download/

### API can't find scaler for Neural Network
**Problem:** Neural Network not trained yet  
**Solution:** Run `uv run python scripts/07_train_neural_network.py`

---

## Terminal Setup Summary

You'll need **3 terminals** running simultaneously:

**Terminal 1: MLflow Server**
```powershell
mlflow ui --port 5000
```

**Terminal 2: FastAPI Server**
```powershell
uv run uvicorn api.main:app --reload --port 8000
```

**Terminal 3: Commands**
```powershell
# Use this for running scripts and tests
uv run python api/test_api.py
```

---

## Verification Checklist

- [ ] MLflow UI shows 3 runs in "air-quality-prediction" experiment
- [ ] All 3 models registered in Model Registry
- [ ] FastAPI running at http://localhost:8000
- [ ] API test client runs successfully
- [ ] Drift analysis reports generated
- [ ] Screenshots taken:
  - [ ] MLflow experiment runs
  - [ ] Model comparison table
  - [ ] Model Registry
  - [ ] FastAPI docs
  - [ ] API test results
  - [ ] Drift reports

---

## Next Steps

1. **Update README.md** with:
   - Team member names
   - Model performance results (from MLflow)
   - Champion model selection

2. **Prepare GitHub Repository:**
   ```powershell
   git init
   git add .
   git commit -m "MLOps Air Quality Prediction Project"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

3. **Record Video Presentation** (6 minutes):
   - Show data cleaning results
   - Demonstrate MLflow UI with all runs
   - Show model comparison
   - Demonstrate API endpoints
   - Show drift analysis reports
   - Explain champion model selection

4. **Upload to YouTube** (unlisted)

5. **Submit GitHub link**

---

## Quick Commands Reference

```powershell
# Start MLflow
mlflow ui --port 5000

# Train models
uv run python scripts/05_train_xgboost.py
uv run python scripts/06_train_random_forest.py
uv run python scripts/07_train_neural_network.py

# Start API
uv run uvicorn api.main:app --reload --port 8000

# Test API
uv run python api/test_api.py

# Run drift analysis
uv run python scripts/08_drift_analysis.py

# View drift reports
start reports/drift_analysis/data_drift_report.html
```

---

**Ready to go! Start with Terminal 1 (MLflow server) and work through the steps.** 🚀
