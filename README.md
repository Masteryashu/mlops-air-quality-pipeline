# MLOps Air Quality Prediction Project

## Overview

This project implements a complete MLOps pipeline for predicting CO(GT) concentration from the Air Quality UCI dataset. The pipeline includes data preprocessing, AutoML analysis with H2O, manual training of three models (XGBoost, Random Forest, Neural Network), MLflow tracking, FastAPI deployment, and drift monitoring.

**Dataset:** Air Quality UCI (9,357 rows → 6,694 after cleaning)  
**Target:** CO(GT) - Carbon Monoxide concentration  
**Models:** XGBoost, Random Forest, Neural Network (MLP)  
**Tools:** H2O AutoML, MLflow, FastAPI, Evidently

---

## Project Structure

```
mlops_airquality/
├── api/                      # FastAPI application
│   ├── main.py              # API endpoints
│   ├── schemas.py           # Pydantic schemas
│   └── test_api.py          # API test client
├── config/                   # Configuration
│   └── mlflow_config.py     # MLflow settings
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned and split data
├── docs/                     # Documentation
├── models/                   # Saved models and artifacts
├── reports/                  # Drift analysis reports
│   └── drift_analysis/
├── scripts/                  # Training and analysis scripts
│   ├── 01_explore_data.py
│   ├── 02_clean_data.py
│   ├── 03_split_data.py
│   ├── 04_h2o_automl.py
│   ├── 05_train_xgboost.py
│   ├── 06_train_random_forest.py
│   ├── 07_train_neural_network.py
│   └── 08_drift_analysis.py
├── .env.example             # Environment variables template
└── pyproject.toml           # Dependencies
```

---

## Setup Instructions

### 1. Prerequisites

- **Python 3.11+**
- **UV package manager** (install from https://github.com/astral-sh/uv)
- **Java 8+** (required for H2O AutoML)

### 2. Install Dependencies

```powershell
# Navigate to project directory
cd mlops_airquality

# Install dependencies with UV
uv sync
```

### 3. Configure Environment (Optional)

For remote MLflow tracking, copy `.env.example` to `.env` and fill in your AWS credentials:

```powershell
cp .env.example .env
# Edit .env with your AWS and MLflow settings
```

---

## Running the Pipeline

### Step 1: Data Exploration

```powershell
uv run python scripts/01_explore_data.py
```

### Step 2: Data Cleaning

```powershell
uv run python scripts/02_clean_data.py
```

**Output:** `data/processed/cleaned_data.csv`

### Step 3: Time-Based Splitting

```powershell
uv run python scripts/03_split_data.py
```

**Output:** `train.csv`, `validate.csv`, `test.csv` (35/35/30 split)

### Step 4: H2O AutoML Analysis

```powershell
uv run python scripts/04_h2o_automl.py
```

**Output:** `models/h2o_automl_results.json`

### Step 5: Train Models with MLflow

First, start MLflow UI (local):

```powershell
mlflow ui --port 5000
```

Then train all three models:

```powershell
# XGBoost
uv run python scripts/05_train_xgboost.py

# Random Forest
uv run python scripts/06_train_random_forest.py

# Neural Network
uv run python scripts/07_train_neural_network.py
```

View results at: http://localhost:5000

### Step 6: Run Drift Analysis
 
 ```powershell
 uv run python scripts/08_drift_analysis.py
 ```
 
 **Output:** HTML reports (`evidently_data_drift.html`, `evidently_performance_drift.html`) in `reports/drift_analysis/`
 
 ### Step 7: Deploy FastAPI
 
 ```powershell
 uv run uvicorn api.main:app --reload --port 8000
 ```
 
API available at: http://localhost:8000  
 Interactive docs: http://localhost:8000/docs
 
 ### Step 8: Test API
 
 In a new terminal:
 
 ```powershell
 uv run python api/test_api.py
 ```
 
 ---
 
 ## API Endpoints
 
 ### Health Check
 ```bash
 GET /
 ```
 
 ### Model Information
 ```bash
 GET /models/info
 ```
 
 ### Predictions
 
 **XGBoost:**
 ```bash
 POST /predict_model1
 ```
 
 **Random Forest:**
 ```bash
 POST /predict_model2
 ```
 
 **Neural Network:**
 ```bash
 POST /predict_model3
 ```
 
 **Example Request:**
 ```bash
 curl -X POST http://localhost:8000/predict_model1 \
   -H "Content-Type: application/json" \
   -d '{
     "PT08_S1CO": 1200.0,
     "NMHCGT": 150.0,
     "C6H6GT": 10.5,
     "PT08_S2NMHC": 950.0,
     "NOxGT": 200.0,
     "PT08_S3NOx": 800.0,
     "NO2GT": 120.0,
     "PT08_S4NO2": 1500.0,
     "PT08_S5O3": 1100.0,
     "T": 20.0,
     "RH": 50.0,
     "AH": 1.0,
     "hour": 12,
     "day_of_week": 3,
     "month": 6,
     "is_weekend": 0
   }'
 ```
 
 ---
 
 ## Remote MLflow Setup (AWS)
 
 See `docs/setup_mlflow_remote.md` for detailed instructions on:
 1. Setting up PostgreSQL on Neon.com
 2. Creating S3 bucket for artifacts
 3. Launching EC2 instance for MLflow server
 4. Configuring local environment
 
 ---
 
 ## Results Summary
 
 ### Data Processing
 - **Original rows:** 9,357
 - **After cleaning:** 6,694 (71.5% retained)
 - **Features:** 16 (added NMHCGT to model schema)
 - **Train/Val/Test:** 35% / 35% / 30%
 
 ### Model Performance
 *(Test Set Results)*
 
 | Model | Test RMSE | Test R² | Status |
 |-------|-----------|---------|--------|
 | XGBoost | 0.6609 | 0.5138 | Registered |
 | **Random Forest** | **0.5549** | **0.6573** | **Champion** |
 | Neural Network | 2.0323 | < 0 | Needs Tuning |
 
 **Champion Model:** Random Forest (Lowest RMSE)

### Drift Analysis
See `reports/drift_analysis/` for detailed reports.

---

## Team Members

*Yashwanth Kosanam*
*Loka Poojitha Dondeti*
*Monika Subramanian*

---

## Video Presentation

*(Add YouTube link here after recording)*

**Duration:** 6 minutes  
**Content:** Complete workflow demonstration from data cleaning to drift analysis

---

## License

Educational project for MLOps course.

---

## Troubleshooting

### H2O AutoML fails
- Ensure Java 8+ is installed: `java -version`
- Download from: https://www.java.com/download/

### Models not loading in API
- Train all models first using scripts 05-07
- Check MLflow UI to verify models are registered

### MLflow connection error
- Ensure MLflow server is running: `mlflow ui --port 5000`
- Check `MLFLOW_TRACKING_URI` in `.env`

---

## Contact

For questions or issues, please contact the team members listed above.
