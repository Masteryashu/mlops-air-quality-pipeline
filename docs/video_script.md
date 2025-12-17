# Video Presentation Script (6 Minutes)

## 0:00 - 1:00 | Introduction & Problem Statement
- **Intro**: "Hi, I'm [Name] and this is my MLOps project for predicting Air Quality."
- **Goal**: "The goal is to predict Carbon Monoxide (CO) levels based on sensor data."
- **Stack**: "I used H2O for AutoML, standard Scikit-learn models, MLflow for tracking, and FastAPI for deployment, all hosted on AWS."

## 1:00 - 2:00 | Data Pipeline
- **Show Code**: Briefly show data structure (`data/raw` vs `data/processed`).
- **Explain**: "I cleaned the dataset, handled missing values, and engineered datetime features."
- **Show Split**: "Data was split 35/35/30 into train, validation, and test sets to preserve time order."

## 2:00 - 3:00 | Model Training & MLflow
- **Action**: "Let's look at the MLflow dashboard."
- **Show UI**: Open `http://44.220.133.154:5000`.
- **Highlight**: "Here are my experiments. You can see runs for XGBoost, Random Forest, and Neural Network."
- **Compare**: "Random Forest was the Champion model with the lowest RMSE (0.55)."
- **Artifacts**: Click on a run and show the artifacts (model file, requirements.txt) stored in S3.

## 3:00 - 4:00 | Deployment (FastAPI)
- **Action**: "Now, let's look at the deployed API."
- **Show Docs**: Open `http://localhost:8000/docs`.
- **Demo**: "I'll make a prediction using the Champion model."
- **Click**: Execute a POST request to `/predict_model2` (Random Forest).
- **Result**: Show the JSON response with the predicted CO level.

## 4:00 - 5:00 | Drift Analysis
- **Context**: "Finally, we monitor the model in production."
- **Show Report**: Open `reports/drift_analysis/evidently_data_drift.html`.
- **Explain**: "I used Evidently AI to check for data drift. As you can see, several features show drift because the test set covers a different time period than training."
- **Performance**: Switch to `evidently_performance_drift.html` (if generated) or mention the RMSE degradation.

## 5:00 - 6:00 | Conclusion
- **Summary**: "To summarize, we built a full pipeline from raw data to a monitored, deployed API."
- **Closing**: "Thank you for watching."
