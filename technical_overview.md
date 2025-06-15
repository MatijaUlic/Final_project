Flight Delay Prediction
MLOps & System Design Project


Course:  MLOps and System Design – EADA Business School
Team:    Matija Ulic, Tim Reding

1. PROBLEM STATEMENT & ANALYSIS

Our goal is to build a predictive model that classifies if a flight
will be delayed. This binary classification problem uses airline data
(features like airline code, departure airport, day/time) stored in
'data/Airlines.csv'.

2. System design choices:
- Preprocessing with label encoding
- XGBoost classifier with hyperparameter tuning
- Experiment tracking via MLflow
- Full test suite with Pytest
- CI/CD pipeline using GitHub Actions

3. PROJECT STRUCTURE

    .
    ├── data/
    │   └── Airlines.csv                     Raw training dataset
    ├── batch_prediction_dataset/
    │   ├── dataset.csv                      Input for batch predictions
    │   ├── results.csv                      Output predictions
    │   └── report.txt                       Summary report
    ├── models/
    │   └── best_model.pkl                  Saved trained model
    ├── notebook/
    │   ├── preprocess.py                    -> preprocessing logic
    │   ├── tune_and_train.py                -> training + MLflow logging
    │   └── predict.py       
    ├── src/
    │   ├── preprocess.py                    -> preprocessing logic
    │   ├── tune_and_train.py                -> training + MLflow logging
    │   └── predict.py                       -> prediction 
    ├── tests/
    │   ├── test_preprocess.py             -> preprocess unit test
    │   ├── test_train_pipeline.py         -> train pipeline test
    │   ├── test_predict.py                -> prediction CLI test
    │   └── test_main_flow.py              -> end-to-end workflow test
    ├── main.py                             Orchestrator script
    ├── requirements.txt                    Project dependencies
    ├── .github/
    │   └── workflows/ci.yml                CI/CD pipeline definition
    └── README.txt                          This file
 



4. USAGE

 Full pipeline (training + prediction):
   $ python -m main

   Workflow:
   - Train and tune model (tune_and_train)
   - Log metrics and save model via MLflow
   - Run batch prediction on default dataset
Run main.py to get full predicton

5. DETAILED CODE EXPLANATION
-
1) src/preprocess.py
   - Loads DataFrame
   - Drops 'id' and 'Flight' if present
   - Label-encodes categorical features
   - Returns (X, y) for training, or X for prediction

2) src/tune_and_train.py
   - Loads and preprocesses data
   - Splits into train/test
   - Baseline model with CV using accuracy
   - Hyperparameter tuning via RandomizedSearchCV
   - Evaluates multiple metrics: accuracy, precision, recall, f1, ROC AUC, PR AUC
   - Logs results and model artifact via MLflow
   - Saves model to 'models/best_model.pkl'

3) src/predict.py
   - Loads model, preprocesses input, predicts both class and probability
   - Outputs results CSV and summary report
   - Supports custom paths via flags

4) main.py
   - Handles orchestration 
   - Runs training and/or prediction modules 
   - Ensures exit on error for CI
6. TESTS
Run:
   $ pytest --maxfail=1 --disable-warnings -q

Tests include:
- Test that preprocessing works and drops 'Delay' for training
- End-to-end train-and-tune pipeline on dummy data
- Full main.py flow including both training and prediction

7. CI / GITHUB ACTIONS

Defined in `.github/workflows/ci.yml`, triggered on pushes and PRs:
- Installs dependencies
- Executes tests
- Runs the full main pipeline (train + predict)
- Validates that model is saved in 'models/'

8. EXPERIMENT TRACKING VIA MLFLOW

- Logs hyperparameters and metrics per run
- Stores model artifact
- Use `mlflow ui` to inspect results and compare runs

9. SUMMARY

This project implements a complete MLOps-driven machine learning pipeline,
including version control, experimental tracking, automated testing, and
CI/CD deployment. 


