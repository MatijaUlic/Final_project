1. The Problem
We want to predict whether a flight will be delayed or arrive on time. To do this, we use past flight data—airline code, origin and destination airports, day of week, flight duration, and scheduled times. The system generates a probability that a flight is delayed and supports both bulk (batch) and real-time predictions.

2. Problem Type
This is a binary classification problem: flights are labeled as delayed (1) or on time (0). Since about 45% of flights are delayed, we track performance using not just accuracy, but also precision, recall, F1-score, ROC-AUC, and PR-AUC, which give us a more complete view of model quality.

3. System Design Highlights
Data & Preprocessing
We drop unneeded ID fields like id and Flight to avoid data leakage.

We convert all categorical text fields into numbers using simple label encoding.

Modeling Approach
We use XGBoost for its efficiency and strong performance with structured data.

We split data into train and test sets using stratification so both sets maintain similar ratios of delayed vs on-time flights.

We also apply cross-validation (with configurable fold count) to ensure robust evaluation and guard against overfitting.

Hyperparameter Tuning
We use RandomizedSearchCV to explore a grid of hyperparameters like tree depth and learning rate.

During development and testing, we adjust the fold count using environment variables (CV_FOLDS, SEARCH_CV) to speed up or scale up the search.

Experiment Tracking
We integrate MLflow to log model configurations and performance metrics.

The best-performing model is saved to models/best_model.pkl for further use or deployment.

CI/CD and Orchestration
We created a main script (main.py) that runs training and batch prediction with one command.

GitHub Actions handles testing, training, and batch prediction automatically on each pull request or merge.

Once training completes, the model is stored and ready for deployment.

Batch and On-Demand Prediction
Drop a CSV into batch_prediction_dataset/dataset.csv, and our predict.py script will run predictions, save results to results.csv, and produce a summary report.

We use the same preprocessing in both training and inference to ensure consistency.