import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier
from src.preprocess import preprocess_data

# Configurable CV settings via environment variables
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))
SEARCH_CV = int(os.getenv("SEARCH_CV", "3"))

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Load and split the dataset
df = pd.read_csv("data/Airlines.csv")
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline cross-validation
base_model = XGBClassifier(eval_metric="logloss", random_state=42)
cv_scores = cross_val_score(base_model, X_train, y_train, cv=CV_FOLDS, scoring="accuracy")
print(f"Baseline CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Hyperparameter grid
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [6, 10, 14],
    "learning_rate": [0.05, 0.1, 0.2],
    "scale_pos_weight": [1, 3, 5, 7],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

search = RandomizedSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    scoring="accuracy",
    cv=SEARCH_CV,
    verbose=1,
    random_state=42
)

# Fit and tune
search.fit(X_train, y_train)
best_params = search.best_params_
best_model = search.best_estimator_
print("Best params:", best_params)

# Evaluate on test set
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

# Log results
print(f"Test accuracy: {test_acc:.4f}")
print(f"Precision:    {precision:.4f}")
print(f"Recall:       {recall:.4f}")
print(f"F1-score:     {f1:.4f}")
print(f"ROC AUC:      {roc_auc:.4f}")
print(f"PR AUC:       {pr_auc:.4f}")

with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", test_acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.sklearn.log_model(best_model, "model")

# Persist the trained model
joblib.dump(best_model, "models/best_model.pkl")
print("All metrics logged and model saved.")
