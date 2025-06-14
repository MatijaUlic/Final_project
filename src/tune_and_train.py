import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from xgboost import XGBClassifier
from src.preprocess import preprocess_data
from sklearn.metrics import accuracy_score

# Ensure models/ directory exists
os.makedirs("models", exist_ok=True)

# Load and preprocess
df = pd.read_csv("data/Airlines.csv")
X, y = preprocess_data(df)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Cross-validation baseline
base_model = XGBClassifier(eval_metric="logloss", random_state=42)
cv_scores = cross_val_score(base_model, X_train, y_train,
                            cv=5, scoring="accuracy")
print(f"Baseline CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Hyperparameter search grid
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
    cv=3,
    verbose=1,
    random_state=42
)

# Perform search
search.fit(X_train, y_train)
best_params = search.best_params_
best_model = search.best_estimator_
print("🔍 Best params:", best_params)

# Evaluate on test set
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"✅ Test accuracy with tuned model: {test_acc:.4f}")

# Log and save model
with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.sklearn.log_model(best_model, "model")
    joblib.dump(best_model, "models/best_model.pkl")
    print("✅ Tuned model trained and saved.")
