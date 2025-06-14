import pandas as pd, joblib, mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.preprocess import preprocess_data

df = pd.read_csv("data/Airlines.csv")
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
mlflow.log_metric("accuracy", acc)
joblib.dump(model, "models/best_model.pkl")
print(f"✅ Model trained and saved. Accuracy: {acc:.4f}")