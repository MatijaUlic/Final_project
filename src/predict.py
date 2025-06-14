import pandas as pd
import joblib
from src.preprocess import preprocess_data

# Load trained model
model = joblib.load("models/best_model.pkl")

# Load batch data
df = pd.read_csv("batch_prediction_dataset/dataset.csv")
X = preprocess_data(df, training=False)

# Predict
preds = model.predict(X)
df['predicted_delay'] = preds

# Save results
df.to_csv("batch_prediction_dataset/results.csv", index=False)

# Generate mini report
delay_counts = df['predicted_delay'].value_counts()
with open("batch_prediction_dataset/report.txt", "w") as f:
    f.write("🛬 Batch Prediction Report\n")
    f.write("========================\n")
    f.write(f"Total flights: {len(df)}\n")
    for cls, count in delay_counts.items():
        label = "Delayed" if cls == 1 else "On Time"
        f.write(f"{label}: {count} ({count/len(df):.1%})\n")

print("✅ Batch predictions completed and saved.")
