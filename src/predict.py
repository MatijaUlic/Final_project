import pandas as pd, joblib
from src.preprocess import preprocess_data

model = joblib.load("models/best_model.pkl")
df = pd.read_csv("batch_prediction_dataset/dataset.csv")
X = preprocess_data(df, training=False)
probas = model.predict_proba(X)[:,1]
df['predicted_delay'] = (probas >= 0.5).astype(int)
df['delay_probability'] = probas
df.to_csv("batch_prediction_dataset/results.csv", index=False)

total = len(df)
on_time = sum(df.predicted_delay == 0)
delayed = sum(df.predicted_delay == 1)
report = (f"🛬 Batch Prediction Report\n"
          f"Total flights: {total}\n"
          f"On Time: {on_time} ({on_time/total:.1%})\n"
          f"Delayed: {delayed} ({delayed/total:.1%})\n")
with open("batch_prediction_dataset/report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("✅ Batch predictions completed.")
print(report)
