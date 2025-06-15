# tests/test_predict.py

import os
import sys
# Make sure tests can find src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from src.predict import main as predict_main

def test_predict_creates_outputs(tmp_path, monkeypatch, capsys):
    models_dir = tmp_path / "models"
    batch_dir = tmp_path / "batch_prediction_dataset"
    models_dir.mkdir()
    batch_dir.mkdir()

    dummy = DummyClassifier(strategy="uniform", random_state=42)
    dummy.fit([[0], [1]], [0, 1])
    joblib.dump(dummy, models_dir / "best_model.pkl")

    df_in = pd.DataFrame({
        "id": [1, 2],
        "Flight": ["AA", "BB"],
        "Feature1": [0, 1],
        "Feature2": [1, 0]
    })
    df_in.to_csv(batch_dir / "dataset.csv", index=False)

    monkeypatch.chdir(tmp_path)

    # Run prediction
    predict_main()

    captured = capsys.readouterr()
    assert " Batch predictions completed." in captured.out

    results = pd.read_csv(batch_dir / "results.csv")
    assert "predicted_delay" in results.columns
    assert "delay_probability" in results.columns
    assert len(results) == 2

    report = (batch_dir / "report.txt").read_text()
    assert "Batch Prediction Report" in report
    assert "Total flights: 2" in report
