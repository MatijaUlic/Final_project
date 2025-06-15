# tests/test_train_pipeline.py
import os
import pandas as pd
import joblib
import subprocess
import sys

def test_tune_and_train_pipeline(tmp_path):
    # 1. Create a dummy dataset with 10 rows
    df = pd.DataFrame({
        "Category": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "Numeric": list(range(1, 11)),
        "Delay":   [0, 1] * 5,
    })
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "Airlines.csv").write_text(df.to_csv(index=False))

    # 2. Adjust environment vars for smaller CV folds
    env = os.environ.copy()
    env["PYTHONPATH"] = str(os.path.abspath("."))
    env["CV_FOLDS"] = "2"
    env["SEARCH_CV"] = "2"

    # 3. Execute the training/tuning script
    subprocess.run(
        [sys.executable, "-m", "src.tune_and_train"],
        cwd=tmp_path,
        env=env,
        check=True
    )

    # 4. Verify the model artifact exists
    model_path = tmp_path / "models" / "best_model.pkl"
    assert model_path.exists()

    # 5. Load the model and test a prediction
    model = joblib.load(model_path)
    pred = model.predict([[0, 1]])
    assert len(pred) == 1
