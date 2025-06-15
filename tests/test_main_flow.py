# tests/test_main_flow.py

import os
import pandas as pd
import subprocess
import sys

def test_main_flow(tmp_path):
    # 1. Prepare training dataset
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df = pd.DataFrame({
        "Category": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "Numeric": list(range(1, 11)),
        "Delay":   [0, 1] * 5,
    })
    (data_dir / "Airlines.csv").write_text(df.to_csv(index=False))

    # 2. Prepare batch prediction input
    batch_dir = tmp_path / "batch_prediction_dataset"
    batch_dir.mkdir()
    pd.DataFrame({
        "id": [1, 2],
        "Flight": ["X", "Y"],
        "Category": ["A", "B"],
        "Numeric": [0, 1]
    }).to_csv(batch_dir / "dataset.csv", index=False)

    # 3. Set environment variables so scripts are found and folds are small
    env = os.environ.copy()
    env["PYTHONPATH"] = str(os.path.abspath("."))
    env["CV_FOLDS"] = "2"
    env["SEARCH_CV"] = "2"

    # 4. Run the main pipeline
    result = subprocess.run(
        [sys.executable, "-m", "main"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True
    )
    # Ensure the main completed successfully
    assert result.returncode == 0, result.stderr

    # 5. Validate existence of important outputs
    assert (tmp_path / "models" / "best_model.pkl").exists()
    assert (batch_dir / "results.csv").exists()
    assert (batch_dir / "report.txt").exists()

    out = result.stdout
    assert "Starting hyperparameter tuning & training..." in out
    assert "Completed training. Now running batch predictions..." in out
    assert "All steps completed successfully." in out
