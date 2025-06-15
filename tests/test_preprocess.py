# tests/test_preprocess.py

import pandas as pd
import pytest
from src.preprocess import preprocess_data

def test_preprocess_training_mode():
    df = pd.DataFrame({
        "id": [1, 2],
        "Flight": ["AA", "BB"],
        "Category": ["X", "Y"],
        "Numeric": [10.0, 20.0],
        "Delay": [0, 1]
    })
    X, y = preprocess_data(df, training=True)

    # Delay should be removed from features
    assert "Delay" not in X.columns
    # id and Flight should be dropped
    assert "id" not in X.columns
    assert "Flight" not in X.columns

    # Category column should be encoded as integers
    assert X["Category"].dtype == "int32" or X["Category"].dtype == "int64"
    # Numeric column unchanged
    assert all(X["Numeric"] == pd.Series([10.0, 20.0]))

    # Target vector correct
    assert list(y) == [0, 1]

def test_preprocess_inference_mode():
    df = pd.DataFrame({
        "Flight": ["AA", "BB"],
        "Category": ["X", "Y"],
        "Numeric": [1.5, 2.5]
    })
    X = preprocess_data(df, training=False)

    # Should return only features with no dropped or missing columns
    assert isinstance(X, pd.DataFrame)
    assert "Category" in X.columns and "Numeric" in X.columns
    assert "Delay" not in X.columns
