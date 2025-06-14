import pytest
from src.preprocess import preprocess_data
import pandas as pd

def test_preprocess_data_output_shape():
    # Sample data for testing
    sample_data = pd.DataFrame({
        "Airline": ["AA", "DL"],
        "AirportFrom": ["JFK", "ATL"],
        "AirportTo": ["LAX", "SEA"],
        "DayOfWeek": [1, 2],
        "Time": [10, 20],
        "Length": [200, 300],
        "Delay": [0, 1]
    })

    X, y = preprocess_data(sample_data)

    # Check shapes
    assert X.shape[0] == 2
    assert y.shape[0] == 2
    assert "Delay" not in X.columns
