import pytest
from src.train import process_data
import pandas as pd
import numpy as np

def test_rul_calculation():
    # Create fake dummy data
    data = {
        'unit': [1, 1, 1],
        'time': [1, 2, 3], # Max time is 3, so RUL should be 2, 1, 0
        'os1': [0,0,0], 'os2': [0,0,0], 'os3': [0,0,0]
    }
    # Add fake sensor columns s1...s21
    for i in range(1, 22):
        data[f's{i}'] = np.random.rand(3)
        
    df = pd.DataFrame(data)
    
    # Run our function
    X, y = process_data(df)
    
    # Check if the math works (Expect RUL to be [2, 1, 0])
    expected_rul = [2, 1, 0]
    assert y.tolist() == expected_rul