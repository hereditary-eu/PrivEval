#https://docs.sdv.dev/sdmetrics
# Custom implementation of Categorical Generalized-CAP (Correct Attribution Probability)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def calculate_metric(args=None, _real_data=None, _synthetic=None, sensitive_attributes=None, **kwargs):
    """
    Custom implementation of Categorical Generalized-CAP.
    Uses machine learning to predict sensitive attributes from key attributes.
    
    Args:
        _real_data: Real dataset
        _synthetic: Synthetic dataset
        sensitive_attributes: List of sensitive column names
    
    Returns:
        float: Generalized-CAP score (lower is better for privacy)
    """
    # Strict GCAP implementation
    if sensitive_attributes is None:
        sensitive_attributes = []

    # Convert to DataFrames if needed
    if not isinstance(_real_data, pd.DataFrame):
        _real_data = pd.DataFrame(_real_data)
    if not isinstance(_synthetic, pd.DataFrame):
        _synthetic = pd.DataFrame(_synthetic)

    # Get key attributes (non-sensitive attributes)
    key_attributes = [col for col in _real_data.columns if col not in sensitive_attributes]

    if not key_attributes or not sensitive_attributes:
        return 0.0  # Perfect privacy if no keys or sensitives

    # One-hot encode key attributes
    real_key = pd.get_dummies(_real_data[key_attributes], dtype=int)
    syn_key = pd.get_dummies(_synthetic[key_attributes], dtype=int)

    # Align columns (fill missing columns with 0)
    real_key, syn_key = real_key.align(syn_key, join='outer', axis=1, fill_value=0)

    from sklearn.neighbors import NearestNeighbors
    
    real_arr = real_key.to_numpy()
    syn_arr = syn_key.to_numpy()
    # Fit NearestNeighbors on synthetic data
    nn = NearestNeighbors(n_neighbors=1, metric='hamming')
    nn.fit(syn_arr)
    # Find nearest synthetic neighbor for each real record
    distances, indices = nn.kneighbors(real_arr, return_distance=True)
    for sens_att in sensitive_attributes:
        matches = 0
        for i, nearest_idx in enumerate(indices.squeeze()):
            if _real_data[sens_att].iloc[i] == _synthetic[sens_att].iloc[nearest_idx]:
                matches += 1
        gcap_score = matches / len(real_arr) if len(real_arr) > 0 else 1

    return float(gcap_score)