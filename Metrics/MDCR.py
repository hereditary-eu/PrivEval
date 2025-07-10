# https://github.com/schneiderkamplab/syntheval/blob/main/src/syntheval/metrics/privacy

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def calculate_metric(args=None, _real_data=None, _synthetic=None, **kwargs):
    """
    Calculate Median Distance to Closest Record (MDCR).
    
    MDCR estimates the risk of re-identification while only having access to synthetic data,
    as an overall score deduced from the median as a measure of distance.
    
    MDCR(Y, Z) = σ(med(dists_E(Y, Z)) / med(dists_E(Y, Y)))
    
    Args:
        _real_data: Real dataset
        _synthetic: Synthetic dataset
    
    Returns:
        float: MDCR score (closer to 0 = lower re-identification risk)
    """
    try:
        # Convert to DataFrames if needed
        if not isinstance(_real_data, pd.DataFrame):
            _real_data = pd.DataFrame(_real_data)
        if not isinstance(_synthetic, pd.DataFrame):
            _synthetic = pd.DataFrame(_synthetic)
        
        # Get common columns and select only numeric columns
        common_cols = list(set(_real_data.columns) & set(_synthetic.columns))
        real_subset = _real_data[common_cols].select_dtypes(include=[np.number])
        syn_subset = _synthetic[common_cols].select_dtypes(include=[np.number])
        
        if real_subset.empty or syn_subset.empty:
            return 0.0
        
        # Handle missing values
        real_subset = real_subset.fillna(real_subset.mean())
        syn_subset = syn_subset.fillna(syn_subset.mean())
        
        # Normalize the data
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_subset)
        syn_scaled = scaler.transform(syn_subset)
        
        # Calculate med(dists_E(Y, Z)) - median distance from synthetic to real
        nn_real = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn_real.fit(real_scaled)
        distances_syn_to_real, _ = nn_real.kneighbors(syn_scaled)
        med_dists_Y_Z = np.median(distances_syn_to_real.flatten())
        
        # Calculate med(dists_E(Y, Y)) - median distance within real data
        if len(real_scaled) < 2:
            return 0.0
        
        nn_real_internal = NearestNeighbors(n_neighbors=2, metric='euclidean')  # 2 to exclude self
        nn_real_internal.fit(real_scaled)
        distances_real_to_real, _ = nn_real_internal.kneighbors(real_scaled)
        # Take second column (index 1) to exclude self-distance (which is 0)
        real_internal_distances = distances_real_to_real[:, 1]
        med_dists_Y_Y = np.median(real_internal_distances)
        
        # Avoid division by zero
        if med_dists_Y_Y == 0:
            return 0.0
        
        # Calculate ratio
        ratio = med_dists_Y_Z / med_dists_Y_Y
        
        # Apply sigmoid function
        mdcr_score = sigmoid(ratio)
        
        return float(mdcr_score)
        
    except Exception as e:
        print(f"Error in MDCR calculation: {e}")
        return 0.0

def sigmoid(x):
    """Sigmoid function to map ratio to [0, 1] range"""
    return 1 / (1 + np.exp(-x))


