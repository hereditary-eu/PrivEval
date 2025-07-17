import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def calculate_metric(args=None, _real_data=None, _synthetic=None, **kwargs):
    """
    Calculate Close Values Probability (CVP) following SynthCity implementation.
    
    CVP measures the probability that an adversary is able to distinguish which real 
    individual a synthetic is generated from, where it is assumed that the adversary 
    only has access to synthetic data.

    CVP(Y, Z) = (Σ 1[^D_E(y, NN_E(y, Z)) < t]) / n

    Args:
        _real_data: Real dataset
        _synthetic: Synthetic dataset
    
    Returns:
        float: CVP score (higher = worse privacy, 1 = all real individuals re-identifiable)
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
        
        # Normalize the data features for fair distance computation
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_subset)
        syn_scaled = scaler.transform(syn_subset)
        
        # Find nearest synthetic neighbor for each real point (correct direction for CVP)
        # NN_E(y, Z) = argmin_{z∈Z} D_E(y, z)
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(syn_scaled)  # Fit on synthetic data
        
        # Get distances from real points to their nearest synthetic neighbors
        distances, _ = nn.kneighbors(real_scaled)  # Query with real data
        raw_distances = distances.flatten()

        # Apply min-max normalization to distances: ^D_E(y, z)
        # ^D_E(y, z) = (D_E(y, z) - D_min) / (D_max - D_min)
        if len(raw_distances) > 0:
            min_dist = np.min(raw_distances)
            max_dist = np.max(raw_distances)
            
            if max_dist > min_dist:
                # Min-max normalize distances to [0, 1]
                normalized_distances = (raw_distances - min_dist) / (max_dist - min_dist)
            else:
                # All distances are the same
                normalized_distances = np.zeros_like(raw_distances)
        else:
            return 0.0
        
        # Threshold for considering a point "close"
        t = 0.2  
        # Count how many real points have normalized distance < t to nearest synthetic
        # 1[^D_E(y, NN_E(y, Z)) < t]
        close_count = np.sum(normalized_distances < t)
        
        # Calculate CVP = (Σ 1[condition]) / n
        cvp_score = close_count / len(real_scaled)
        
        return float(cvp_score)
        
    except Exception as e:
        print(f"Error in CloseValuesProbability: {e}")
        return 0.0