#https://synthcity.readthedocs.io/en/latest/metrics.html
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def calculate_metric(args=None, _real_data=None, _synthetic=None, **kwargs):
    """
    Calculate Nearest Synthetic Neighbour Distance (NSND) following SynthCity implementation.
    
    NSND measures the re-identification risk as a normalised average of nearest neighbour 
    distances from real individuals to synthetic individuals.
    
    Args:
        _real_data: Real dataset
        _synthetic: Synthetic dataset
    
    Returns:
        float: NSND score (higher = lower re-identification risk)
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
        
        # Normalize the data features (not the distances)
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_subset)
        syn_scaled = scaler.transform(syn_subset)
        
        # Step 1: Calculate dists_E(Y, Z) - distances from each real point to nearest synthetic
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(syn_scaled)
        
        distances, _ = nn.kneighbors(real_scaled)
        dists_Y_Z = distances.flatten()  # Convert to 1D array
        
        # Step 2: Apply min-max normalization to the distances themselves
        # NSND = Σ((x - min(dists)) / (max(dists) - min(dists))) / n
        
        if len(dists_Y_Z) == 0:
            return 0.0
        
        min_dist = np.min(dists_Y_Z)
        max_dist = np.max(dists_Y_Z)
        
        # Avoid division by zero
        if max_dist == min_dist:
            # If all distances are the same, return 0 (perfect similarity)
            return 0.0
        
        # Apply min-max normalization to distances
        normalized_distances = (dists_Y_Z - min_dist) / (max_dist - min_dist)
        
        # Calculate mean of normalized distances
        nsnd_score = np.mean(normalized_distances)
        
        return 1 - float(nsnd_score)
        
    except Exception as e:
        print(f"Error in NearestSynNeighborDistance: {e}")
        return 0.0