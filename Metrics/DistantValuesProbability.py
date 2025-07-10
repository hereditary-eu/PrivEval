import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def calculate_metric(args=None, _real_data=None, _synthetic=None, **kwargs):
    """
    Calculate Distant Values Probability (DVP) following SynthCity implementation.

    DVP measures the inverse probability that a synthetic individual has a large 
    distance to real individuals. It calculates the proportion of real individuals 
    that have normalized distances >= t to their nearest synthetic neighbor.

    DVP(Y, Z) = 1 - (Σ 1[Ṽ_E(y, NN_E(y, Z)) >= t]) / n

    Args:
        _real_data: Real dataset
        _synthetic: Synthetic dataset

    Returns:
        float: DVP score (higher = better privacy) 
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
        
        # Find nearest synthetic neighbor for each real point
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(syn_scaled)
        
        # Get distances from real points to their nearest synthetic neighbors
        distances, _ = nn.kneighbors(real_scaled)
        raw_distances = distances.flatten()
        
        # Apply min-max normalization to distances: Ṽ_E(·,·)
        # This is the key difference - normalize the distances, not the features
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
        
        # Threshold for considering a point "distant"
        t = 0.8 
        # Count how many real points have normalized distance >= t to nearest synthetic
        distant_count = np.sum(normalized_distances >= t)
        
        # Calculate DVP = 1 - (proportion of distant real points)
        proportion_distant = distant_count / len(real_scaled)
        dvp_score = 1 - proportion_distant
        
        return float(dvp_score)
        
    except Exception as e:
        print(f"Error in DistantValuesProbability: {e}")
        return 0.0