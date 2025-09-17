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

    DVP(Y, Z) = 1 - (Σ 1[^D_E(y, NN_E(y, Z)) >= t]) / n

    Args:
        _real_data: Real dataset
        _synthetic: Synthetic dataset

    Returns:
        float: DVP score (lower = better privacy) 
    """
    try:
        # Convert to DataFrames if needed
        if not isinstance(_real_data, pd.DataFrame):
            _real_data = pd.DataFrame(_real_data)
        if not isinstance(_synthetic, pd.DataFrame):
            _synthetic = pd.DataFrame(_synthetic)
        
        # Get common columns
        common_cols = list(set(_real_data.columns) & set(_synthetic.columns))
        real_subset = _real_data[common_cols].copy()
        syn_subset = _synthetic[common_cols].copy()

        # Separate numeric and categorical columns
        num_cols = real_subset.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [col for col in common_cols if col not in num_cols]

        # Handle missing values
        for col in num_cols:
            real_subset[col] = real_subset[col].fillna(real_subset[col].mean())
            syn_subset[col] = syn_subset[col].fillna(syn_subset[col].mean())
        for col in cat_cols:
            real_subset[col] = real_subset[col].fillna('missing').astype(str)
            syn_subset[col] = syn_subset[col].fillna('missing').astype(str)

        # One-hot encode categorical columns
        all_cat = pd.concat([real_subset[cat_cols], syn_subset[cat_cols]], axis=0)
        all_cat_encoded = pd.get_dummies(all_cat, dtype=int)
        real_cat_encoded = all_cat_encoded.iloc[:len(real_subset), :]
        syn_cat_encoded = all_cat_encoded.iloc[len(real_subset):, :]

        # Concatenate numeric and encoded categorical columns
        real_final = pd.concat([real_subset[num_cols].reset_index(drop=True), real_cat_encoded.reset_index(drop=True)], axis=1)
        syn_final = pd.concat([syn_subset[num_cols].reset_index(drop=True), syn_cat_encoded.reset_index(drop=True)], axis=1)

        if real_final.empty or syn_final.empty:
            return 0.0

        # Normalize the data features for fair distance computation
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_final)
        syn_scaled = scaler.transform(syn_final)
        
        # Find nearest synthetic neighbor for each real point
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(syn_scaled)
        
        # Get distances from real points to their nearest synthetic neighbors
        distances, _ = nn.kneighbors(real_scaled)
        raw_distances = distances.flatten()

        # Apply min-max normalization to distances: ^D_E(·,·)
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