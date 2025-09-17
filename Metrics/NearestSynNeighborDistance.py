#https://synthcity.readthedocs.io/en/latest/metrics.html
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def calculate_metric(args=None, _real_data=None, _synthetic=None, **kwargs):
    """
    Calculate Nearest Synthetic Neighbour Distance (NSND) following a modification of the SynthCity implementation.
    
    NSND measures the re-identification risk as a normalised average of nearest neighbour 
    distances from real individuals to synthetic individuals.
    
    Args:
        _real_data: Real dataset
        _synthetic: Synthetic dataset
    
    Returns:
        float: NSND score (higher = higher re-identification risk)
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

        # Normalize the data features (not the distances)
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_final)
        syn_scaled = scaler.transform(syn_final)
        
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