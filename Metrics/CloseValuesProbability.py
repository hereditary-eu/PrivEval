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