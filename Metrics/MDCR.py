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

        if cat_cols:
            # One-hot encode categorical columns
            all_cat = pd.concat([real_subset[cat_cols], syn_subset[cat_cols]], axis=0)
            all_cat_encoded = pd.get_dummies(all_cat, dtype=int)
            real_cat_encoded = all_cat_encoded.iloc[:len(real_subset), :]
            syn_cat_encoded = all_cat_encoded.iloc[len(real_subset):, :]

            # Concatenate numeric and encoded categorical columns
            real_final = pd.concat([real_subset[num_cols].reset_index(drop=True), real_cat_encoded.reset_index(drop=True)], axis=1)
            syn_final = pd.concat([syn_subset[num_cols].reset_index(drop=True), syn_cat_encoded.reset_index(drop=True)], axis=1)
        else:
            # Only numeric columns
            real_final = real_subset[num_cols].reset_index(drop=True)
            syn_final = syn_subset[num_cols].reset_index(drop=True)

        if real_final.empty or syn_final.empty:
            return 0.0

        # Normalize the data
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_final)
        syn_scaled = scaler.transform(syn_final)
        
        # Calculate med(dists_E(Y, Z)) - median distance from each real to nearest synthetic
        nn_syn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn_syn.fit(syn_scaled)
        distances_real_to_syn, _ = nn_syn.kneighbors(real_scaled)
        med_dists_Y_Z = np.median(distances_real_to_syn.flatten())

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
        mdcr_score = 1 - sigmoid(ratio)

        return float(mdcr_score)
        
    except Exception as e:
        print(f"Error in MDCR calculation: {e}")
        return 0.0

def sigmoid(x):
    """Sigmoid function to map ratio to [0, 1] range"""
    return 1 / (1 + np.exp(-x))


