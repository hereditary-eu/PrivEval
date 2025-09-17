import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def calculate_metric(args=None, _real_data=None, _synthetic=None, **kwargs):
    """
    Calculate Nearest Neighbor Adversarial Accuracy following SynthEval implementation.
    
    Based on: https://github.com/schneiderkamplab/syntheval/blob/main/src/syntheval/metrics/privacy/metric_nn_adversarial_accuracy.py
    
    Args:
        _real_data: Real dataset
        _synthetic: Synthetic dataset
        k: Number of nearest neighbors to consider
    
    Returns:
        float: NNAA score (closer to 0.5 is better for synthetic data privacy)
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

        if real_final.empty or syn_final.empty or len(real_final) < 5 or len(syn_final) < 5:
            return 0.5

        # Normalize
        scaler = StandardScaler()
        X_real = scaler.fit_transform(real_final.values)
        X_syn = scaler.transform(syn_final.values)

        # Fit KNN models
        nn_real = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X_real)
        nn_syn = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X_syn)

        # For each real point, distance to nearest real neighbors (excluding self)
        real_to_real = []
        for point in X_real:
            dists = nn_real.kneighbors([point], return_distance=True)[0][0][1]
            real_to_real.append(dists)

        # For each real point, get distance nearest synthetic neighbors
        real_to_syn = []
        for point in X_real:
            dists = nn_syn.kneighbors([point], return_distance=True)[0][0][0]
            real_to_syn.append(dists)

        # For each synthetic point, get mean distance to nearest real neighbors
        syn_to_real = []
        for point in X_syn:
            dists = nn_real.kneighbors([point], return_distance=True)[0][0][0]
            syn_to_real.append(dists)

        # For each synthetic point, get mean distance to nearest synthetic neighbors (excluding self)
        syn_to_syn = []
        for point in X_syn:
            dists = nn_syn.kneighbors([point], return_distance=True)[0][0][1]
            syn_to_syn.append(dists)

        # Adversarial accuracy: proportion of times real_to_syn < real_to_real and syn_to_real < syn_to_syn
        real_adv = np.mean(np.array(real_to_syn) < np.array(real_to_real))
        syn_adv = np.mean(np.array(syn_to_real) < np.array(syn_to_syn))

        # NNAA is the mean of the two adversarial accuracies
        nnaa_score = (real_adv + syn_adv) / 2.0

        return float(nnaa_score)

    except Exception as e:
        print(f"Error in NNAA calculation: {e}")
        return 0.5