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

        # Get common numeric columns
        common_cols = list(set(_real_data.columns) & set(_synthetic.columns))
        real_subset = _real_data[common_cols].select_dtypes(include=[np.number])
        syn_subset = _synthetic[common_cols].select_dtypes(include=[np.number])

        if real_subset.empty or syn_subset.empty or len(real_subset) < 5 or len(syn_subset) < 5:
            return 0.5

        # Fill missing values
        real_subset = real_subset.fillna(real_subset.mean())
        syn_subset = syn_subset.fillna(syn_subset.mean())

        # Normalize
        scaler = StandardScaler()
        X_real = scaler.fit_transform(real_subset.values)
        X_syn = scaler.transform(syn_subset.values)        

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

        # For each synthetic point, get mean distance to k nearest real neighbors
        syn_to_real = []
        for point in X_syn:
            dists = nn_real.kneighbors([point], return_distance=True)[0][0][0]
            syn_to_real.append(dists)

        # For each synthetic point, get mean distance to k nearest synthetic neighbors (excluding self)
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