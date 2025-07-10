#Authenticity metric from SynthCity https://github.com/vanderschaarlab/synthcity
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def calculate_metric(args=None, _real_data=None, _synthetic=None, alpha=0.95, **kwargs):
    """
    Args:
        _real_data: Real dataset
        _synthetic: Synthetic dataset
        alpha: Coverage level (default 0.95 for 95% coverage)
    
    Returns:
        float: 1 - Authenticity score (lower is better)
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
        
        if real_subset.empty or syn_subset.empty or len(real_subset) < 2 or len(syn_subset) < 2:
            return 0.0
        
        # Handle missing values
        real_subset = real_subset.fillna(real_subset.mean())
        syn_subset = syn_subset.fillna(syn_subset.mean())
        
        # Normalize the data
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_subset)
        syn_scaled = scaler.transform(syn_subset)
        
        # Calculate k-nearest neighbor distances in real data to establish the support
        k = min(5, len(real_scaled) - 1)
        if k < 1:
            return 0.0
        
        # Fit KNN on real data
        real_knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        real_knn.fit(real_scaled)
        
        # Get distances to k-th nearest neighbor for each real point
        real_distances, _ = real_knn.kneighbors(real_scaled)
        real_kth_distances = real_distances[:, -1]  # k-th nearest neighbor distance
        
        # Calculate the alpha-th percentile of these distances as the support radius
        support_radius = np.percentile(real_kth_distances, alpha * 100)
        
        # For each synthetic point, check if it's within the support
        # (i.e., within support_radius of at least one real point)
        syn_knn_distances, _ = real_knn.kneighbors(syn_scaled)
        syn_nearest_distances = syn_knn_distances[:, 0]  # Distance to nearest real point
        
        # Count synthetic points within the support
        within_support = np.sum(syn_nearest_distances <= support_radius)
        
        # Calculate authenticity as the fraction of synthetic points within support
        authenticity = within_support / len(syn_scaled) if len(syn_scaled) > 0 else 0.0
        
        return 1 - float(authenticity)
        
    except Exception as e:
        print(f"Error in Authenticity calculation: {e}")
        return 0.0