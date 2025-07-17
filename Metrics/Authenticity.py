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
        
        # Set up nearest neighbor models
        # For finding nearest real neighbors (need 2 to exclude self)
        nbrs_real = NearestNeighbors(n_neighbors=2, metric='euclidean')
        nbrs_real.fit(real_scaled)
        
        # For finding nearest synthetic neighbors
        nbrs_synth = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nbrs_synth.fit(syn_scaled)
        
        # Get distances from real points to nearest real neighbors (excluding self)
        real_to_real_distances, _ = nbrs_real.kneighbors(real_scaled)
        real_to_real = real_to_real_distances[:, 1]  # Distance to nearest real (excluding self at index 0)
        
        # Get distances from real points to nearest synthetic neighbors
        real_to_synth_distances, real_to_synth_args = nbrs_synth.kneighbors(real_scaled)
        real_to_synth = real_to_synth_distances.squeeze()  # Distance to nearest synthetic
        
        # Calculate authenticity: count cases where nearest real is closer than nearest synthetic
        authentic_mask = real_to_real < real_to_synth
        authenticity = np.mean(authentic_mask)
        
        return 1 - float(authenticity)
        
    except Exception as e:
        print(f"Error in Authenticity calculation: {e}")
        return 0.0