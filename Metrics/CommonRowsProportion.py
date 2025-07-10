import pandas as pd
import numpy as np

def calculate_metric(args=None, _real_data=None, _synthetic=None, **kwargs):
    """
    Calculate Common Rows Proportion (CRP) following SynthCity implementation.
    
    CRP measures the risk of re-identification as the probability of a synthetic 
    individual being exactly the same as a real individual.
    
    CRP(Y, Z) = |Y ∩ Z| / (|Y| + 1e-8)
    
    Args:
        _real_data: Real dataset
        _synthetic: Synthetic dataset
    
    Returns:
        float: Proportion of real individuals that have identical synthetic counterparts
    """
    try:
        # Convert to DataFrames if needed
        if not isinstance(_real_data, pd.DataFrame):
            _real_data = pd.DataFrame(_real_data)
        if not isinstance(_synthetic, pd.DataFrame):
            _synthetic = pd.DataFrame(_synthetic)
        
        # Ensure same columns
        common_cols = list(set(_real_data.columns) & set(_synthetic.columns))
        real_subset = _real_data[common_cols]
        syn_subset = _synthetic[common_cols]
        
        # Convert to string representation for exact matching
        real_strings = real_subset.astype(str).apply(lambda x: '|'.join(x), axis=1)
        syn_strings = syn_subset.astype(str).apply(lambda x: '|'.join(x), axis=1)
        
        # Find intersection |Y ∩ Z|
        real_set = set(real_strings)
        syn_set = set(syn_strings)
        intersection_size = len(real_set & syn_set)
        
        # Calculate CRP = |Y ∩ Z| / (|Y| + 1e-8)
        crp = intersection_size / (len(real_subset) + 1e-8)
        
        return float(crp)
        
    except Exception as e:
        print(f"Error in CommonRowsProportion: {e}")
        return 0.0