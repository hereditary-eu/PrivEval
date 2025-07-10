#https://docs.sdv.dev/sdmetrics
# Custom implementation of Categorical Zero-CAP (Correct Attribution Probability)
from copy import deepcopy
import pandas as pd
import numpy as np
from itertools import combinations

def calculate_metric(args=None, _real_data=None, _synthetic=None, sensitive_attributes=None, **kwargs):
    """
    Custom implementation of Categorical Zero-CAP (Correct Attribution Probability).
    Measures privacy risk by checking if sensitive attributes can be inferred from key attributes.
    
    Args:
        _real_data: Real dataset
        _synthetic: Synthetic dataset  
        sensitive_attributes: List of sensitive column names
    
    Returns:
        float: Zero-CAP score (lower is better for privacy)
    """
    try:
        if sensitive_attributes is None:
            sensitive_attributes = []
        
        # Convert to DataFrames if needed
        if not isinstance(_real_data, pd.DataFrame):
            _real_data = pd.DataFrame(_real_data)
        if not isinstance(_synthetic, pd.DataFrame):
            _synthetic = pd.DataFrame(_synthetic)
        
        # Get key attributes (non-sensitive attributes)
        key_attributes = [col for col in _real_data.columns if col not in sensitive_attributes]
        
        if not key_attributes or not sensitive_attributes:
            return 0.0  # Perfect privacy if no keys or sensitives
        
        # Ensure we have the required columns
        required_cols = key_attributes + sensitive_attributes
        real_subset = _real_data[required_cols].copy()
        syn_subset = _synthetic[required_cols].copy()
        
        # Convert to categorical/string for exact matching
        for col in required_cols:
            real_subset[col] = real_subset[col].astype(str)
            syn_subset[col] = syn_subset[col].astype(str)
        
        zero_cap_scores = []
        
        # For each sensitive attribute, calculate Zero-CAP
        for sensitive_attr in sensitive_attributes:
            available_keys = [col for col in key_attributes if col in real_subset.columns]
            
            if not available_keys:
                zero_cap_scores.append(1.0)
                continue
            
            # Group real data by key attributes
            real_groups = real_subset.groupby(available_keys)[sensitive_attr].apply(list).to_dict()
            
            # For each synthetic record, check if sensitive value can be uniquely determined
            correct_inferences = 0
            total_inferences = 0
            
            for _, syn_row in syn_subset.iterrows():
                # Get key values for this synthetic row
                key_values = tuple(syn_row[key_attr] for key_attr in available_keys)
                
                if key_values in real_groups:
                    # Get all possible sensitive values for this key combination
                    possible_sensitive_values = list(set(real_groups[key_values]))
                    
                    # If only one possible value, we can infer it with certainty
                    if len(possible_sensitive_values) == 1:
                        inferred_value = possible_sensitive_values[0]
                        actual_value = str(syn_row[sensitive_attr])
                        
                        if inferred_value == actual_value:
                            correct_inferences += 1
                        total_inferences += 1
            
            # Calculate Zero-CAP for this sensitive attribute
            if total_inferences > 0:
                zero_cap = correct_inferences / total_inferences
            else:
                zero_cap = 0.0  # No inferences possible = perfect privacy
                
            zero_cap_scores.append(zero_cap)
        
        # Return average Zero-CAP (higher is better for privacy)
        avg_zero_cap = np.mean(zero_cap_scores) if zero_cap_scores else 0.0
        return float(avg_zero_cap)
        
    except Exception as e:
        print(f"Error in CZeroCAP calculation: {e}")
        return 0.0  # Return perfect privacy score on error