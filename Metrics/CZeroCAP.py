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
    # Strict ZCAP implementation
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

    # Convert to string for exact matching
    for col in required_cols:
        real_subset[col] = real_subset[col].astype(str)
        syn_subset[col] = syn_subset[col].astype(str)

    zero_cap_scores = []
    for sensitive_attr in sensitive_attributes:
        available_keys = [col for col in key_attributes if col in real_subset.columns]
        if not available_keys:
            zero_cap_scores.append(1.0)
            continue
        # Merge real and synthetic on key attributes
        merged = pd.merge(
            real_subset[available_keys + [sensitive_attr]].reset_index(),
            syn_subset[available_keys + [sensitive_attr]].reset_index(),
            on=available_keys,
            suffixes=('_real', '_syn')
        )
        # For each real record, count matches and sensitive matches
        match_counts = merged.groupby('index_real').size()
        sens_match_counts = merged[merged[sensitive_attr + '_real'] == merged[sensitive_attr + '_syn']].groupby('index_real').size()
        # Calculate zcap for each real record
        zcap_values = []
        for idx in real_subset.index:
            total_matches = match_counts.get(idx, 0)
            sens_matches = sens_match_counts.get(idx, 0)
            if total_matches == 0:
                zcap = 0.0
            else:
                zcap = sens_matches / total_matches
            zcap_values.append(zcap)
        zero_cap = np.mean(zcap_values) if zcap_values else 0.0
        zero_cap_scores.append(zero_cap)

    # Return average ZCAP
    avg_zero_cap = np.mean(zero_cap_scores) if zero_cap_scores else 0.0
    return float(avg_zero_cap)