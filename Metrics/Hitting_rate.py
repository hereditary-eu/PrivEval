# https://github.com/schneiderkamplab/syntheval/blob/main/src/syntheval/metrics/privacy/metric_hitting_rate.py

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def calculate_metric(args=None, _real_data=None, _synthetic=None, **kwargs):
    """
    Calculate Hitting Rate (HitR) following SynthEval specification.
    
    HitR estimates the risk of determining whether an individual contributed their data 
    to the real dataset by finding synthetic individuals with the same categorical values 
    and continuous values within a close range.
    
    Args:
        _real_data: Real dataset
        _synthetic: Synthetic dataset
    
    Returns:
        float: Hitting Rate score (closer to 1 = higher risk)
    """
    try:
        # Convert to DataFrames if needed
        if not isinstance(_real_data, pd.DataFrame):
            _real_data = pd.DataFrame(_real_data)
        if not isinstance(_synthetic, pd.DataFrame):
            _synthetic = pd.DataFrame(_synthetic)
        
        # Get common columns
        common_cols = list(set(_real_data.columns) & set(_synthetic.columns))
        if not common_cols:
            return 0.0
        
        real_subset = _real_data[common_cols].copy()
        syn_subset = _synthetic[common_cols].copy()
        
        if real_subset.empty or syn_subset.empty:
            return 0.0
        
        # Handle missing values (use mode for categorical, mean for numerical)
        for col in common_cols:
            if real_subset[col].dtype in ['object', 'category']:
                mode_val = real_subset[col].mode().iloc[0] if not real_subset[col].mode().empty else 'unknown'
                real_subset[col] = real_subset[col].fillna(mode_val)
                syn_subset[col] = syn_subset[col].fillna(mode_val)
            else:
                mean_val = real_subset[col].mean()
                real_subset[col] = real_subset[col].fillna(mean_val)
                syn_subset[col] = syn_subset[col].fillna(mean_val)
        
        # Separate categorical and continuous attributes
        # A_C = {a ∈ A | t(a) = Σ ∨ t(a) = {0,1}} (categorical/binary)
        # A_R = {a ∈ A | t(a) = R} (continuous/numerical)
        
        categorical_cols = []
        continuous_cols = []
        
        for col in common_cols:
            if real_subset[col].dtype in ['object', 'category', 'bool']:
                categorical_cols.append(col)
            elif real_subset[col].dtype in ['int64', 'float64']:
                # Check if it's binary (0,1) or truly continuous
                unique_vals = real_subset[col].unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                    categorical_cols.append(col)
                else:
                    continuous_cols.append(col)
        
        # Calculate thresholds for continuous attributes
        # h(a) = (max(y_i[a]) - min(y_i[a])) / 30
        thresholds = {}
        for col in continuous_cols:
            col_min = real_subset[col].min()
            col_max = real_subset[col].max()
            thresholds[col] = (col_max - col_min) / 30 if col_max != col_min else 0.0
        
        # Group synthetic data by categorical columns for fast lookup
        if categorical_cols:
            syn_groups = syn_subset.groupby(categorical_cols)
        else:
            syn_groups = None

        def is_hit(real_row):
            # Get candidate synthetic rows
            if categorical_cols:
                key = tuple(real_row[col] for col in categorical_cols)
                try:
                    candidates = syn_groups.get_group(key)
                except KeyError:
                    return 0
            else:
                candidates = syn_subset

            # If no continuous columns, any candidate is a hit
            if not continuous_cols:
                return int(len(candidates) > 0)

            # For continuous columns, check thresholds vectorized
            mask = np.ones(len(candidates), dtype=bool)
            for col in continuous_cols:
                real_val = real_row[col]
                threshold = thresholds[col]
                mask &= (candidates[col] >= real_val - threshold) & (candidates[col] <= real_val + threshold)
            return int(mask.any())

        # Parallelize the hit calculation
        hit_count = sum(
            Parallel(n_jobs=-1, backend="loky")(
                delayed(is_hit)(row)
                for _, row in real_subset.iterrows()
            )
        )

        hitting_rate = hit_count / len(real_subset) if len(real_subset) > 0 else 0.0
        return float(hitting_rate)

    except Exception as e:
        print(f"Error in Hitting_rate calculation: {e}")
        return 0.0