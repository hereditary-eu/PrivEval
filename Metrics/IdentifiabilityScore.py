import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder

def calculate_metric(args=None, _real_data=None, _synthetic=None, **kwargs):
    """
    Calculate Identifiability Score following SynthCity implementation.
    
    Estimates the risk of re-identifying real individuals by checking if synthetic
    points are closer to real points than the second nearest real point.
    Uses entropy-based attribute weighting.
    
    Args:
        _real_data: Real dataset
        _synthetic: Synthetic dataset
    
    Returns:
        float: Identifiability score (higher = more re-identification risk)
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
        
        if real_subset.empty or syn_subset.empty or len(real_subset) < 3:
            return 0.0
        
        # Handle categorical columns - encode them first
        categorical_cols = real_subset.select_dtypes(include=['object', 'category']).columns
        numerical_cols = real_subset.select_dtypes(include=[np.number]).columns
        
        # Encode categorical columns
        encoded_real = real_subset.copy()
        encoded_syn = syn_subset.copy()
        
        for col in categorical_cols:
            # Combine both datasets to ensure consistent encoding
            combined_values = pd.concat([real_subset[col], syn_subset[col]], axis=0)
            le = LabelEncoder()
            le.fit(combined_values)
            
            encoded_real[col] = le.transform(real_subset[col])
            encoded_syn[col] = le.transform(syn_subset[col])
        
        # Handle missing values
        encoded_real = encoded_real.fillna(encoded_real.mean())
        encoded_syn = encoded_syn.fillna(encoded_syn.mean())
        
        # Calculate entropy-based weights for each attribute
        weights = []
        epsilon = 1e-8  # Small constant to avoid division by zero
        
        for col in common_cols:
            # Calculate entropy H(Y[a]) for attribute a
            if col in categorical_cols:
                # For categorical attributes (now encoded as numerical)
                value_counts = encoded_real[col].value_counts(normalize=True)
                entropy = -np.sum(value_counts * np.log(value_counts + epsilon))
            else:
                # For numerical attributes, discretize first
                try:
                    # Use quantile-based discretization
                    discrete_values = pd.qcut(encoded_real[col], q=min(10, len(encoded_real[col].unique())), 
                                            duplicates='drop')
                    value_counts = discrete_values.value_counts(normalize=True)
                    entropy = -np.sum(value_counts * np.log(value_counts + epsilon))
                except:
                    # Fallback: treat as single category if discretization fails
                    entropy = 0.0
            
            # Weight is inverse of entropy: w_j = 1/H(Y[a_j])
            weight = 1.0 / (entropy + epsilon)
            weights.append(weight)
        
        weights = np.array(weights)
        
        # Now all data is numerical - normalize it
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(encoded_real)
        syn_scaled = scaler.transform(encoded_syn)
        
        # Apply weighted transformation: ŷ_{i,j} = y_{i,j} / (w_j + ε)
        weights_expanded = weights.reshape(1, -1)
        real_weighted = real_scaled / (weights_expanded + epsilon)
        syn_weighted = syn_scaled / (weights_expanded + epsilon)
        
        # Find nearest neighbors using weighted data
        # For each real point, find:
        # 1. Nearest synthetic point
        # 2. Second nearest real point
        
        nn_syn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn_syn.fit(syn_weighted)

        nn_real = NearestNeighbors(n_neighbors=2, metric='euclidean')  # Need 2 to get 2nd nearest (excluding self)
        nn_real.fit(real_weighted)
        
        identifiable_count = 0
        
        for i, real_point in enumerate(real_weighted):
            # Distance to nearest synthetic point
            dist_to_syn, _ = nn_syn.kneighbors([real_point])
            nearest_syn_dist = dist_to_syn[0][0]
            
            # Distance to second nearest real point (excluding self)
            dist_to_real, _ = nn_real.kneighbors([real_point])
            # dist_to_real[0] contains [self, nearest_real, second_nearest_real]
            second_nearest_real_dist = dist_to_real[0][1]  # Second element is nearest (excluding self)
            
            # Check if synthetic point is closer than second nearest real point
            # I(Y, Z) counts cases where D_E(y_i, NN_E(y_i, Z)) < D_E(y_i, 2NN_E(y_i, Y))
            if nearest_syn_dist < second_nearest_real_dist:
                identifiable_count += 1
        
        # Calculate identifiability score as proportion
        identifiability_score = identifiable_count / len(real_weighted)
        
        return float(identifiability_score)
        
    except Exception as e:
        print(f"Error in IdentifiabilityScore: {e}")
        return 0.0