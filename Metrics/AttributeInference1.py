#Code modified from https://github.com/yy6linda/synthetic-ehr-benchmarking 

#Risk of an attacker being able to infer real, sensitive attributes

import os
import numpy as np
import os.path
import pandas as pd
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors


def calculate_metric(args, _real_data, _synthetic, sensitive_attributes=None):
    real_data = deepcopy(_real_data)
    syn_data = deepcopy(_synthetic)

    num_samples = len(real_data)

    #Get key attributes
    key_attributes = []
    for column in real_data.columns:
            if column not in sensitive_attributes:
                key_attributes.append(column)

    cont_attributes = []
    for column in real_data.columns:
            if real_data[column].dtype == np.float64:
                cont_attributes.append(column)
    
    # Only remove if cont_attributes exists and is not empty
    if cont_attributes:
        if cont_attributes[0] in key_attributes:
            key_attributes.remove(cont_attributes[0])
    
    #reorder columns
    real_reordered = real_data[key_attributes]
    syn_reordered = syn_data[key_attributes]

    all_data = pd.concat([real_reordered, syn_reordered])
    all_data_no_cont = pd.get_dummies(all_data, dtype=int)
    
    real_data_no_cont = all_data_no_cont[:real_data.shape[0]]
    syn_data_no_cont = all_data_no_cont[real_data.shape[0]:]
    
    # Handle case where there are no continuous attributes
    if cont_attributes:
        real_data_no_sens = pd.concat([real_data_no_cont, real_data[cont_attributes[0]]], axis=1)
        syn_data_no_sens = pd.concat([syn_data_no_cont, syn_data[cont_attributes[0]]], axis=1)
    else:
        real_data_no_sens = real_data_no_cont
        syn_data_no_sens = syn_data_no_cont
    
    # Find nearest neighbors using key attributes only
    estimator = NearestNeighbors(n_neighbors=1).fit(
        syn_data_no_sens.to_numpy().reshape(len(syn_data_no_sens), -1)
        )
    idxs = estimator.kneighbors(
                real_data_no_sens.to_numpy().reshape(len(real_data_no_sens), -1), 1, return_distance=False
            ).squeeze()
    
    # Calculate TP, FP for each real-synthetic pair
    predictions = []
    actual = []
    
    for i in range(len(idxs)):
        nearest_syn_idx = idxs[i]
        
        for sens_att in sensitive_attributes:
            # Get actual sensitive attribute value
            actual_value = real_data[sens_att].iloc[i]
            
            # Get predicted sensitive attribute value from nearest synthetic neighbor
            predicted_value = syn_data[sens_att].iloc[nearest_syn_idx]
            
            # Check if prediction is correct
            if sens_att in cont_attributes:
                # For continuous attributes, use 10% tolerance
                if abs(predicted_value - actual_value) <= 0.1 * abs(actual_value):
                    predictions.append(1)  # Correct prediction
                else:
                    predictions.append(0)  # Incorrect prediction
            else:
                # For categorical attributes, exact match
                if predicted_value == actual_value:
                    predictions.append(1)  # Correct prediction
                else:
                    predictions.append(0)  # Incorrect prediction
                    
            actual.append(1)  # All are actual positive cases
    
    # Calculate overall TP, FP, FN
    TP = sum(predictions)
    FP = len(predictions) - TP
    FN = 0  # No false negatives in this formulation
    
    # Calculate F1 score (harmonic mean of precision and recall)
    if TP > 0:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        f1_score = 0
    
    # Calculate entropy-based weights for each unique record pattern
    row_counts = real_data.value_counts().reset_index(name="Count")
    real_data_with_count = real_data.merge(row_counts, how='left')
    
    # Calculate total entropy for normalization
    unique_counts = real_data.value_counts()
    total_probs = unique_counts / len(real_data)
    total_entropy = -np.sum(total_probs * np.log(total_probs + 1e-8))
    
    total_air = 0
    
    for i in range(len(real_data)):
        # Probability of this specific record pattern
        prob = real_data_with_count['Count'].iloc[i] / len(real_data)
        
        # Weight based on rarity (inverse probability normalized by entropy)
        if total_entropy > 0:
            weight = -prob * np.log(prob + 1e-8) / total_entropy
        else:
            weight = 1.0 / len(real_data)
        
        # Add weighted F1 score
        total_air += weight * f1_score
    
    # Normalize by total number of samples
    air = total_air
    
    return float(air)