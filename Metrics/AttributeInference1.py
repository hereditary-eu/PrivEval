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
    
    key_attributes.remove(cont_attributes[0])
    
    #reorder columns
    real_reordered = real_data[key_attributes]
    syn_reordered = syn_data[key_attributes]

    all_data = pd.concat([real_reordered, syn_reordered])
    all_data_no_cont = pd.get_dummies(all_data, dtype=int)
    
    #convert data
    # real_data_no_cont = pd.get_dummies(real_reordered, dtype=int)
    # syn_data_no_cont = pd.get_dummies(syn_reordered, dtype=int)
    
    
    
    real_data_no_cont = all_data_no_cont[:real_data.shape[0]]
    syn_data_no_cont = all_data_no_cont[real_data.shape[0]:]
    
    real_data_no_sens = pd.concat([real_data_no_cont, real_data[cont_attributes[0]]], axis=1)
    syn_data_no_sens = pd.concat([syn_data_no_cont, syn_data[cont_attributes[0]]], axis=1)
    
    
    
    estimator = NearestNeighbors(n_neighbors=1).fit(
        syn_data_no_sens.to_numpy().reshape(len(syn_data_no_sens), -1)
        )
    idxs = estimator.kneighbors(
                real_data_no_sens.to_numpy().reshape(len(real_data_no_sens), -1), 1, return_distance=False
            ).squeeze()
    
    TP = 0
    FP = 0
    FN = 0
    for sens_att in sensitive_attributes:
        for i in range(len(idxs)):
            if sens_att in cont_attributes:
                if abs(syn_data[sens_att][i] - real_data[sens_att][idxs[i]]) <= 0.1 * real_data[sens_att][idxs[i]]:
                    TP += 1
                else:
                    FP += 1
            else:
                if syn_data[sens_att][i] == real_data[sens_att][idxs[i]]:
                    TP += 1
                else:
                    FP += 1
    
    
    total_air = 0
    row_counts = real_data.value_counts().reset_index(name="Count")
    # Merge those counts back onto real_data
    real_data_with_count = real_data.merge(row_counts, how='left')
    prob = 0
    entropy = 0
    weight = 0
    total_air = 0
    
    for i in range(len(real_data)):
        prob = real_data_with_count['Count'].iloc[i] / len(real_data)
        entropy = prob * np.log(prob)
        weight = (prob * np.log(prob)) / entropy if entropy != 0 else 0

        if TP > 0:
            total_air += weight * 2 * (TP / (TP + FP)) * (TP / (TP + FN)) / ((TP / (TP + FP)) + (TP / (TP + FN)))
        else:
            total_air += 0
    # for i in range(len(real_data)):
    #     for j in range(len(syn_data)):
    #         total_prop = 
            
    air = total_air/num_samples      
    return air