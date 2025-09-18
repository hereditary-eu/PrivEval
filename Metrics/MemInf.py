### Metric from https://github.com/schneiderkamplab/syntheval

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from copy import deepcopy


def calculate_metric(args=None, _real_data=None, _synthetic=None, num_eval_iter=5):
    """
    Calculate Membership Inference Risk (MIR) following the exact description.
    
    MIR estimates the risk of determining whether an individual contributed their data 
    to the real dataset while assuming that the adversary has access to synthetic data 
    and the person of interest.
    
    Args:
        _real_data: Real dataset (labeled as 1)
        _synthetic: Synthetic dataset (labeled as 0)
        num_eval_iter: Number of evaluation iterations
    
    Returns:
        float: MIR score (recall of real data classification)
    """
    try:
        real_data = deepcopy(_real_data)
        synthetic = deepcopy(_synthetic)
        
        # Ensure we have enough data
        if len(real_data) < 10 or len(synthetic) < 10:
            return 0.0
        
        # Step 1: Create labels as described
        # L_Y = [1, 1, ..., 1] with |L_Y| = |Y|
        # L_Z = [0, 0, ..., 0] with |L_Z| = |Z|
        L_Y = [1] * len(real_data)
        L_Z = [0] * len(synthetic)
        
        # Step 2: Create combined dataset D = Y ∪ Z and labels L = L_Y ∪ L_Z
        D = pd.concat([real_data, synthetic], axis=0, ignore_index=True)
        L = pd.Series(L_Y + L_Z)
        
        # Handle categorical columns - one-hot encode
        cat_cols = D.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            D_encoded = pd.get_dummies(D, columns=cat_cols, drop_first=True)
        else:
            D_encoded = D
        
        # Handle missing values
        D_encoded = D_encoded.fillna(D_encoded.mean())
        
        # Run multiple iterations and average results
        recall_scores = []
        
        for _ in range(num_eval_iter):
            # Step 3: Split into train/test sets
            try:
                D_train, D_test, L_train, L_test = train_test_split(
                    D_encoded, L, test_size=0.3, random_state=None, stratify=L
                )
            except ValueError:
                # If stratification fails, do simple split
                D_train, D_test, L_train, L_test = train_test_split(
                    D_encoded, L, test_size=0.3, random_state=None
                )
            
            # Step 4: Train LightGBM classifier
            # f_θ = LightGBM(D_train)
            cls = LGBMClassifier(verbosity=-1, random_state=42)
            cls.fit(D_train, L_train)
            
            # Step 5: Make predictions
            # ĥ_i = 1 if f_θ(x_j) >= 0.5, else 0
            y_pred = cls.predict(D_test)
            
            # Step 6: Calculate recall (MIR)
            # MIR = (True Positives) / (True Positives + False Negatives)
            # = (correctly identified real samples) / (total real samples in test)
            recall = recall_score(L_test, y_pred)
            recall_scores.append(recall)
        
        # Return average recall across iterations
        mir_score = np.mean(recall_scores)
        
        return 1 - float(mir_score)
        
    except Exception as e:
        print(f"Error in MIR calculation: {e}")
        return 0.0



