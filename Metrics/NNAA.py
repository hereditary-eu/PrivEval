# https://github.com/schneiderkamplab/syntheval/blob/main/src/syntheval/metrics/privacy/metric_nn_adversarial_accuracy.py

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def calculate_metric(args=None, _real_data=None, _synthetic=None, k=5, **kwargs):
    """
    Calculate Nearest Neighbor Adversarial Accuracy following SynthEval implementation.
    
    Based on: https://github.com/schneiderkamplab/syntheval/blob/main/src/syntheval/metrics/privacy/metric_nn_adversarial_accuracy.py
    
    Args:
        _real_data: Real dataset
        _synthetic: Synthetic dataset
        k: Number of nearest neighbors to consider
    
    Returns:
        float: NNAA score (closer to 0.5 is better for synthetic data privacy)
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
        
        if real_subset.empty or syn_subset.empty or len(real_subset) < 5 or len(syn_subset) < 5:
            return 0.5
        
        # Handle missing values
        real_subset = real_subset.fillna(real_subset.mean())
        syn_subset = syn_subset.fillna(syn_subset.mean())
        
        # Convert to numpy arrays
        X_real = real_subset.values
        X_syn = syn_subset.values
        
        # Normalize the data
        scaler = StandardScaler()
        X_real_scaled = scaler.fit_transform(X_real)
        X_syn_scaled = scaler.transform(X_syn)
        
        # Adjust k if needed
        k = min(k, len(X_real_scaled) - 1, len(X_syn_scaled) - 1)
        if k < 1:
            return 0.5
        
        # Following SynthEval: Create adversarial features
        # For each sample, compute NN distances within real and synthetic datasets separately
        
        # KNN within real data
        nn_real = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        nn_real.fit(X_real_scaled)
        
        # KNN within synthetic data  
        nn_syn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        nn_syn.fit(X_syn_scaled)
        
        features = []
        labels = []
        
        # Process real data points
        for i, point in enumerate(X_real_scaled):
            # Find k nearest neighbors in real data (excluding self)
            distances_real, _ = nn_real.kneighbors([point])
            real_nn_distances = distances_real[0][1:]  # Exclude self
            
            # Find k nearest neighbors in synthetic data
            distances_syn, _ = nn_syn.kneighbors([point])
            syn_nn_distances = distances_syn[0][:k]  # Take first k
            
            # Create feature vector following SynthEval approach
            # Features: [mean_real_dist, std_real_dist, mean_syn_dist, std_syn_dist, 
            #           min_real_dist, min_syn_dist, ratio_mean_distances]
            mean_real = np.mean(real_nn_distances)
            std_real = np.std(real_nn_distances) if len(real_nn_distances) > 1 else 0.0
            mean_syn = np.mean(syn_nn_distances)
            std_syn = np.std(syn_nn_distances) if len(syn_nn_distances) > 1 else 0.0
            min_real = np.min(real_nn_distances)
            min_syn = np.min(syn_nn_distances)
            
            # Ratio of mean distances (avoiding division by zero)
            ratio = mean_real / (mean_syn + 1e-8)
            
            feature_vector = [
                mean_real, std_real, mean_syn, std_syn,
                min_real, min_syn, ratio
            ]
            
            features.append(feature_vector)
            labels.append(0)  # Real data label
        
        # Process synthetic data points
        for i, point in enumerate(X_syn_scaled):
            # Find k nearest neighbors in real data
            distances_real, _ = nn_real.kneighbors([point])
            real_nn_distances = distances_real[0][:k]  # Take first k
            
            # Find k nearest neighbors in synthetic data (excluding self)
            distances_syn, _ = nn_syn.kneighbors([point])
            syn_nn_distances = distances_syn[0][1:]  # Exclude self
            
            # Create feature vector
            mean_real = np.mean(real_nn_distances)
            std_real = np.std(real_nn_distances) if len(real_nn_distances) > 1 else 0.0
            mean_syn = np.mean(syn_nn_distances)
            std_syn = np.std(syn_nn_distances) if len(syn_nn_distances) > 1 else 0.0
            min_real = np.min(real_nn_distances)
            min_syn = np.min(syn_nn_distances)
            
            # Ratio of mean distances
            ratio = mean_real / (mean_syn + 1e-8)
            
            feature_vector = [
                mean_real, std_real, mean_syn, std_syn,
                min_real, min_syn, ratio
            ]
            
            features.append(feature_vector)
            labels.append(1)  # Synthetic data label
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Handle potential NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e8, neginf=-1e8)
        
        # Check if we have enough data
        if len(X) < 10:
            return 0.5
        
        # Split into train/test following SynthEval approach
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        except ValueError:
            # If stratification fails, do simple split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        # Train MLP classifier following SynthEval parameters
        clf = MLPClassifier(
            hidden_layer_sizes=(100,),  # SynthEval uses 100 units
            max_iter=1000,  # More iterations
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            alpha=0.0001  # L2 regularization
        )
        
        # Fit the classifier
        clf.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return float(accuracy)
        
    except Exception as e:
        print(f"Error in NNAA calculation: {e}")
        return 0.5

