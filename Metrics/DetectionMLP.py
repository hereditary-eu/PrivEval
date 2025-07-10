import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def calculate_metric(args=None, _real_data=None, _synthetic=None, k_folds=5, **kwargs):
    """
    Calculate DetectionMLP (D-MLP) following SynthCity implementation.
    
    D-MLP assesses how well a shallow neural network can distinguish between 
    real and synthetic data using k-fold cross-validation and AUC metric.
    
    Args:
        _real_data: Real dataset
        _synthetic: Synthetic dataset
        k_folds: Number of cross-validation folds
    
    Returns:
        float: Average AUC score (closer to 0.5 = better privacy)
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
        
        if real_subset.empty or syn_subset.empty:
            return 0.5
        
        # Handle missing values
        real_subset = real_subset.fillna(real_subset.mean())
        syn_subset = syn_subset.fillna(syn_subset.mean())
        
        # Create labels as specified in the description
        # L_Y = [1, 1, ..., 1] (real data labeled as 1)
        # L_Z = [0, 0, ..., 0] (synthetic data labeled as 0)
        real_labels = np.ones(len(real_subset))
        syn_labels = np.zeros(len(syn_subset))
        
        # Combine data: D = Y ∪ Z, L = L_Y ∪ L_Z
        X = np.vstack([real_subset.values, syn_subset.values])
        y = np.hstack([real_labels, syn_labels])
        
        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform k-fold cross-validation
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        auc_scores = []
        
        for train_idx, test_idx in skf.split(X_scaled, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train shallow MLP classifier
            # f_θ_i = MLP(D_train_i, L_train_i)
            mlp = MLPClassifier(
                hidden_layer_sizes=(100, 50),  # Shallow network
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            
            mlp.fit(X_train, y_train)
            
            # Get probability predictions for AUC calculation
            y_pred_proba = mlp.predict_proba(X_test)[:, 1]  # Probability of class 1 (real)
            
            # Calculate AUC for this fold
            auc = roc_auc_score(y_test, y_pred_proba)
            auc_scores.append(auc)
        
        # Calculate average AUC across all folds
        # DMLP = (1/k) * Σ AUC(L_test_i, f_θ_i(D_test_i))
        avg_auc = np.mean(auc_scores)
        
        return float(avg_auc)
        
    except Exception as e:
        print(f"Error in DetectionMLP: {e}")
        return 0.5