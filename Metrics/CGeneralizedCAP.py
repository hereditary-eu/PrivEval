#https://docs.sdv.dev/sdmetrics
# Custom implementation of Categorical Generalized-CAP (Correct Attribution Probability)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def calculate_metric(args=None, _real_data=None, _synthetic=None, sensitive_attributes=None, **kwargs):
    """
    Custom implementation of Categorical Generalized-CAP.
    Uses machine learning to predict sensitive attributes from key attributes.
    
    Args:
        _real_data: Real dataset
        _synthetic: Synthetic dataset
        sensitive_attributes: List of sensitive column names
    
    Returns:
        float: Generalized-CAP score (lower is better for privacy)
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
        
        # Handle missing values
        for col in required_cols:
            real_subset[col] = real_subset[col].fillna('missing')
            syn_subset[col] = syn_subset[col].fillna('missing')
        
        cap_scores = []
        
        # For each sensitive attribute, train a classifier
        for sensitive_attr in sensitive_attributes:
            try:
                # Combine real and synthetic data for training
                combined_data = pd.concat([real_subset, syn_subset], ignore_index=True)
                
                # Encode categorical variables
                encoded_data = combined_data.copy()
                label_encoders = {}
                
                for col in key_attributes:
                    if col in encoded_data.columns:
                        le = LabelEncoder()
                        encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
                        label_encoders[col] = le
                
                # Encode target variable
                target_le = LabelEncoder()
                encoded_target = target_le.fit_transform(combined_data[sensitive_attr].astype(str))
                
                # Prepare features and target
                X = encoded_data[key_attributes]
                y = encoded_target
                
                # Check if we have enough data and classes
                if len(X) < 10 or len(np.unique(y)) < 2:
                    cap_scores.append(0.0)  # Perfect privacy if not enough data
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                # Train classifier
                clf = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    min_samples_split=5,
                    min_samples_leaf=2
                )
                
                clf.fit(X_train, y_train)
                
                # Predict on test set
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Calculate random baseline accuracy
                unique_classes = np.unique(y)
                random_accuracy = 1.0 / len(unique_classes) if len(unique_classes) > 0 else 0.0
                
                # Adjust for random baseline
                if accuracy > random_accuracy:
                    cap_score = (accuracy - random_accuracy) / (1.0 - random_accuracy)
                else:
                    cap_score = 0.0
                
                cap_scores.append(cap_score)
                
            except Exception as e:
                print(f"Error processing sensitive attribute {sensitive_attr}: {e}")
                cap_scores.append(0.0)

        # Return average CAP (lower is better for privacy)
        avg_cap = np.mean(cap_scores) if cap_scores else 0.0
        return float(avg_cap)
        
    except Exception as e:
        print(f"Error in CGeneralizedCAP calculation: {e}")
        return 0.0  # Return perfect privacy score on error