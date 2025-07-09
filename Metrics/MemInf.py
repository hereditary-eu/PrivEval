### Metric from https://github.com/schneiderkamplab/syntheval

import numpy as np
import pandas as pd
from logging import warning
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from copy import deepcopy


class MIAClassifier:
    """The Metric Class for Membership Inference Attack evaluation.

    Attributes:
    self.real_data : DataFrame
    self.synt_data : DataFrame
    self.hout_data : DataFrame
    self.cat_cols  : list of strings
    self.num_cols  : list of strings
    """

    def __init__(self, real_data, synt_data, hout_data, cat_cols=None, num_cols=None):
        self.real_data = real_data
        self.synt_data = synt_data
        self.hout_data = hout_data
        self.cat_cols = cat_cols if cat_cols is not None else []
        self.num_cols = num_cols if num_cols is not None else []
        self.results = {}

    def name(self) -> str:
        """Name/keyword to reference the metric"""
        return "mia"

    def type(self) -> str:
        """Set to 'privacy' or 'utility'"""
        return "privacy"

    def evaluate(self, num_eval_iter=5) -> float | dict:
        """Function for computing the precision, recall, and F1-score of a membership 
        inference attack using a LightGBM classifier
        
        Args:
            num_eval_iter (int): Number of iterations to run the classifier

        Returns:
            dict: Precision, recall, and F1-score of the membership inference
        """
        try:
            assert self.hout_data is not None
        except AssertionError:
            print(" Warning: Membership inference attack metric did not run, holdout data was not supplied!")
            return {"MIA recall": 0.0}
        
        if len(self.real_data) < len(self.hout_data) // 2:
                warning(
                    "The holdout data is more than double the size of the real data. The holdout data will be downsampled to match the size of the real data. real size: %s, holdout size: %s", len(self.real_data), len(self.hout_data)
                )
        
        # One-hot encode. All data is combined to ensure consistent encoding
        combined_data = pd.concat(
            [self.real_data, self.synt_data, self.hout_data], ignore_index=True
        )
        combined_data_encoded = pd.get_dummies(
            combined_data, columns=self.cat_cols, drop_first=True
        )

        # Separate into the three datasets
        real = combined_data_encoded.iloc[: len(self.real_data)].reset_index(drop=True)
        syn = combined_data_encoded.iloc[
            len(self.real_data) : len(self.real_data) + len(self.synt_data)
        ].reset_index(drop=True)
        hout = combined_data_encoded.iloc[
            len(self.real_data) + len(self.synt_data) :
        ].reset_index(drop=True)

        # Run classifier multiple times and average the results
        pre_results = {
            "precision": [],
            "recall": [],
            "f1": [],
        }
        for _ in range(num_eval_iter):
            hout_train, hout_test = train_test_split(hout, test_size=0.25)
            syn_samples = syn.sample(n=len(hout_train))

            # Create training data consisting of synthetic and holdout data
            X_train = pd.concat([syn_samples, hout_train], axis=0, ignore_index=True)
            y_train = pd.Series([1] * len(syn_samples) + [0] * len(hout_train))

            # Shuffle
            shuffle_idx = np.arange(len(X_train))
            np.random.shuffle(shuffle_idx)
            X_train = X_train.iloc[shuffle_idx]
            y_train = y_train.iloc[shuffle_idx]
            
            # Create test set by combining some random data from the real and holdout data with an equal number of records from each dataframe
            if len(real) < len(hout_test):
                hout_sample = hout_test.sample(n=len(real))
                real_sample = real
            else:
                real_sample = real.sample(n=len(hout_test))
                hout_sample = hout_test
            X_test = pd.concat(
                [
                    real_sample,
                    hout_sample,
                ],
                axis=0,
                ignore_index=True,
            )
            y_test = pd.Series([1] * len(real_sample) + [0] * len(hout_sample))

            cls = LGBMClassifier(verbosity=-1).fit(X_train, y_train)
            # Get predictions
            holdout_predictions = cls.predict(X_test)

            # Calculate precision, recall, and F1-score
            pre_results["precision"].append(
                precision_score(y_test, holdout_predictions)
            )
            pre_results["recall"].append(recall_score(y_test, holdout_predictions))
            pre_results["f1"].append(
                f1_score(y_test, holdout_predictions, average="macro")
            )

        precision = np.mean(pre_results["precision"])
        precision_se = np.std(pre_results["precision"], ddof=1) / np.sqrt(
            num_eval_iter
        )

        recall = np.mean(pre_results["recall"])
        recall_se = np.std(pre_results["recall"], ddof=1) / np.sqrt(num_eval_iter)

        f1 = np.mean(pre_results["f1"])
        f1_se = np.std(pre_results["f1"], ddof=1) / np.sqrt(num_eval_iter)

        self.results = {
            "MIA precision": precision,
            "MIA precision se": precision_se,
            "MIA recall": recall,
            "MIA recall se": recall_se,
            "MIA macro F1": f1,
            "MIA macro F1 se": f1_se,
        }

        return self.results


def calculate_metric(args, _real_data, _synthetic):
    """Wrapper function to maintain compatibility with existing code"""
    real_data = deepcopy(_real_data)
    synthetic = deepcopy(_synthetic)

    # Split real data into train and holdout
    train_df, holdout_df = train_test_split(real_data, test_size=0.2, random_state=42)
    
    # Determine categorical and numerical columns
    cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create MIA classifier
    mia = MIAClassifier(train_df, synthetic, holdout_df, cat_cols, num_cols)
    
    # Evaluate and return recall score
    results = mia.evaluate(num_eval_iter=5)
    
    return results.get("MIA recall", 0.0)
