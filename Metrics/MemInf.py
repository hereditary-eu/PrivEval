# https://github.com/schneiderkamplab/syntheval/blob/main/src/syntheval/metrics/privacy

import numpy as np
import pandas as pd

from syntheval import SynthEval
from sklearn.model_selection import train_test_split
from copy import deepcopy

def calculate_metric(args, _real_data, _synthetic):
    real_data = deepcopy(_real_data)
    synthetic = deepcopy(_synthetic)

    train_df, holdout_df = train_test_split(real_data, test_size=0.2, random_state=42)

    evaluator = SynthEval(train_df, holdout_dataframe = holdout_df)

    evaluator.evaluate(synthetic, mia_risk = {"num_eval_iter": 5})
    result = evaluator._raw_results['mia_risk']['MIA recall']

    return result
