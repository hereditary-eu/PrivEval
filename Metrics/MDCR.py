# https://github.com/schneiderkamplab/syntheval/blob/main/src/syntheval/metrics/privacy

#import pdb

from copy import deepcopy

from syntheval import SynthEval
from sklearn.model_selection import train_test_split
from numpy import exp

def calculate_metric(args, _real_data, _synthetic):
    real_data = deepcopy(_real_data)
    synthetic = deepcopy(_synthetic)

    evaluator = SynthEval(real_data)

    evaluator.evaluate(synthetic, dcr={})

    #result = evaluator._raw_results['dcr']['mDCR']
    
    return sigmoid(3-evaluator._raw_results['dcr']['mDCR'])

def sigmoid(x):
    return 1/(1 + exp(-x)) 


