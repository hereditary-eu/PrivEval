#https://synthcity.readthedocs.io/en/latest/metrics.html

from synthcity import metrics

from copy import deepcopy
def calculate_metric(args, _real_data, _synthetic, _metrics):
    real_data = _real_data
    synthetic = _synthetic

    scores = metrics.Metrics.evaluate(
        real_data,
        synthetic,
        metrics = _metrics,
        use_cache = False
    )

    return scores