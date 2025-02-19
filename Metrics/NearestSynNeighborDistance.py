#https://synthcity.readthedocs.io/en/latest/metrics.html

from synthcity import metrics

from copy import deepcopy
def calculate_metric(args, _real_data, _synthetic):
    real_data = deepcopy(_real_data)
    synthetic = deepcopy(_synthetic)

    metric_dict = {'sanity': ['nearest_syn_neighbor_distance']}

    score = metrics.Metrics.evaluate(
        real_data,
        synthetic,
        metrics=metric_dict
    )

    result = score["mean"][0]

    return 1-result