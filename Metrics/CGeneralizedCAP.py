#https://docs.sdv.dev/sdmetrics

from sdmetrics.single_table import CategoricalGeneralizedCAP

from copy import deepcopy
def calculate_metric(args, _real_data, _synthetic, sensitive_attributes=None):
    real_data = deepcopy(_real_data)
    synthetic = deepcopy(_synthetic)

    #Load the sentive attributes from sensitive_attributes.txt
    if sensitive_attributes is None:
        sensitive_file = open("sensitive_attributes.txt", "r")
        sensitive_data = sensitive_file.read()
        sensitive_attributes = sensitive_data.split("\n")
        sensitive_file.close()

    #Get key attributes
    key_attributes = []
    for column in real_data.columns:
            if column not in sensitive_attributes:
                key_attributes.append(column)

    print("CGeneralizedCAP: Computing")
    result = CategoricalGeneralizedCAP.compute(real_data=real_data, synthetic_data=synthetic,
        key_fields=key_attributes,
        sensitive_fields=sensitive_attributes,
        )

    return 1-result