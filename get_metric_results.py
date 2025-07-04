import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from Metrics import AttributeInference1 as AIR
from Metrics import CGeneralizedCAP as GCAP
from Metrics import CZeroCAP as CZCAP
from Metrics import NNAA
from Metrics import MemInf as MIR
from Metrics import Hitting_rate
from Metrics import MDCR
from Metrics import DCR
from Metrics import NNDR


def get_metric_results(real_data, syn_data, real_labels, syn_labels, sensitive_attributes=None):
    import subprocess
    import sys
    from Metrics import Hidden_rate
    import warnings
    warnings.filterwarnings('ignore')

    # Install a package (e.g., scipy)
    #subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy~=1.14.1"])
    metrics = {
                    'sanity': ['common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
                    'stats': ['alpha_precision'],
                    'detection': ['detection_mlp'],
                    'privacy': ['identifiability_score'],
                }
    
    #mir = MIR.calculate_metric(args=None, _real_data=real_data, _synthetic=syn_data)


    from Metrics import All_synthcity

    synthcity_results = All_synthcity.calculate_metric(args = None, _real_data=real_data, _synthetic=real_data, _metrics=metrics)
    crp = synthcity_results['mean'][1]
    nsnd = 1-synthcity_results['mean'][2]
    cvp = synthcity_results['mean'][3]
    dvp = 1-synthcity_results['mean'][4]
    auth = synthcity_results['mean'][10]
    mlp = synthcity_results['mean'][11]
    id_score = synthcity_results['mean'][12]
    air = AIR.calculate_metric(args = None, _real_data=real_data, _synthetic=syn_data, sensitive_attributes=sensitive_attributes)
    gcap = GCAP.calculate_metric(args = None, _real_data=real_labels, _synthetic=syn_labels, sensitive_attributes=sensitive_attributes)
    zcap = CZCAP.calculate_metric(args = None, _real_data=real_labels, _synthetic=syn_labels, sensitive_attributes=sensitive_attributes)
    mdcr = MDCR.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    hitR = Hitting_rate.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    nnaa = NNAA.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    dcr = DCR.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    nndr = NNDR.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    hidd = Hidden_rate.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
     
    priv_results = np.around([air, gcap, zcap, 
                            mdcr, hitR, #mir, 
                            nnaa, crp, nsnd, 
                            cvp, dvp, auth, 
                            mlp, id_score, 
                            dcr, nndr, hidd
                            ], 2).tolist()
    
    metric_list = ["Attribute Inference Risk", "GeneralizedCAP", "ZeroCAP", 
                   "Median Distance to Closest Record", "Hitting Rate",
                   #"Membership Inference Risk", 
                   "Nearest Neighbour Adversarial Accuracy",
                   "Common Row Proportion", "Nearest Synthetic Neighbour Distance",
                   "Close Value Probability", "Distant Value Probability",
                   "Authenticity", "DetectionMLP", "Identifiability Score"
                   , "Distance to Closest Record", "Nearest Neighbour Distance Ratio", "Hidden Rate"
                   ]
    
    results = pd.DataFrame({'Metric':metric_list, 'Result':priv_results})
    
    return results

# #Get metric results for PrivBayes
# from sklearn.preprocessing import LabelEncoder
# import pandas as pd
# from get_metric_results import get_metric_results

# df = pd.read_csv("Data/real.csv", index_col=False)
# epsilon = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0]
# for eps in epsilon:
#     syn_df = pd.read_csv(f"Data/privbayes(e={eps}).csv", index_col=False)

#     all_data = pd.concat([df, syn_df], ignore_index=True)

#     cat_cols = all_data.select_dtypes(include=['object', 'bool']).columns

#     # Initialize a dictionary to hold encoded data
#     encoded_data = {}
#     for col in cat_cols:
#         if all_data[col].dtype == 'bool':
#             encoded_data[col] = all_data[col].astype(int)
#         else:
#             le = LabelEncoder()
#             encoded_data[col] = le.fit_transform(all_data[col].astype(str))

#     num_cols = all_data.select_dtypes(exclude=['object', 'bool']).columns
#     for col in num_cols:
#         encoded_data[col] = all_data[col]

#     all_labels = pd.DataFrame(encoded_data)
#     real_len = len(df)
#     real_labels = all_labels[:real_len]
#     syn_labels = all_labels[real_len:]

#     metric_results = get_metric_results(df, syn_df, real_labels, syn_labels, sensitive_attributes=['Revenue'])
#     print("Metric Results:")
#     print(metric_results)

#     metric_results.to_csv(f"metric_results/privbayes(e={eps})_metric_results.csv", index=False)