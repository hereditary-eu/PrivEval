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
from Metrics import Hidden_rate
from Metrics import All_synthcity


def get_metric_results(real_data, syn_data, real_labels, syn_labels, sensitive_attributes=None):

    # Install a package (e.g., scipy)
    #subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy~=1.14.1"])
    metrics = {
                    'sanity': ['common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
                    'stats': ['alpha_precision'],
                    'detection': ['detection_mlp'],
                    'privacy': ['identifiability_score'],
                }
    
    try:
        mir = MIR.calculate_metric(args=None, _real_data=real_data, _synthetic=syn_data)
    except Exception as e:
        print(f"Error in MIR calculation: {e}")
        mir = 0.0

    try:
        synthcity_results = All_synthcity.calculate_metric(args = None, _real_data=real_data, _synthetic=real_data, _metrics=metrics)
        crp = synthcity_results['mean'][1]
        nsnd = 1-synthcity_results['mean'][2]
        cvp = synthcity_results['mean'][3]
        dvp = 1-synthcity_results['mean'][4]
        auth = synthcity_results['mean'][10]
        mlp = synthcity_results['mean'][11]
        id_score = synthcity_results['mean'][12]
    except Exception as e:
        print(f"Error in SynthCity calculation: {e}")
        crp, nsnd, cvp, dvp, auth, mlp, id_score = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    try:
        air = AIR.calculate_metric(args=None, _real_data=real_data, _synthetic=syn_data, sensitive_attributes=sensitive_attributes)
    except Exception as e:
        print(f"Error in AIR calculation: {e}")
        air = 0.0
    
    try:
        gcap = GCAP.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels, sensitive_attributes=sensitive_attributes)
    except Exception as e:
        print(f"Error in GCAP calculation: {e}")
        gcap = 0.0
    
    try:
        zcap = CZCAP.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels, sensitive_attributes=sensitive_attributes)
    except Exception as e:
        print(f"Error in ZCAP calculation: {e}")
        zcap = 0.0
    try:
        mdcr = MDCR.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    except Exception as e:
        print(f"Error in MDCR: {e}")
        mdcr = 0.0
    
    try:
        hitR = Hitting_rate.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    except Exception as e:
        print(f"Error in Hitting_rate: {e}")
        hitR = 0.0
    
    try:
        nnaa = NNAA.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    except Exception as e:
        print(f"Error in NNAA: {e}")
        nnaa = 0.0
    
    try:
        dcr = DCR.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    except Exception as e:
        print(f"Error in DCR: {e}")
        dcr = 0.0
    
    try:
        nndr = NNDR.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    except Exception as e:
        print(f"Error in NNDR: {e}")
        nndr = 0.0
    
    try:
        hidd = Hidden_rate.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    except Exception as e:
        print(f"Error in Hidden_rate: {e}")
        hidd = 0.0
        
    # Convert results to a list with 2 decimal places
    priv_results = np.around([air, gcap, zcap, 
                            mdcr, hitR, mir, 
                            nnaa, crp, nsnd, 
                            cvp, dvp, auth, 
                            mlp, id_score, 
                            dcr, nndr, hidd
                            ], 2).tolist()
    
    metric_list = ["Attribute Inference Risk", "GeneralizedCAP", "ZeroCAP", 
                   "Median Distance to Closest Record", "Hitting Rate",
                   "Membership Inference Risk", 
                   "Nearest Neighbour Adversarial Accuracy",
                   "Common Row Proportion", "Nearest Synthetic Neighbour Distance",
                   "Close Value Probability", "Distant Value Probability",
                   "Authenticity", "DetectionMLP", "Identifiability Score"
                   , "Distance to Closest Record", "Nearest Neighbour Distance Ratio", "Hidden Rate"
                   ]
    
    results = pd.DataFrame({'Metric':metric_list, 'Result':priv_results})
    
    return results
