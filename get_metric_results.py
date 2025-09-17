import pandas as pd
import numpy as np
from Metrics import AttributeInference as AIR
from Metrics import CGeneralizedCAP as GCAP
from Metrics import CZeroCAP as CZCAP
from Metrics import NNAA
from Metrics import MemInf as MIR
from Metrics import Hitting_rate
from Metrics import MDCR
from Metrics import DCR
from Metrics import NNDR
from Metrics import Hidden_rate
from Metrics import CommonRowsProportion as CRP
from Metrics import NearestSynNeighborDistance as NSND
from Metrics import CloseValuesProbability as CVP
from Metrics import DistantValuesProbability as DVP
from Metrics import Authenticity as Auth
from Metrics import DetectionMLP as DMLP
from Metrics import IdentifiabilityScore as IS


def get_metric_results(real_data, syn_data, real_labels, syn_labels, sensitive_attributes=None):
    
    try:
        mir = MIR.calculate_metric(args=None, _real_data=real_data, _synthetic=syn_data)
        print(f"MIR calculated: {mir}")
    except Exception as e:
        print(f"Error in MIR calculation: {e}")
        mir = 0.0

    try:
        crp = CRP.calculate_metric(args=None, _real_data=real_data, _synthetic=syn_data)
        print(f"CRP calculated: {crp}")
    except Exception as e:
        print(f"Error in CRP calculation: {e}")
        crp = 0.0
    try:
        nsnd = NSND.calculate_metric(args=None, _real_data=real_data, _synthetic=syn_data)
        print(f"NSND calculated: {nsnd}")
    except Exception as e:
        print(f"Error in NSND calculation: {e}")
        nsnd = 0.0
    try:
        cvp = CVP.calculate_metric(args=None, _real_data=real_data, _synthetic=syn_data)
        print(f"CVP calculated: {cvp}")
    except Exception as e:
        print(f"Error in CVP calculation: {e}")
        cvp = 0.0
    try:
        dvp = DVP.calculate_metric(args=None, _real_data=real_data, _synthetic=syn_data)
        print(f"DVP calculated: {dvp}")
    except Exception as e:
        print(f"Error in DVP calculation: {e}")
        dvp = 0.0
    try:
        auth = Auth.calculate_metric(args=None, _real_data=real_data, _synthetic=syn_data)
        print(f"Auth calculated: {auth}")
    except Exception as e:
        print(f"Error in Auth calculation: {e}")
        auth = 0.0
    try:
        mlp = DMLP.calculate_metric(args=None, _real_data=real_data, _synthetic=syn_data)
        print(f"MLP calculated: {mlp}")
    except Exception as e:
        print(f"Error in DMLP calculation: {e}")
        mlp = 0.0
    try:
        id_score = IS.calculate_metric(args=None, _real_data=real_data, _synthetic=syn_data)
        print(f"IS calculated: {id_score}")
    except Exception as e:
        print(f"Error in IS calculation: {e}")
        id_score = 0.0
    try:
        air = AIR.calculate_metric(args=None, _real_data=real_data, _synthetic=syn_data, sensitive_attributes=sensitive_attributes)
        print(f"AIR calculated: {air}")
    except Exception as e:
        print(f"Error in AIR calculation: {e}")
        air = 0.0
    
    try:
        gcap = GCAP.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels, sensitive_attributes=sensitive_attributes)
        print(f"GCAP calculated: {gcap}")
    except Exception as e:
        print(f"Error in GCAP calculation: {e}")
        gcap = 0.0
    
    try:
        zcap = CZCAP.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels, sensitive_attributes=sensitive_attributes)
        print(f"ZCAP calculated: {zcap}")
    except Exception as e:
        print(f"Error in ZCAP calculation: {e}")
        zcap = 0.0
    try:
        mdcr = MDCR.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
        print(f"MDCR calculated: {mdcr}")
    except Exception as e:
        print(f"Error in MDCR: {e}")
        mdcr = 0.0
    
    try:
        hitR = Hitting_rate.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
        print(f"Hitting_rate calculated: {hitR}")
    except Exception as e:
        print(f"Error in Hitting_rate: {e}")
        hitR = 0.0
    
    try:
        nnaa = NNAA.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
        print(f"NNAA calculated: {nnaa}")
    except Exception as e:
        print(f"Error in NNAA: {e}")
        nnaa = 0.0
    
    try:
        dcr = DCR.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
        print(f"DCR calculated: {dcr}")
    except Exception as e:
        print(f"Error in DCR: {e}")
        dcr = 0.0
    
    try:
        nndr = NNDR.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
        print(f"NNDR calculated: {nndr}")
    except Exception as e:
        print(f"Error in NNDR: {e}")
        nndr = 0.0
    
    try:
        hidd = Hidden_rate.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
        print(f"Hidden_rate calculated: {hidd}")
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
