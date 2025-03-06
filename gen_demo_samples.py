import pandas as pd
import numpy as np
from faker import Faker
import random
from DataSynthesizer1.DataDescriber import DataDescriber
from DataSynthesizer1.DataGenerator import DataGenerator
from synthesis.synthesizers.privbayes import PrivBayes
from datetime import datetime
import os
from Metrics import AttributeInference as AIR
from Metrics import CGeneralizedCAP as GCAP
from Metrics import All_synthcity
from Metrics import AttributeInference1 as AIR
from Metrics import CGeneralizedCAP as GCAP
from Metrics import CZeroCAP as CZCAP
# from Metrics import DCR
# from Metrics import NNDR
# from Metrics import Hidden_rate
from Metrics import NNAA
from Metrics import MemInf as MIR
from Metrics import Hitting_rate
from Metrics import MDCR
from sklearn.preprocessing import LabelEncoder
from Metrics import All_synthcity
import pandas as pd
import numpy as np

def get_metric_results(real_data, syn_data):
    real_data['Height'].astype('Float32')
    syn_data['Height'].astype('Float32')

    all_data = pd.concat([real_data, syn_data])
    fn_encoder = LabelEncoder()
    ln_encoder = LabelEncoder()
    fl_encoder = LabelEncoder()
    na_encoder = LabelEncoder()
    r_fn = fn_encoder.fit_transform(all_data['First Name'])
    r_ln = ln_encoder.fit_transform(all_data['Last Name'])
    r_na = na_encoder.fit_transform(all_data['Nationality'])
    r_fl = fl_encoder.fit_transform(all_data['Favorite Icecream'])
    all_labels = pd.DataFrame({'First Name':r_fn, 'Last Name': r_ln, 'Height': all_data['Height'],'Nationality': r_na, 'Favorite Icecream':r_fl, 'Like Liquorice': all_data['Like Liquorice'], 'Times Been to Italy': all_data['Times Been to Italy'], 'First Time London': all_data['First Time London'], 'Steps per Day': all_data['Steps per Day']})
    real_labels = all_labels[:len(real_data)]
    syn_labels = all_labels[-len(real_data):]
    
    metrics = {
                    'sanity': ['common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
                    'stats': ['alpha_precision'],
                    'detection': ['detection_mlp'],
                    'privacy': ['identifiability_score'],
                }
    #air = AIR.calculate_metric(args = None, _real_data=real_dat, _synthetic=syn_dat)
    synthcity_results = All_synthcity.calculate_metric(args = None, _real_data=real_labels, _synthetic=syn_labels, _metrics=metrics)
    crp = synthcity_results['mean'][1]
    nsnd = 1-synthcity_results['mean'][2]
    cvp = synthcity_results['mean'][3]
    dvp = 1-synthcity_results['mean'][4]
    auth = synthcity_results['mean'][10]
    mlp = synthcity_results['mean'][11]
    id_score = synthcity_results['mean'][12]
    air = AIR.calculate_metric(args = None, _real_data=real_data, _synthetic=syn_data)
    gcap = GCAP.calculate_metric(args = None, _real_data=real_labels, _synthetic=syn_labels)
    zcap = CZCAP.calculate_metric(args = None, _real_data=real_labels, _synthetic=syn_labels)
    mdcr = MDCR.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    hitR = Hitting_rate.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    mir = MIR.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    nnaa = NNAA.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    #---These metrics simply take too long to run
    # dcr = DCR.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    # nndr = NNDR.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
    # hidd = Hidden_rate.calculate_metric(args=None, _real_data=real_labels, _synthetic=syn_labels)
     
    priv_results = np.around([air, gcap, zcap, 
                            mdcr, hitR, mir, 
                            nnaa, crp, nsnd, 
                            cvp, dvp, auth, 
                            mlp, id_score, 
                            #dcr, nndr, hidd
                            ], 2).tolist()
    
    metric_list = ["Attribute Inference Risk", "GeneralizedCAP", "ZeroCAP", 
                   "Median Distance to Closest Record", "Hitting Rate",
                   "Membership Inference Risk", "Nearest Neighbour Adversarial Accuracy",
                   "Common Row Proportion", "Nearest Synthetic Neighbour Distance",
                   "Close Value Probability", "Distant Value Probability",
                   "Authenticity", "DetectionMLP", "Identifiability Score"
                   # "Distance to Closest Record", "Nearest Neighbour Distance Ratio", "Hidden Rate"
                   ]
    
    results = pd.DataFrame({'Metric':metric_list, 'Result':priv_results})
    
    return results
#generate_real_data(1499, 1)
#generate_real_data(1499, 0)


# eps_list = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2.5, 5]
# for liquorice in [0,1]:
#     real_dir = f'sample_data_{liquorice}.csv'
#     real_data = pd.read_csv(real_dir)
#     fn_encoder = LabelEncoder()
#     ln_encoder = LabelEncoder()
#     r_fn = fn_encoder.fit_transform(real_data['First Name'])
#     r_ln = ln_encoder.fit_transform(real_data['Last Name'])
#     real_data["First Name"] = r_fn
#     real_data["Last Name"] = r_ln
#     for eps in eps_list:
#         Syn_no_bin = synthesize_no_bin(real_data, eps)
#         Syn_no_bin["First Name"] = fn_encoder.inverse_transform(Syn_no_bin["First Name"])
#         Syn_no_bin["Last Name"] = ln_encoder.inverse_transform(Syn_no_bin["Last Name"])
#         Syn_no_bin.to_csv(f"demo_syn/syn_bin_{liquorice}_{eps}.csv")
        
        #synthesize_bin(real_data, eps).to_csv(f"demo_syn/syn_no_{liquorice}_{eps}.csv")

eps_list = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2.5, 5]
for liquorice in [0,1]:
    real_dir = f'sample_data_{liquorice}.csv'
    real_data = pd.read_csv(real_dir, index_col=False)
    for eps in eps_list:
        syn_no = pd.read_csv(f"demo_syn_new/syn_no_{liquorice}_{eps}.csv", index_col=False)
        get_metric_results(real_data, syn_no).to_csv(f"metric_full/syn_no_{liquorice}_{eps}.csv")