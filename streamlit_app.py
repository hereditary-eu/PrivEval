import altair as alt
import pandas as pd
import streamlit as st
import numpy as np
from faker import Faker
import random
from synthesis.synthesizers.privbayes import PrivBayes
from sklearn.preprocessing import LabelEncoder
from Metrics import All_synthcity
from Metrics import AttributeInference1 as AIR
from Metrics import CGeneralizedCAP as GCAP
from Metrics import CZeroCAP as CZCAP
from Metrics import NNAA
from Metrics import MemInf as MIR
from Metrics import Hitting_rate
from Metrics import MDCR
from saiph.projection import fit_transform
from saiph.projection import transform
import matplotlib.pyplot as plt
from DataSynthesizer1.DataDescriber import DataDescriber
from DataSynthesizer1.DataGenerator import DataGenerator
from datetime import datetime
import math
import os

# Show the page title and description.
st.set_page_config(page_title="Privacy estimation", page_icon="😈", layout='wide')
st.title("Privacy Estimation of Your Data")


#st.session_state.stage = 0
if 'stage' not in st.session_state:
    st.session_state.stage = 0
    st.session_state['first_name'] = "Jane"
    st.session_state['last_name'] = "Doe"
    st.session_state['height'] = 178.63
    st.session_state['flavour'] = "Pistacio"
    st.session_state.df = None
    st.session_state.pd = None
    st.session_state['results'] = []
    st.session_state['epsilon'] = 2.0
   

def generate_real_data():
    #Code for generating a baby dataset
    name_gen = Faker()
    num_samples = 9
    heights = np.around(list(np.random.normal(loc=50, scale=1, size=num_samples)), 2)
    classic_icecreams = [
        "Vanilla", "Chocolate", "Strawberry", "Mint Chocolate Chip",
        "Cookies and Cream", "Rocky Road", "Butter Pecan", "Neapolitan",
        "Pistachio", "French Vanilla"
    ]
    fav_icecream = list(random.choices(classic_icecreams, k=num_samples))

    # Generate random first and last names
    name_df = pd.DataFrame({
        'First_Name': [name_gen.first_name() for _ in range(num_samples)],
        'Last_Name': [name_gen.last_name() for _ in range(num_samples)]
    })
    height_df = pd.DataFrame({'Height': heights})
    icecream_df = pd.DataFrame({'Flavour': fav_icecream})
    full_df = pd.concat([name_df, height_df, icecream_df], axis=1)
    
    return full_df

def add_sens_individual(fn, ln, he, fl):
    sens_individual = [fn, ln, he, fl]
    new_individual_df = pd.DataFrame([sens_individual], columns=generate_real_data().columns)
    all_individuals = pd.concat([generate_real_data(), new_individual_df], ignore_index=True)
    return all_individuals

def scatter_plot(coord_real, coord_synth):
    # Scatter Plot
    plt.figure()
    # Plot DataFrame 1
    plt.scatter(coord_real['Dim. 1'], coord_real['Dim. 2'], color='blue', label='Real', alpha=0.7)
    # Plot DataFrame 2
    plt.scatter(coord_synth['Dim. 1'], coord_synth['Dim. 2'], color='red', label='Synthetic', alpha=0.5)
    plt.title('Scatter Plot of real and synthetic data')
    plt.xlabel('X-coord')
    plt.ylabel('Y-coord')
    plt.legend()
    plt.grid(True)

    # Show Plot
    #plt.show()
    return plt

def scatter_plot_real(coord_real):
    # Scatter Plot
    plt.figure()
    # Plot DataFrame 1
    plt.scatter(coord_real['Dim. 1'], coord_real['Dim. 2'], color='blue', label='Real', alpha=1)
    # Plot DataFrame 2
    plt.title('Scatter Plot of real and synthetic data')
    plt.xlabel('X-coord')
    plt.ylabel('Y-coord')
    plt.legend()
    plt.grid(True)

    # Show Plot
    #plt.show()
    return plt

def get_metric_results(real_data, syn_data):
    real_data['Height'].astype('Float32')
    syn_data['Height'].astype('Float32')

    all_data = pd.concat([real_data, syn_data])
    fn_encoder = LabelEncoder()
    ln_encoder = LabelEncoder()
    fl_encoder = LabelEncoder()
    r_fn = fn_encoder.fit_transform(all_data['First_Name'])
    r_ln = ln_encoder.fit_transform(all_data['Last_Name'])
    r_fl = fl_encoder.fit_transform(all_data['Flavour'])
    all_labels = pd.DataFrame({'First_Name':r_fn, 'Last_Name': r_ln, 'Height': all_data['Height'],'Flavour':r_fl})
    real_labels = all_labels[:len(real_data)]
    syn_labels = all_labels[-len(real_data):]

    metrics = {
                    'sanity': ['common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
                    'stats': ['alpha_precision'],
                    'detection': ['detection_mlp'],
                    'privacy': ['identifiability_score'],
                }

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
    
    st.session_state['priv_results'] = np.around(
                                        [air, gcap, zcap, 
                                        mdcr, hitR, mir, 
                                        nnaa, crp, nsnd, 
                                        cvp, dvp, auth, 
                                        mlp, id_score], 2).tolist()
    
    metric_list = ["Attribute Inference Risk", "GeneralizedCAP", "ZeroCAP", 
                   "Median Distance to Closest Record", "Hitting Rate",
                   "Membership Inference Risk", "Nearest Neighbour Adversarial Accuracy",
                   "Common Row Proportion", "Nearest Synthetic Neighbour Distance",
                   "Close Value Probability", "Distant Value Probability",
                   "Authenticity", "DetectionMLP", "Identifiability Score"
                   ]
    
    results = pd.DataFrame({'Metric':metric_list, 'Result':st.session_state['priv_results']})
    
    return results

def metric_bar_chart(results02, results1, results5, non_priv_metric_result, syn_results, epsilon):
    ind = np.arange(len(non_priv_metric_result))
    width = 0.175
    
    fig, ax = plt.subplots(figsize=(6,5.8))
    fig.suptitle("Metric Results (0 = Private, 1 = Non Private)", fontsize=14)
    ax.barh(ind, results02['Result'], width, color='darkblue', label='ϵ=0.2')
    ax.barh(ind + width, results1['Result'], width, color='mediumblue', label='ϵ=1')
    ax.barh(ind + width*2, results5['Result'], width, color='royalblue', label='ϵ=5')
    ax.barh(ind+ width*3, non_priv_metric_result['Result'], width, color='cornflowerblue', label='ϵ=∞ (Non-Private)')
    ax.barh(ind + width*4, syn_results['Result'], width, color='red', label=f'ϵ={epsilon} (Your Data)')

    ax.set(yticks=ind + width, yticklabels=non_priv_metric_result['Metric'], xlabel="Score")
    ax.legend()
    
    figure = fig

    return figure

def line_plot_func(results02, results1, results5, non_priv_results, syn_results, epsilon):
    metric_list = ['AIR','GCAP', 'ZCAP', 'MDCR', 'HitR', 'MIR', 'NNAA', 'CRP', 'NSND', 'CVP', 'DVP', 'Auth', 'D-MLP', 'IdScore']
    metric_df = pd.DataFrame({'Metric':metric_list})
    non_eps_df = pd.DataFrame({'Epsilon':[11]*len(non_priv_results)})
    eps02_df = pd.DataFrame({'Epsilon':[0.2]*len(non_priv_results)})
    eps1_df = pd.DataFrame({'Epsilon':[1]*len(non_priv_results)})
    eps5_df = pd.DataFrame({'Epsilon':[5]*len(non_priv_results)})
    syn_eps_df = pd.DataFrame({'Epsilon':[epsilon]*len(non_priv_results)})
    
    non_priv_w_eps = pd.concat([metric_df, non_priv_results['Result'], non_eps_df], axis=1)
    priv02_w_eps = pd.concat([metric_df, results02['Result'], eps02_df], axis=1)
    priv1_w_eps = pd.concat([metric_df, results1['Result'], eps1_df], axis=1)
    priv5_w_eps = pd.concat([metric_df, results5['Result'], eps5_df], axis=1)
    syn_w_eps = pd.concat([metric_df, syn_results['Result'], syn_eps_df], axis=1)
    
    df = pd.concat([non_priv_w_eps, syn_w_eps, priv02_w_eps, priv1_w_eps, priv5_w_eps], axis=0)

    title_txt = f'Lineplot of metrics results for ϵ =(0.2, 1, 5, ∞(at epsilon=11), {epsilon})'
    #plt.figure()
    
    df.pivot(index='Epsilon', columns='Metric', values='Result').plot(marker='x', title=title_txt, 
                                                                      alpha=0.5, figsize=(6,5),
                                                                      ylabel='Score', xlabel='ϵ'
                                                                      ).legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    return plt

def synthesize(real_data, eps):
    st.session_state['categorical_attributes'] = {'First_Name': True, 'Last_Name': True, 'Height': False, 'Flavour': True}
    st.session_state['attribute_dtypes'] ={"First_Name": "String", "Last_Name": "String", "Height": "Float", "Flavour": "String"}
    
    describer = DataDescriber(histogram_bins=10, category_threshold=10)
    timestamp = datetime.now()
    real_data.to_csv(f'{timestamp}.csv', index=False)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=f'{timestamp}.csv', 
                                                            epsilon=eps, 
                                                            k=2,
                                                            attribute_to_is_candidate_key={'First_Name': False, 'Last_Name': False, 'Height': False, 'Flavour': False},
                                                            attribute_to_datatype={"First_Name": "String", "Last_Name": "String", "Height": "Float", "Flavour": "String"},
                                                            attribute_to_is_categorical={'First_Name': True, 'Last_Name': True, 'Height': False, 'Flavour': True}
                                                            )
    description = f'descriptions/{timestamp}.json'
    syn_path = f'syn_data/{timestamp}.csv'
    describer.save_dataset_description_to_file(description)
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(n=len(real_data), description_file=description, seed=timestamp)
    generator.save_synthetic_data(syn_path)
    result = pd.read_csv(syn_path, index_col=False).round(2)
    os.remove(f'{timestamp}.csv')
    os.remove(f'descriptions/{timestamp}.json')
    os.remove(f'syn_data/{timestamp}.csv')
    
    return result

def set_state(i):
    st.session_state.stage = i

if st.session_state.stage == 0:
    st.write("This app demonstrates how false privacy estimation can influence YOUR privacy!")
    st.write("Please tell me your information below:")
    st.session_state['first_name'] = st.text_input("First Name")
    st.session_state['last_name'] = st.text_input("Last Name")
    st.session_state['height'] = st.number_input("Height")
    st.session_state['flavour'] = st.text_input("Favorite Icrecream Flavour")
    st.button(label="Submit", on_click=set_state, args=[1])

if st.session_state.stage == 1:
    st.write(":exclamation: :exclamation: OH NO :exclamation: :exclamation:")
    st.session_state.pd = pd.DataFrame(add_sens_individual(st.session_state['first_name'], st.session_state['last_name'], st.session_state['height'], st.session_state['flavour']))
    st.dataframe(st.session_state.pd, use_container_width=True, hide_index=True)
    st.write("YOU'VE JUST SUBMITTED INFORMATION ABOUT YOURSELF TO A DATASET THAT ONLY CONTAINS INFORMATION ABOUT BABIES:exclamation::worried:")
    st.write("Dont worry, Differential Privacy has got you covered :heart_eyes:")
    st.session_state['coord_real_pca'], st.session_state['model_pca'] = fit_transform(st.session_state.pd, nf=2)
    st.button(label="Synthesize The Dataset", on_click=set_state, args=[2])

    syn_data_inf = synthesize(st.session_state.pd, 0)
    st.session_state['non_private'] = syn_data_inf
    
if st.session_state.stage == 2 or st.session_state.stage == 3:
    syn_data02 = synthesize(st.session_state.pd, 0.2)
    st.session_state['syn_data02'] = syn_data02
    
    syn_data1 = synthesize(st.session_state.pd, 1)
    st.session_state['syn_data1'] = syn_data1
    
    syn_data5 = synthesize(st.session_state.pd, 5)
    st.session_state['syn_data5'] = syn_data5

    syn_data = synthesize(st.session_state.pd, st.session_state['epsilon'])
    st.session_state['syn_data'] = syn_data
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Real Dataset:")
        st.dataframe(st.session_state.pd, use_container_width=True, hide_index=True)
        st.pyplot(scatter_plot_real(st.session_state['coord_real_pca']), use_container_width = True)
    with col2:
        st.write("Your Synthetic Dataset:")
        st.dataframe(st.session_state['syn_data'], use_container_width=True, hide_index=True)
        syn_coords_pca = transform(st.session_state['syn_data'], st.session_state['model_pca'])
        st.pyplot(scatter_plot(st.session_state['coord_real_pca'], syn_coords_pca), use_container_width = True)
    with col3:
        st.write("Non-Private Synthetic Dataset:")
        st.dataframe(st.session_state['non_private'], use_container_width=True, hide_index=True)
        syn_coords_pca_priv = transform(st.session_state['non_private'], st.session_state['model_pca'])
        st.pyplot(scatter_plot(st.session_state['coord_real_pca'], syn_coords_pca_priv), use_container_width = True)
    
    st.session_state['epsilon'] = st.slider("$\epsilon$-value (lower = more privacy):", 0.0, 10.0, 2.0)
    st.write("Hmm... Is your sensitive information safe now? 🤔")
    st.write("Let's use privacy metrics to estimate our privacy 💡☝️")
    st.button(label="Show Metric Results", on_click=set_state, args=[3])
    
    
if st.session_state.stage >= 3:
    st.session_state['syn_results'] = get_metric_results(st.session_state.pd, st.session_state['syn_data'])
    st.session_state['non_priv_results'] = get_metric_results(st.session_state.pd, st.session_state['non_private'])
    st.session_state['results02'] = get_metric_results(st.session_state.pd, st.session_state['syn_data02'])
    st.session_state['results1'] = get_metric_results(st.session_state.pd, st.session_state['syn_data1'])
    st.session_state['results5'] = get_metric_results(st.session_state.pd, st.session_state['syn_data5'])
    st.session_state['bar_chart'] = metric_bar_chart(st.session_state['results02'], st.session_state['results1'], 
                                   st.session_state['results5'], st.session_state['non_priv_results'], 
                                   st.session_state['syn_results'], st.session_state['epsilon'])
    st.session_state['line'] = line_plot_func(st.session_state['results02'], st.session_state['results1'], 
                                 st.session_state['results5'], st.session_state['non_priv_results'], 
                                 st.session_state['syn_results'], st.session_state['epsilon'])
    st.write("All metrics shown have been configured to fit in range [0,1], where 0 = full privacy, 1 = no privacy", )
    col1, col2 = st.columns(2)
    #real_data = pd.DataFrame(st.session_state.pd)
    #syn_data = pd.DataFrame(st.session_state['syn_data'])
    #syn_data_inf = pd.DataFrame(st.session_state['non_private'])
    with col1:
        st.pyplot(st.session_state['bar_chart'], use_container_width = True)
    with col2:
        st.pyplot(st.session_state['line'], use_container_width = True)

    st.write("Interested in knowing how these scores are computed for your synthetic data?")
    st.write("Explore how the different metrics are computed, and how there may be issues when using them below:")
    idScore_tab, dmlp_tab, auth_tab, dvp_tab, cvp_tab, nsnd_tab, crp_tab, nnaa_tab, mir_tab, hitr_tab, mdcr_tab, zcap_tab, gcap_tab, air_tab = st.tabs(
        ["IdScore", "D-MLP", "Auth", "DVP", "CVP", "NSND", "CRP", "NNAA", "MIR", "HitR", "MDCR", "ZCAP", "GCAP", "AIR"])
    with idScore_tab:
        st.subheader("Identifiability Score (IdScore):")
        
    with dmlp_tab:
        st.subheader("DetectionMLP (D-MLP):")
        
    with auth_tab:
        st.subheader("Authenticity (Auth):")
        
    with dvp_tab:        
        st.subheader("Distant Value Probability (DVP):")
        
    with cvp_tab:        
        st.subheader("Close Value Probability (CVP):")
        
    with nsnd_tab:        
        st.subheader("Nearest Synthetic Neighbour Distance (NSND):")
        
    with crp_tab:        
        st.subheader("Common Row Proportion (CRP):")
        
    with nnaa_tab:        
        st.subheader("Nearest Neighbour Adversarial Accuracy (NNAA):")
        
    with mir_tab:        
        st.subheader("Membership Inference Risk (MIR):")
        
    with hitr_tab:        
        st.subheader("Hitting Rate (HitR):")
        
    with mdcr_tab:        
        st.subheader("Median Distance to Closest Record (MDCR):")
        
    with zcap_tab:        
        st.subheader("Zero Correct Attribution Probability (ZCAP):")
        
    with gcap_tab:        
        st.subheader("Generalized Correct Attribution Probability (GCAP):")
        
    with air_tab:        
        st.subheader("Attribute Inference Risk (AIR):")
        
    

st.button(label="Start Over", on_click=set_state, args=[0])

