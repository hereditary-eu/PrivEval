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
from sklearn.neighbors import NearestNeighbors
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
    num_samples = 29
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
    your_x = coord_real['Dim. 1'].iloc[len(st.session_state.pd)-1]
    your_y = coord_real['Dim. 2'].iloc[len(st.session_state.pd)-1]
    # Scatter Plot
    plt.figure()
    # Plot DataFrame 1
    plt.scatter(coord_real['Dim. 1'], coord_real['Dim. 2'], color='royalblue', label='Real', alpha=1)
    # Plot DataFrame 2
    plt.scatter(coord_synth['Dim. 1'], coord_synth['Dim. 2'], color='red', label='Synthetic', alpha=0.5)
    plt.scatter(your_x, your_y, color='cyan', edgecolors='black', linewidths=1,  label='You', alpha=0.5)
    
    plt.title('Scatter Plot of real and synthetic data')
    plt.xlabel('X-coord')
    plt.ylabel('Y-coord')
    plt.legend()
    plt.grid(True)

    # Show Plot
    #plt.show()
    return plt

def scatter_plot_real(coord_real):
    your_x = coord_real['Dim. 1'].iloc[len(st.session_state.pd)-1]
    your_y = coord_real['Dim. 2'].iloc[len(st.session_state.pd)-1]
    # Scatter Plot
    plt.figure()
    # Plot DataFrame 1
    plt.scatter(coord_real['Dim. 1'], coord_real['Dim. 2'], color='royalblue', label='Real', alpha=1)
    plt.scatter(your_x, your_y, color='cyan', edgecolors='black', linewidths=1,  label='You', alpha=1)
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

    synthcity_results = All_synthcity.calculate_metric(args = None, _real_data=real_data, _synthetic=real_data, _metrics=metrics)
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
    ax.barh(ind -width*2, results02['Result'], width, color='darkblue', label='ϵ=0.2')
    ax.barh(ind - width, results1['Result'], width, color='mediumblue', label='ϵ=1')
    ax.barh(ind, results5['Result'], width, color='royalblue', label='ϵ=5')
    ax.barh(ind+ width, non_priv_metric_result['Result'], width, color='cornflowerblue', label='ϵ=∞ (Non-Private)')
    ax.barh(ind + width*2, syn_results['Result'], width, color='red', label=f'ϵ={epsilon} (Your Data)')

    ax.set(yticks=ind , yticklabels=non_priv_metric_result['Metric'], xlabel="Risk")
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
                                                                      alpha=0.7, figsize=(6,5),
                                                                      ylabel='Risk', xlabel='ϵ'
                                                                      ).legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    return plt

def synthesize(real_data, eps):
    # st.session_state['categorical_attributes'] = {'First_Name': True, 'Last_Name': True, 'Height': False, 'Flavour': True}
    # st.session_state['attribute_dtypes'] ={"First_Name": "String", "Last_Name": "String", "Height": "Float", "Flavour": "String"}
    
    # describer = DataDescriber()
    # timestamp = datetime.now()
    # real_data.to_csv(f'{timestamp}.csv', index=False)
    # describer.describe_dataset_in_correlated_attribute_mode(dataset_file=f'{timestamp}.csv', 
    #                                                         epsilon=eps, 
    #                                                         k=2,
    #                                                         attribute_to_is_candidate_key={'First_Name': False, 'Last_Name': False, 'Height': False, 'Flavour': False},
    #                                                         attribute_to_datatype={"First_Name": "String", "Last_Name": "String", "Height": "Float", "Flavour": "String"},
    #                                                         attribute_to_is_categorical={'First_Name': True, 'Last_Name': True, 'Height': False, 'Flavour': True}
    #                                                         )
    # description = f'descriptions/{timestamp}.json'
    # syn_path = f'syn_data/{timestamp}.csv'
    # describer.save_dataset_description_to_file(description)
    # generator = DataGenerator()
    # generator.generate_dataset_in_correlated_attribute_mode(n=len(real_data), description_file=description, seed=timestamp)
    # generator.save_synthetic_data(syn_path)
    # result = pd.read_csv(syn_path, index_col=False).round(2)
    # os.remove(f'{timestamp}.csv')
    # os.remove(f'descriptions/{timestamp}.json')
    # os.remove(f'syn_data/{timestamp}.csv')
    
    # instantiate and fit synthesizer
    pb = PrivBayes(epsilon=eps, verbose=False)
    pb.fit(real_data)

    # Synthesize data
    gen_data  = pb.sample()

    # Save to csv file
    result = pd.DataFrame(gen_data.values, columns=gen_data.columns, index=range(real_data.shape[0]))
    
    return result

def nearest_neighbor_hamming(query_df, dataset_df):
    from scipy.spatial.distance import hamming
    
    # Finds the nearest neighbor to the query in the dataset using Hamming distance.
    
    # Parameters:
    # query_df (pd.DataFrame): A DataFrame with a single row representing the query.
    # dataset_df (pd.DataFrame): A DataFrame with multiple rows representing the dataset.
    
    # Returns:
    # tuple: (index of nearest neighbor, nearest neighbor vector, hamming distance)
    
    query = query_df.iloc[0].values
    dataset = dataset_df.values
    
    min_distance = float('inf')
    nearest_index = -1
    nearest_vector = None
    
    for i, data_point in enumerate(dataset):
        distance = hamming(query, data_point) * len(query)  # Convert to absolute distance
        if distance < min_distance:
            min_distance = distance
            nearest_index = i
            nearest_vector = data_point
            
    nearest_neighbor_df = pd.DataFrame([nearest_vector], columns=dataset_df.columns)
    return nearest_index, nearest_neighbor_df, min_distance

def air_nn(r, data, k):
    # k: k nearest neighbours

    # diff_array = np.abs(data - r)
    # diff_array_max = np.amax(diff_array, axis=0)
    # diff_array_max2 = np.maximum(diff_array_max, 1)
    # diff_array_rate = diff_array/diff_array_max2
    # diff = np.sum(diff_array_rate, axis=1)
    # thresh = np.sort(diff)[k-1]
    # idxs = np.arange(len(data))[diff <= thresh]
    
    # return idxs
    # Ensure that data and r are numeric arrays (float type)
    r = np.asarray(r, dtype=float)
    data = np.asarray(data, dtype=float)

    # Calculate the absolute difference between the data and r
    diff_array = np.abs(data - r)
    
    # Max of each column, with a minimum of 1 to avoid division by zero
    diff_array_max = np.amax(diff_array, axis=0)
    diff_array_max2 = np.maximum(diff_array_max, 1)
    
    # Calculate the difference rate
    diff_array_rate = diff_array / diff_array_max2
    
    # Sum the differences across all columns (axis=1)
    diff = np.sum(diff_array_rate, axis=1)
    
    # Find the threshold based on the k-th smallest value
    thresh = np.sort(diff)[k-1]
    
    # Get indices of rows where the difference is less than or equal to the threshold
    idxs = np.arange(len(data))[diff <= thresh]
    
    return idxs

def get_dummy_datasets(real_data, syn_data):
    all_data = pd.concat([real_data, syn_data])
    all = pd.get_dummies(all_data).to_numpy()
    real = all[:real_data.shape[0]]
    fake = all[real_data.shape[0]:]
    
    return real, fake

def set_state(i):
    st.session_state.stage = i

def metric_applicability(real_data, syn_data, metric_results, non_priv_metric_results):
    cat_attributes = ['First_Name', 'Last_Name', 'Height']
    sens_attributes = ['Flavour']
    continuous_columns = [col for col in real_data.columns if pd.api.types.is_numeric_dtype(real_data[col])]
    
    applicability_column = pd.DataFrame(metric_results['Metric'])
    applicability_column['Applicability'] = '✅'
    
    if any(col in real_data[cat_attributes].columns for col in continuous_columns):
        applicability_column.loc[applicability_column['Metric']=='ZeroCAP', 'Applicability'] = '⛔️'
        applicability_column.loc[applicability_column['Metric']=='GeneralizedCAP', 'Applicability'] = '⚠️'
        applicability_column.loc[applicability_column['Metric']=='Attribute Inference Risk', 'Applicability'] = '⚠️'
    if any(col in real_data[sens_attributes].columns for col in continuous_columns):
        applicability_column.loc[applicability_column['Metric']=='ZeroCAP', 'Applicability'] = '⛔️'
        applicability_column.loc[applicability_column['Metric']=='GeneralizedCAP', 'Applicability'] = '⛔️'
        applicability_column.loc[applicability_column['Metric']=='Attribute Inference Risk', 'Applicability'] = '⚠️'
        
    applicability_df = metric_results.merge(applicability_column, on='Metric')
    applicability_df.rename(columns={"Score": "Risk"})
    return applicability_df

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

    syn_data_inf = synthesize(st.session_state.pd, np.inf)
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
    
if st.session_state.stage == 3:
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
    st.write("Do the metrics actually work for your synthetic dataset?")
    st.write("✅: The metric is good, and no assumption is missing")
    st.write("⚠️: The metric requires some assumption which is potentially not met")
    st.write("⛔️: The metric is not reliable in any sense.")
    st.dataframe(metric_applicability(st.session_state.pd, st.session_state['syn_data'], st.session_state['syn_results'], st.session_state['non_priv_results']), hide_index=True)
    col1, col2 = st.columns(2)
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
        col1, col2 = st.columns(2, border=True)
        with col1:
            st.write("MDCR measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated.")
            st.write("The attacker follows four steps to guess a sensitive value:")
            st.write("1. For each real individual find the distance to the nearest neighbour in the real dataset.")
            st.write("2. For each real individual find the distance to the nearest neighbour in the synthetic dataset.")
            st.write("3. Calculate the median of distances between real individuals.")
            st.write("4. Calculate the median of distances between real and synthetic individuals.")
            st.write("The MDCR is then calculated as:")
            st.latex(r'''\frac{\mu(dists\phantom{i}real\phantom{i}to\phantom{i}real)}
                                  {\mu(dists\phantom{i}real\phantom{i}to\phantom{i}synthetic)}''')
            st.write("Your nearest neighbours for this metric is:")
            
            all_data = pd.concat([st.session_state.pd, st.session_state['syn_data']])
            fn_encoder = LabelEncoder()
            ln_encoder = LabelEncoder()
            fl_encoder = LabelEncoder()
            r_fn = fn_encoder.fit_transform(all_data['First_Name'])
            r_ln = ln_encoder.fit_transform(all_data['Last_Name'])
            r_fl = fl_encoder.fit_transform(all_data['Flavour'])
            all_labels = pd.DataFrame({'First_Name':r_fn, 'Last_Name': r_ln, 'Height': all_data['Height'],'Flavour':r_fl})
            real_labels = all_labels[:len(st.session_state.pd)]
            syn_labels = all_labels[-len(st.session_state.pd):]
            nn = NearestNeighbors(n_neighbors=2)
            d_real = []
            d_syn = []
            nn.fit(real_labels)
            dists_real, idxs_real = nn.kneighbors(real_labels)
            dists_syn, idxs_syn = nn.kneighbors(syn_labels)
            for i in range(2):
                d_real.append(dists_real[:,i])
                d_syn.append(dists_syn[:,i])
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.write("Real neighbour:")
                st.dataframe(st.session_state.pd.iloc[[idxs_real[len(st.session_state.pd)-1, 1]]], use_container_width=True, hide_index=True)
                st.write(f"With distance: {round(dists_real[len(st.session_state.pd)-1, 1], 2)}")
            with col1_2:
                st.write("Synthetic neighbour:")
                
                
        with col2:
            st.write("Real Dataset:")
            st.dataframe(st.session_state.pd, use_container_width=True, hide_index=True)
            st.write("Your Synthetic Dataset:")
            st.dataframe(st.session_state['syn_data'], use_container_width=True, hide_index=True)
            #st.session_state.pd
            
    with zcap_tab:        
        st.subheader("Zero Correct Attribution Probability (ZCAP):")
        col1, col2 = st.columns(2, border=True)
        with col1:
            st.write("ZCAP measures the risk of inference attacks by assessing how easily an attacker, using public real data and synthetic data, can infer sensitive values. It quantifies this difficulty with the Correct Attribution Probability (CAP) algorithm.")
            st.write("The attacker follows four steps to guess a sensitive value:")
            st.write("1. Select a row from the real dataset and note its key fields.")
            st.write("2. Find all synthetic rows matching these key fields (the synthetic equivalence class).")
            st.write("*If there are no matching key fields, the row has a score of 0.*")
            st.write("3. Use the sensitive values in to vote on the real row’s sensitive fields.")
            st.write("4. The final score is the proportion of correct votes, ranging from 0 to 1.")
            st.write("This attack is repeated for all rows in the real dataset, and the score is an overall probability of guessing the sensitive column correctly.")
            st.write("For your data, the key fields are:")
            cat_attributes = ['First_Name', 'Last_Name', 'Height']
            ind_vals = st.session_state.pd[cat_attributes].iloc[[len(st.session_state.pd)-1]]
            st.dataframe(ind_vals, hide_index=True)
            syndat=st.session_state['syn_data']
            if any((ind_vals == syndat[cat_attributes].iloc[i]).all(axis=1).any() for i in range(len(syndat[cat_attributes]))):
                st.write("These rows with matching key fields in the synthetic dataset:")
                matching_rows = st.session_state['syn_data'][st.session_state['syn_data'].apply(lambda row: (ind_vals == row[cat_attributes]).all(axis=1).any(), axis=1)]
                st.dataframe(matching_rows, hide_index=True)
                st.write("Your row contributes:")
                st.latex(r'''\frac{|rows\phantom{i}with\phantom{i}same\phantom{i}key\phantom{i}and\phantom{i}sensitive\phantom{i}fields|}
                                  {|rows\phantom{i}with\phantom{i}same\phantom{i}key\phantom{i}fields|}''')
                st.write("The problem that could occurs here is, that have continuous attributes (height) in either the key fields or sensitive fields.")
                st.write("Therefore, the randomness induced by the synthesizer makes finding a match highly unlikely.")
                
            if not any((ind_vals == syndat[cat_attributes].iloc[i]).all(axis=1).any() for i in range(len(syndat[cat_attributes]))):
                st.write("There are not any synthetic rows with matching key fields.")
                st.write("Therefore, the score for your data is 0.")
                st.write("The problem that occurs here is, that we have continuous attributes in either the key fields or sensitive fields.")
                st.write("Therefore, the randomness induced by the synthesizer makes finding a match highly unlikely.")
            
        with col2:
            st.write("Real Dataset:")
            st.dataframe(st.session_state.pd, use_container_width=True, hide_index=True)
            st.write("Your Synthetic Dataset:")
            st.dataframe(st.session_state['syn_data'], use_container_width=True, hide_index=True)
            #st.session_state.pd
        
    with gcap_tab:        
        st.subheader("Generalized Correct Attribution Probability (GCAP):")
        col1, col2 = st.columns(2, border=True)
        with col1:
            st.write("GCAP measures the risk of inference attacks by assessing how easily an attacker, using public real data and synthetic data, can infer sensitive values. It quantifies this difficulty with the Correct Attribution Probability (CAP) algorithm.")
            st.write("The attacker follows four steps to guess a sensitive value:")
            st.write("1. Select a row from the real dataset and note its key fields.")
            st.write("2. Find all synthetic rows matching these key fields (the synthetic equivalence class).")
            st.write("*If there are no matching key fields, the nearest synthetic neighbours using Hamming distance on the key fields are chosen*")
            st.write("3. Use the sensitive values in to vote on the real row’s sensitive fields.")
            st.write("4. The final score is the proportion of correct votes, ranging from 0 to 1.")
            st.write("This attack is repeated for all rows in the real dataset, and the score is an overall probability of guessing the sensitive column correctly.")
            st.write("For your data, the key fields are:")
            cat_attributes = ['First_Name', 'Last_Name', 'Height']
            ind_vals = st.session_state.pd[cat_attributes].iloc[[len(st.session_state.pd)-1]]
            st.dataframe(ind_vals, use_container_width=True, hide_index=True)
            syndat=st.session_state['syn_data']
            if any((ind_vals == syndat[cat_attributes].iloc[i]).all(axis=1).any() for i in range(len(syndat[cat_attributes]))):
                st.write("These rows have matching key fields in the synthetic dataset:")
                matching_rows = st.session_state['syn_data'][st.session_state['syn_data'].apply(lambda row: (ind_vals == row[cat_attributes]).all(axis=1).any(), axis=1)]
                st.dataframe(matching_rows, hide_index=True, use_container_width=True)
                st.write("Your row contributes:")
                st.latex(r'''\frac{|rows\phantom{i}with\phantom{i}same\phantom{i}key\phantom{i}and\phantom{i}sensitive\phantom{i}fields|}
                                  {|rows\phantom{i}with\phantom{i}same\phantom{i}key\phantom{i}fields|}''')
                st.write("**The problems that may occur:**")
                st.write("1. Your contribution to the score will not be very influential if no other matches are found, meaning that the score will be close to 0.")
                st.write("2. There are continuous attributes in either the key fields or sensitive fields. Therefore, the randomness induced by the synthesizer makes finding matching key fields very unlikely.")
            
            else:
                neighbour_index, neighbour, distance = nearest_neighbor_hamming(ind_vals, syndat[cat_attributes])
                st.write("This row is your nearest synthetic neighbouring key fields:")
                st.dataframe(neighbour, use_container_width=True, hide_index=True)
                st.write("These are the sensitive fields for both individuals:")
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    st.dataframe(pd.DataFrame({'Your Sensitive Field': st.session_state.pd['Flavour'].iloc[[len(st.session_state.pd)-1]]}), use_container_width=True, hide_index=True)
                with col1_2:
                    st.dataframe(pd.DataFrame({"Neighbour's Sensitive Field": syndat['Flavour'].iloc[[neighbour_index]]}), use_container_width=True, hide_index=True)
                
                if syndat['Flavour'].values[neighbour_index] == st.session_state.pd['Flavour'].values[len(st.session_state.pd)-1]:
                    st.write("You have the same sensitive field, and your privacy is therefore in jeopardy.")
                    st.write("You therefore contribute a score of 1 to the metric calculation.")
                    st.write("**The problems that may occur:**")
                    st.write("1. Your contribution to the score will not be very influential if no other matches are found, meaning that the score will be close to 0.")
                    st.write("2. There are continuous attributes in either the key fields or sensitive fields. Therefore, finding a neighbour is influenced differently for height. This may be vissible from the key fields of your second nearest neighbour:")
                    neighbour_index1, neighbour1, distance1 = nearest_neighbor_hamming(ind_vals, syndat[cat_attributes].drop([syndat[cat_attributes].index[neighbour_index]]))
                    st.dataframe(neighbour1, use_container_width=True, hide_index=True)
                
                else:
                    st.write("You do not have the same sensitive field, and your privacy is maintained.")
                    st.write("You therefore contribute a score of 0 to the metric calculation.")
                    st.write("**The problems that may occur:**")
                    st.write("1. Your contribution to the score will not be very influential if no other matches are found, meaning that the score will be close to 0.")
                    st.write("2. There are continuous attributes in either the key fields or sensitive fields. Therefore, finding a neighbour is influenced differently for height. This may be vissible from the key fields of your 2nd nearest neighbour:")
                    neighbour_index1, neighbour1, distance1 = nearest_neighbor_hamming(ind_vals, syndat[cat_attributes].drop([syndat[cat_attributes].index[neighbour_index]]))
                    st.dataframe(neighbour1, use_container_width=True, hide_index=True)
                    
                    
        with col2:
            st.write("Real Dataset:")
            st.dataframe(st.session_state.pd, use_container_width=True, hide_index=True)
            st.write("Your Synthetic Dataset:")
            st.dataframe(st.session_state['syn_data'], use_container_width=True, hide_index=True)
            
    with air_tab:        
        st.subheader("Attribute Inference Risk (AIR):")
        col1, col2 = st.columns(2, border=True)
        with col1:
            st.write("AIR measures the risk of inference attacks by assessing how easily an attacker, using public real data and synthetic data, can infer sensitive values. It quantifies this difficulty with the a weighted F1-score.")
            st.write("*To calculate this metric, a one-hot encoding for categorical attributes must be used, as it only works for binary and continuous data.*")
            st.write("The attacker follows four steps to guess a sensitive value:")
            st.write("1. Select a row from the real dataset and note its key fields.")
            st.write("2. Find the (k=1) nearest synthetic neighbour(s) using a normalized Hamming distance on the key fields.")
            st.write("3. Evaluate the binary and continuous attributes seperately for infering the sensitive fields.")
            st.write("i. Binary attributes: Computes true positives, false positives, false negatives.")
            st.write("ii. Continuous attributes: Checks if predictions are within ±10% of actual values.")
            st.write("4. Compute the F1-Score according to:")
            st.latex(r'''F_{1}(row) = \frac{2 * \frac{TP}{TP+FP} * \frac{TP}{TP+FN}}{\frac{TP}{TP+FP} + \frac{TP}{TP+FN}}''')
            st.write("This attack is repeated for all rows in the real dataset, and the score is weighted performance in predicting the sensitive column, which is calculated as:")
            st.latex(r'''AIR = \sum\limits_{row \in Real Dataset} w(row) * F_{1}(row)''')
            st.write("where the w(row) is is the weight, such that:")
            st.latex(r'''w(row) = \frac{Pr(row)  \phantom{i} log (Pr(row))}{\sum\limits_{r \in Real Dataset} Pr(r)  \phantom{i} log (Pr(r))}''')
            #, and the score is an overall probability of guessing the sensitive column correctly.
            st.write("**For your data**")
            st.write("The key fields are:")
            cat_attributes = ['First_Name', 'Last_Name', 'Height']
            ind_vals = st.session_state.pd[cat_attributes].iloc[[len(st.session_state.pd)-1]]
            st.dataframe(ind_vals, use_container_width=True, hide_index=True)
            syndat=st.session_state['syn_data']
            dummy_real_cat, dummy_syn_cat = get_dummy_datasets(st.session_state.pd[cat_attributes], syndat[cat_attributes])
            dummy_real, dummy_syn = get_dummy_datasets(st.session_state.pd, syndat)
            dummy_ind_vals = dummy_real_cat[len(st.session_state.pd)-1]
            idx = air_nn(dummy_ind_vals, dummy_syn_cat, k=1)
            idx2 = air_nn(dummy_ind_vals, dummy_syn_cat, k=2)[0]
            st.write("The key Fields of nearest synthetic neighbour(s) using a normalized Hamming distance is:")
            st.dataframe(syndat[cat_attributes].iloc[idx], use_container_width=True, hide_index=True)
            st.write("These are the sensitive fields for both individuals:")
            dummy_real_indv, dummy_syn_indv = get_dummy_datasets(st.session_state.pd['Flavour'], syndat['Flavour'])
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.dataframe(pd.DataFrame({'Your Sensitive Field': st.session_state.pd['Flavour'].iloc[[len(st.session_state.pd)-1]]}), use_container_width=True, hide_index=True)
            with col1_2:
                st.dataframe(pd.DataFrame({"Neighbour's Sensitive Field": syndat['Flavour'].iloc[idx]}), use_container_width=True, hide_index=True)
            col1_1_1, col1_2_1 = st.columns(2)
            with col1_1_1:
                st.write("(One-Hot encoded)")
                st.write(dummy_real_indv[[len(st.session_state.pd)-1]])
                
            with col1_2_1:
                st.write("(One-Hot encoded)")
                st.write(dummy_syn_indv[idx])
                
            real_label = np.array(dummy_real_indv[[len(st.session_state.pd)-1]])
            pred_label = np.array(dummy_syn_indv[idx])
            match = (real_label == 1) & (pred_label == 1)
            row_counts = st.session_state.pd[cat_attributes].value_counts(normalize=True)
            prob_df = st.session_state.pd[cat_attributes]
            prob_df['Probability'] = prob_df.apply(lambda row: row_counts[tuple(row)], axis=1)
            safe_probs = np.clip(prob_df['Probability'], 1e-10, None)
            numerator = safe_probs * np.log(safe_probs)
            denominator = numerator.sum()
            prob_df['Weight'] = numerator / denominator
            precision = round(match.sum() / (match.sum()+(len(pred_label)-match.sum())), 2)
            recall = 1
            f_one = round((2*precision*recall) / (precision+recall), 2)
            if np.any(match):
                st.write("You and your neighbour have matching sensitive fields!")
                st.write("Your contribution to the score is therefore:")
            else:
                st.write("You and your neighbour do not have matching sensitive fields.")
                st.write("Your contribution to the score is therefore:")
            st.latex(r"precision = \frac{"rf"{match.sum()}"r"}{"rf"{match.sum()}"r"+"rf"{(len(pred_label)-match.sum())}"r"} = "rf"{precision}")
            st.latex(rf"recall = {recall}"r",\phantom{i}as\phantom{i}no\phantom{i}false\phantom{i}negatives.")
            st.latex(r"F_1 = \frac{2*"rf"{precision}"r"*"rf"{recall}"r"}{"rf"{precision}"r"+"rf"{recall}"r"} = "rf"{f_one}")
            st.latex(r"weight = \frac{"rf"{abs(round(numerator[len(st.session_state.pd)-1], 2))}"r"}{"rf"{abs(round(denominator, 2))}"r"} = {"rf"{prob_df['Weight'].iloc[len(st.session_state.pd)-1]}"r"}")
            st.latex(rf'''AIR = {round((2*precision*recall) / (1+recall), 2)}*{prob_df['Weight'].iloc[len(st.session_state.pd)-1]}= {round(f_one*(prob_df['Weight'].iloc[len(st.session_state.pd)-1]), 2)}''')
            st.write("**The problems that may occur:**")
            st.write("1. Your contribution to the score will not be very influential if no other matches are found, meaning that the score will be close to 0.")
            st.write("2. There are continuous attributes in either the key fields or sensitive fields. Therefore, finding a neighbour is influenced differently for height. This may be vissible from the key fields of your 2nd nearest neighbour:")
            st.dataframe(syndat[cat_attributes].iloc[[idx2]], use_container_width=True, hide_index=True)
        with col2:
            st.write("Real Dataset:")
            st.dataframe(st.session_state.pd, use_container_width=True, hide_index=True)
            st.write("Your Synthetic Dataset:")
            st.dataframe(st.session_state['syn_data'], use_container_width=True, hide_index=True)
            st.write("Real Dataset (One-Hot Encoded):")
            st.write(dummy_real)
            st.write("Your Synthetic Dataset (One-Hot Encoded):")
            st.write(dummy_syn)
    



