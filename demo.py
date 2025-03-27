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
from sklearn.manifold import TSNE
from sklearn.preprocessing import OrdinalEncoder
from scipy.spatial.distance import cdist
# Show the page title and description.
st.set_page_config(page_title="Privacy Advisor", page_icon="😈", layout='wide')

#st.session_state.stage = 0
def set_state(i):
    st.session_state.stage = i
    
def generate_real_data(num_samples, liquorice):
    #Code for generating a baby dataset
    name_gen = Faker()
    heights = np.around(list(np.random.normal(loc=170, scale=10, size=num_samples)), 2)
    classic_icecreams = [
        "Vanilla", "Chocolate", "Strawberry", "Mint Chocolate Chip",
        "Cookies and Cream", "Rocky Road", "Butter Pecan", "Neapolitan",
        "Pistachio", "French Vanilla"
    ]
    fav_icecream = list(random.choices(classic_icecreams, k=num_samples))

    # Generate random first and last names
    name_df = pd.DataFrame({
        'First Name': [name_gen.first_name() for _ in range(num_samples)],
        'Last Name': [name_gen.last_name() for _ in range(num_samples)]
    })
    height_df = pd.DataFrame({'Height': heights})
    icecream_df = pd.DataFrame({'Flavour': fav_icecream})
    basic_df = pd.concat([name_df, height_df, icecream_df], axis=1)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Define country list and correlation rules
    countries = ["USA", "Canada", "Germany", "France", "Italy", "China", "Brazil", "Australia", "Japan", "UK", "Sweden", "Norway", "Denmark", "Finland"]

    # Ice cream preferences (default: random choice)
    ice_creams = ["Vanilla", "Chocolate", "Strawberry", "Mint", "Pistachio", "Stracciatella"]

    # Generate data
    data = []
    for i in range(num_samples):
        person = {}

        # Assign country
        person["Country of Origin"] = random.choice(countries)

        # Assign favorite ice cream with correlation (Italy → Stracciatella preference)
        if person["Country of Origin"] == "Italy":
            person["Favorite Icecream"] = np.random.choice(ice_creams, p=[0.1, 0.1, 0.1, 0.1, 0.2, 0.4])
        else:
            person["Favorite Icecream"] = random.choice(ice_creams)

        # Assign liking for liquorice (Nordic countries → Higher probability)
        if person["Country of Origin"] in ["Sweden", "Norway", "Denmark", "Finland"]:
            person["Like Liquorice"] = np.random.choice([1, 0], p=[0.9, 0.1])  # 70% chance for Nordic countries
        else:
            person["Like Liquorice"] = np.random.choice([1, 0], p=[0.2, 0.8])  # 20% for others

        # Assign number of times visited Italy (Random integer, but higher if from Europe)
        if person["Country of Origin"] in ["Germany", "France", "UK", "Sweden", "Norway", "Denmark", "Finland", "Italy"]:
            person["Times Visited Italy"] = np.random.poisson(2)  # Higher average visits
        else:
            person["Times Visited Italy"] = np.random.poisson(0.5)  # Lower average visits

        # First time in London (UK residents more likely to say yes)
        person["First Time London"] = 1 if person["Country of Origin"] == "UK" else np.random.choice([1, 0], p=[0.2, 0.8])

        # Number of steps per day (Normal distribution with realistic values)
        person["Steps per Day"] = max(1000, int(np.random.normal(8000, 3000)))  # Avoids negative steps

        data.append(person)

    # Create DataFrame
    df = pd.DataFrame(data)
    
    full_df = pd.concat([basic_df, df], axis=1)
    
    if liquorice == 0:
        # Sample row: UK resident who does NOT like liquorice
        indiv = {
            "First Name": "James",
            "Last Name": "Smith",
            "Height": round(random.gauss(175, 10), 2),
            "Country of Origin": "UK",
            "Favorite Icecream": "Strawberry",
            "Like Liquorice": 0,
            "Times Visited Italy": 2,
            "First Time London": 0,
            "Steps per Day": 7500
        }

    if liquorice == 1:
        # Sample row: Sweden resident who LIKES liquorice
        indiv = {
            "First Name": "Lars",
            "Last Name": "Andersson",
            "Height": round(random.gauss(185, 10), 2), 
            "Country of Origin": "Sweden",
            "Favorite Icecream": "Chocolate",
            "Like Liquorice": 1,
            "Times Visited Italy": 3,
            "First Time London": 0,
            "Steps per Day": 8000
        }
    full_df = pd.concat([full_df, indiv], ignore_index=True)
        
    # Save to CSV (optional)
    full_df.to_csv("sample_people_data.csv", index=False)
def get_data(like_liquorice, epsilon):
    st.session_state.real_data = pd.read_csv(f'sample_data_{like_liquorice}.csv', index_col=False)
    st.session_state.syn_data_bin = pd.read_csv(f'demo_syn/syn_no_{like_liquorice}_{epsilon}.csv', index_col=False).drop(columns=['Unnamed: 0']) #They got switched around during synthesis
    st.session_state.syn_data_no_bin = pd.read_csv(f'demo_syn/syn_bin_{like_liquorice}_{epsilon}.csv', index_col=False).drop(columns=['Unnamed: 0'])#They got switched around during synthesis
    st.session_state.metric_results_bin = pd.read_csv(f'metric_results/syn_no_{like_liquorice}_{epsilon}.csv', index_col=False).drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2'])#They got switched around during synthesis
    st.session_state.metric_results_no_bin = pd.read_csv(f'metric_results/syn_bin_{like_liquorice}_{epsilon}.csv', index_col=False).drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])#They got switched around during synthesis
    st.session_state.epsilon = epsilon
    #recalculate CRP
    intersection = (st.session_state.real_data.merge(st.session_state.syn_data_bin, how="inner", indicator=False).drop_duplicates())
    st.session_state.metric_results_bin.loc[st.session_state.metric_results_bin['Metric']=='Common Row Proportion', 'Result'] = len(intersection) / (len(st.session_state.real_data) + 1e-8)
    st.session_state.coord_real, model_pca = fit_transform(st.session_state.real_data, nf=2)
    st.session_state.syn_coords = transform(st.session_state.syn_data_bin, model_pca)
    all_data = pd.concat([st.session_state.real_data, st.session_state.syn_data_bin])
    fn_encoder = LabelEncoder()
    ln_encoder = LabelEncoder()
    fl_encoder = LabelEncoder()
    na_encoder = LabelEncoder()
    r_fn = fn_encoder.fit_transform(all_data['First Name'])
    r_ln = ln_encoder.fit_transform(all_data['Last Name'])
    r_na = na_encoder.fit_transform(all_data['Nationality'])
    r_fl = fl_encoder.fit_transform(all_data['Favorite Icecream'])
    all_labels = pd.DataFrame({'First Name':r_fn, 'Last Name': r_ln, 'Height': all_data['Height'],'Nationality': r_na, 'Favorite Icecream':r_fl, 'Like Liquorice': all_data['Like Liquorice'], 'Times Been to Italy': all_data['Times Been to Italy'], 'First Time London': all_data['First Time London'], 'Steps per Day': all_data['Steps per Day']})
    st.session_state.real_labels = all_labels[:len(st.session_state.real_data)]
    st.session_state.syn_labels = all_labels[-len(st.session_state.real_data):]
    tsne = TSNE(n_components=2)
    st.session_state.real_coords_tsne = tsne.fit_transform(st.session_state.real_labels)
    st.session_state.syn_coords_tsne = tsne.fit_transform(st.session_state.syn_labels)
    st.session_state.tsne_df_real = pd.DataFrame(st.session_state.real_coords_tsne)
    st.session_state.tsne_df_syn = pd.DataFrame(st.session_state.syn_coords_tsne)

def scatter_plot_real(coord_real):
    your_x = coord_real['Dim. 1'].iloc[st.session_state.indiv_index]
    your_y = coord_real['Dim. 2'].iloc[st.session_state.indiv_index]
    # Scatter Plot
    plt.figure()
    # Plot DataFrame 1
    plt.scatter(coord_real['Dim. 1'], coord_real['Dim. 2'], marker='.',color='royalblue', label='Real', alpha=1)
    plt.scatter(your_x, your_y, marker='X', color='gold', edgecolors='k', linewidth=1,s=75,  label='You', alpha=1)
    # Plot DataFrame 2
    plt.title('Scatter plot of real data (PCA)')
    plt.xlabel('X-coord')
    plt.ylabel('Y-coord')
    plt.legend()
    plt.grid(True)

    # Show Plot
    #plt.show()
    return plt
def scatter_plot_real_tsne(coord_real):
    real = pd.DataFrame(coord_real)
    your_x = real[0].iloc[st.session_state.indiv_index]
    your_y = real[1].iloc[st.session_state.indiv_index]
    # Scatter Plot
    plt.figure()
    # Plot DataFrame 1
    plt.scatter(real[0], real[1], color='royalblue',marker='.',label='Real', alpha=0.5)
    plt.scatter(your_x, your_y, marker='X', color='gold', edgecolors='k', linewidth=1,s=75,  label='You', alpha=1)
    # Plot DataFrame 2
    plt.title('Scatter plot of real data (t-SNE)')
    plt.xlabel('X-coord')
    plt.ylabel('Y-coord')
    plt.legend()
    plt.grid(True)

    return plt
def scatter_plot(coord_real, coord_synth):
    if len(coord_real)>5:
        your_x = coord_real['Dim. 1'].iloc[st.session_state.indiv_index]
        your_y = coord_real['Dim. 2'].iloc[st.session_state.indiv_index]
    else:
        your_x = coord_real['Dim. 1'].iloc[len(coord_real)-1]
        your_y = coord_real['Dim. 2'].iloc[len(coord_real)-1]
        
    # Scatter Plot
    plt.figure()
    # Plot DataFrame 1
    plt.scatter(coord_real['Dim. 1'], coord_real['Dim. 2'], marker='.',color='royalblue', label='Real', alpha=1)
    # Plot DataFrame 2
    plt.scatter(coord_synth['Dim. 1'], coord_synth['Dim. 2'],marker=7, color='red', label='Synthetic', alpha=0.5)
    plt.scatter(your_x, your_y, marker='X', color='gold', edgecolors='k', linewidth=1,s=75,  label='You', alpha=1)
    
    plt.title('Scatter plot of real and synthetic data (PCA)')
    plt.xlabel('X-coord')
    plt.ylabel('Y-coord')
    plt.legend()
    plt.grid(True)

    # Show Plot
    #plt.show()
    return plt
def scatter_plot_tsne(coord_real, coord_synth):
    real = pd.DataFrame(coord_real)
    syn = pd.DataFrame(coord_synth)
    if len(coord_real)>5:
        your_x = real[0].iloc[st.session_state.indiv_index]
        your_y = real[1].iloc[st.session_state.indiv_index]
    else:
        your_x = real[0].iloc[len(coord_real)-1]
        your_y = real[1].iloc[len(coord_real)-1]
    # Scatter Plot
    plt.figure()

    # Plot DataFrame 1
    plt.scatter(real[0], real[1], marker='.', color='royalblue', label='Real', alpha=0.5)

    # Plot DataFrame 2
    plt.scatter(syn[0], syn[1], marker=7,color='red', label='Synthetic', alpha=0.5)
    
    #Plot you
    plt.scatter(your_x, your_y, marker='X', color='gold', edgecolors='k', linewidth=1,s=75,  label='You', alpha=1)
    

    plt.title('Scatter plot of real and synthetic data (t-SNE)')
    plt.xlabel('X-coord')
    plt.ylabel('Y-coord')
    plt.legend()
    plt.grid(True)

    return plt
def scatter_plot_tsne_old_new(coord_real, coord_synth_old, coord_synth_new):
    real = pd.DataFrame(coord_real)
    syn_old = pd.DataFrame(coord_synth_old)
    syn_new = pd.DataFrame(coord_synth_new)
    if len(coord_real)>5:
        your_x = real[0].iloc[st.session_state.indiv_index]
        your_y = real[1].iloc[st.session_state.indiv_index]
    else:
        your_x = real[0].iloc[len(coord_real)-1]
        your_y = real[1].iloc[len(coord_real)-1]
    # Scatter Plot
    plt.figure()

    # Plot DataFrame 1
    plt.scatter(real[0], real[1], marker='.', color='royalblue', label='Real', alpha=0.5)

    # Plot DataFrame 1
    plt.scatter(syn_old[0], syn_old[1], marker=7, color='green', label=f'Prev. Synthetic', alpha=0.1)
    
    # Plot DataFrame 2
    plt.scatter(syn_new[0], syn_new[1], marker=7,color='red', label=f'New Synthetic', alpha=0.5)
    
    #Plot you
    plt.scatter(your_x, your_y, marker='X', color='gold', edgecolors='k', linewidth=1,s=75,  label='You', alpha=1)
    

    plt.title('Scatter plot of real and synthetic data (t-SNE)')
    plt.xlabel('X-coord')
    plt.ylabel('Y-coord')
    plt.legend()
    plt.grid(True)

    return plt
def scatter_plot_tsne_non_u(coord_real, coord_synth):
    real = pd.DataFrame(coord_real)
    syn = pd.DataFrame(coord_synth)
    
    # Scatter Plot
    plt.figure()

    # Plot DataFrame 1
    plt.scatter(real[0], real[1], marker='.',color='royalblue', label='Real', alpha=0.5)

    # Plot DataFrame 2
    plt.scatter(syn[0], syn[1], marker=7,color='red', label='Synthetic', alpha=0.5)
    

    plt.title('Scatter plot of real and synthetic data (t-SNE)')
    plt.xlabel('X-coord')
    plt.ylabel('Y-coord')
    plt.legend()
    plt.grid(True)

    return plt
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

def _create_matrix_with_ones(indices, num_rows):
    matrix = np.zeros((len(indices),num_rows), dtype=int)
    for i, index in enumerate(indices):
        matrix[i,index] = 1
    return matrix
def _gower_matrix_sklearn(data_x, data_y=None, cat_features: list = None, weights=None, num_attribute_ranges=None, nums_metric='L1'):
    """Modified version of the python gower distance metric implementation
    url: https://pypi.org/project/gower/"""

    X = data_x
    if data_y is None: Y = data_x 
    else: Y = data_y 

    if not isinstance(X, np.ndarray): X = np.asarray(X)
    if not isinstance(Y, np.ndarray): Y = np.asarray(Y)

    x_n_rows, x_n_cols = X.shape
    y_n_rows, y_n_cols = Y.shape 
    
    out_shape = np.zeros((x_n_rows, y_n_rows), dtype=np.float32)

    ### Bit to infer, cat_features if nothing is supplied 
    if cat_features is None:
        if not isinstance(X, np.ndarray): 
            is_number = np.vectorize(lambda x: not np.issubdtype(x, np.number))
            cat_features = is_number(X.dtypes)    
        else:
            cat_features = np.zeros(x_n_cols, dtype=bool)
            for col in range(x_n_cols):
                if not np.issubdtype(type(X[0, col]), np.number):
                    cat_features[col]=True
    else:          
        cat_features = np.array(cat_features)

    ### Separate out weights
    if weights is None:
        weights = np.ones(X.shape[1])
            
    weights_cat = weights[cat_features]
    weights_num = weights[np.logical_not(cat_features)]

    ### Subsetting
    Z = np.concatenate((X,Y))
    
    x_index = range(0,x_n_rows)
    y_index = range(x_n_rows,x_n_rows+y_n_rows)
    
    Z_num = Z[:,np.logical_not(cat_features)]
    Z_cat = Z[:,cat_features]

    ### Make the denominator for the nummerical normalisation 
    if num_attribute_ranges is None:
        num_attribute_ranges = np.max(np.stack((np.array(np.ptp(Z_num,axis=0),dtype=np.float64),np.ones(len(weights_num)))),axis=0)

    X_num = Z_num[x_index,]
    Y_num = Z_num[y_index,]
    ### Do the nummerical step
    if not np.array_equal(cat_features,np.ones(X.shape[1])):
        if nums_metric == 'L1':
                nums_sum = cdist(X_num.astype(float), Y_num.astype(float), 'minkowski', p=1, w=(weights_num/num_attribute_ranges))

        elif nums_metric == 'EXP_L2':
                nums_sum = cdist(X_num.astype(float), Y_num.astype(float), 'minkowski', p=2, w=(weights_num/num_attribute_ranges**2))#/np.sqrt(len(weights_num))

        else: raise NotImplementedError("The keyword literal is not a valid!")
    else: nums_sum = out_shape
    
    ### Do the categorical step
    if not np.array_equal(cat_features,np.zeros(X.shape[1])):
        Z_cat_enc = OrdinalEncoder().fit_transform(Z_cat)

        X_cat = Z_cat_enc[x_index,]
        Y_cat = Z_cat_enc[y_index,]

        cat_sum = cdist(X_cat.astype(int),Y_cat.astype(int), 'hamming', w=weights_cat)*len(weights_cat)
    else: cat_sum = out_shape
    
    return (nums_sum+cat_sum)/weights.sum()
def gower_knn(a, b, num):
    gower_variant = 'L1'
    
    d = []  # List to store distances
    idxs = []  # List to store indices of nearest neighbors
    
    if np.array_equal(a, b):
        matrix = _gower_matrix_sklearn(a, nums_metric=gower_variant) + np.eye(len(a))
        for _ in range(num):
            min_idx = matrix.argmin(axis=1)  # Find nearest neighbor indices
            min_dist = matrix.min(axis=1)  # Find nearest neighbor distances
            
            d.append(min_dist)
            idxs.append(min_idx)
            
            matrix += _create_matrix_with_ones(min_idx[:, np.newaxis], len(a))  # Mask out found neighbors
    else:
        matrix = _gower_matrix_sklearn(a, b, nums_metric=gower_variant)
        for _ in range(num):
            min_idx = matrix.argmin(axis=1)
            min_dist = matrix.min(axis=1)
            
            d.append(min_dist)
            idxs.append(min_idx)
            
            matrix += _create_matrix_with_ones(min_idx[:, np.newaxis], len(b))
    
    return np.array(d).T, np.array(idxs).T

def has_problematic_synthetic_neighbors():
        threshold = 0.5  # Define the threshold for "about the same distance"
        nn_real = NearestNeighbors(n_neighbors=len(st.session_state.tsne_df_syn))  
        nn_real.fit(st.session_state.tsne_df_real)

        # Step 1: Fit NearestNeighbors to find real nearest neighbors
        nn_real_point = NearestNeighbors(n_neighbors=2)  # 2 to include itself and its nearest neighbor
        nn_real_point.fit(st.session_state.tsne_df_real)

        # Step 2: Iterate over all real data points
        for i in range(len(st.session_state.tsne_df_real)):
            real_point = st.session_state.tsne_df_real.iloc[[i]]  # Keep as DataFrame
            dists_real, _ = nn_real.kneighbors(real_point)

            # Find the distance to the nearest real neighbor
            dist_real_nn, _ = nn_real_point.kneighbors(real_point)
            max_distance = dist_real_nn[0, 1]  # The nearest neighbor distance (skip the first which is itself)

            # Find synthetic neighbors that meet both conditions
            dists = dists_real[0]  # Get distances to synthetic points
            valid_synthetic_points = np.where((np.abs(dists - dists[1]) < threshold) & (dists <= max_distance))[0]

            # If any real point fails the condition, return False
            if len(valid_synthetic_points) < 3:
                return False

        return True  # All real points have sufficient synthetic neighbors
    
def air_no_prot():
    key_fields = ['First Name', 'Last Name', 'Height', 'Nationality', 'Favorite Icecream', 'Times Been to Italy', 'First Time London', 'Steps per Day']
    syndat=st.session_state.syn_data_bin
    dummy_real_cat, dummy_syn_cat = get_dummy_datasets(st.session_state.real_data[key_fields], syndat[key_fields])
    dummy_ind_vals = dummy_real_cat[st.session_state.indiv_index]
    idx = air_nn(dummy_ind_vals, dummy_syn_cat, k=1)
    dummy_real_indv, dummy_syn_indv = get_dummy_datasets(st.session_state.real_data['Like Liquorice'], syndat['Like Liquorice'])
    real_label = np.array(dummy_real_indv[[st.session_state.indiv_index]])
    pred_label = np.array(dummy_syn_indv[idx])
    match = (real_label == 1) & (pred_label == 1)
    row_counts = st.session_state.real_data[key_fields].value_counts(normalize=True)
    prob_df = st.session_state.real_data[key_fields]
    prob_df['Probability'] = prob_df.apply(lambda row: row_counts[tuple(row)], axis=1)
    safe_probs = np.clip(prob_df['Probability'], 1e-10, None)
    numerator = safe_probs * np.log(safe_probs)
    denominator = numerator.sum()
    prob_df['Weight'] = numerator / denominator
    precision = round(match.sum() / (match.sum()+(len(pred_label)-match.sum())), 2)
    recall = 1
    f_one = round((2*precision*recall) / (precision+recall), 2)
    
    return f_one*(prob_df['Weight'].iloc[st.session_state.indiv_index]) > 0
def gcap_no_prot():
    key_fields = ['First Name', 'Last Name', 'Height', 'Nationality', 'Favorite Icecream', 'Times Been to Italy', 'First Time London', 'Steps per Day']
    ind_vals = st.session_state.real_data[key_fields].iloc[[st.session_state.indiv_index]]
    syndat=st.session_state.syn_data_bin
    neighbour_index, neighbour, distance = nearest_neighbor_hamming(ind_vals, syndat[key_fields])
    return syndat['Like Liquorice'].values[neighbour_index] == st.session_state.real_data['Like Liquorice'].values[st.session_state.indiv_index]
def zcap_no_prot():
    key_fields = ['First Name', 'Last Name', 'Height', 'Nationality', 'Favorite Icecream', 'Times Been to Italy', 'First Time London', 'Steps per Day']
    ind_vals = st.session_state.real_data[key_fields].iloc[[st.session_state.indiv_index]]
    
    ind_like_liquorice = st.session_state.real_data.loc[st.session_state.indiv_index, 'Like Liquorice']
    matching_rows = st.session_state.syn_data_bin[st.session_state.syn_data_bin.apply(lambda row: (ind_vals == row[key_fields]).all(axis=1).any(), axis=1)]
    has_same_like_liquorice = (matching_rows['Like Liquorice'] == ind_like_liquorice).any()
    return has_same_like_liquorice
def mdcr_no_prot():
        return st.session_state.dists_real[st.session_state.indiv_index, 1] > st.session_state.dists_syn[st.session_state.indiv_index, 0]
def hitr_no_prot():
    ind_vals = st.session_state.real_data.iloc[[st.session_state.indiv_index]]
    cat_attr = ['First Name', 'Last Name', 'Height', 'Nationality', 'Favorite Icecream', 'Times Been to Italy', 'First Time London', 'Steps per Day']
    if any((ind_vals[cat_attr] == st.session_state.syn_data_bin[cat_attr].iloc[i]).all(axis=1).any() for i in range(len(st.session_state.syn_data_bin[cat_attr]))):
        return True
    else: return False
def nnaa_no_prot():
    target_value = st.session_state.indiv_index
    indices = np.argwhere(st.session_state.idx_syn_real_gower[:, 0] == target_value)
    if indices.size > 0:  # Check if the array is not empty
        indice = indices[0, 0]  # Extract the first occurrence
        syn_nb = st.session_state.idx_syn_syn_gower[indice, 0]
    else:
        indice = None
        syn_nb = None
    if st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 0] > st.session_state.dists_real_real_gower[st.session_state.indiv_index, 1]:
        score1 = 1
    else: score1 = 0
    if indice != None:
        if st.session_state.dists_real_syn_gower[indice, 0] > st.session_state.dists_syn_syn_gower[syn_nb, 1]:
            score2 = 1
        else: score2 = 0
        
    else: 
        score2 = 0
    return score1 == 1 or score2 == 1
def crp_no_prot():
    ind_vals = st.session_state.real_data.iloc[[st.session_state.indiv_index]]
    matching_rows = st.session_state.syn_data_bin[st.session_state.syn_data_bin.apply(lambda row: (ind_vals == row).all(axis=1).any(), axis=1)]
    if len(matching_rows) > 0:
        return True
    return False
def nsnd_no_prot():
    return ((st.session_state.dists_syn[st.session_state.indiv_index, 0] - min(st.session_state.dists_syn[:, 0]))) / (max(st.session_state.dists_syn[:, 0]) - min(st.session_state.dists_syn[:, 0]) + 1e-8) > 0.5
def cvp_no_prot():
    return st.session_state.dists_syn[st.session_state.indiv_index, 0] < 0.2
def dvp_no_prot():
    return st.session_state.dists_syn[st.session_state.indiv_index, 0] < 0.8
def auth_no_prot():
    return (st.session_state.dists_syn[st.session_state.indiv_index, 0] - st.session_state.dists_real[st.session_state.indiv_index, 1]) < 0
def idS_no_prot():
    X_gt_ = st.session_state.real_labels.to_numpy().reshape(len(st.session_state.real_data), -1)
    X_syn_ = st.session_state.syn_labels.to_numpy().reshape(len(st.session_state.syn_data_bin), -1)
    
    def compute_entropy(labels: np.ndarray) -> np.ndarray:
        from scipy.stats import entropy
        value, counts = np.unique(np.round(labels), return_counts=True)
        return entropy(counts)
    no, x_dim = X_gt_.shape
    W = np.zeros(
        [
            x_dim,
        ]
    )
    for i in range(x_dim):
        W[i] = compute_entropy(X_gt_[:, i])
    X_hat = X_gt_
    X_syn_hat = X_syn_
    eps = st.session_state.epsilon
    W = np.ones_like(W)
    for i in range(x_dim):
        X_hat[:, i] = X_gt_[:, i] * 1.0 / (W[i] + eps)
        X_syn_hat[:, i] = X_syn_[:, i] * 1.0 / (W[i] + eps)
    nbrs = NearestNeighbors(n_neighbors=2).fit(X_hat)
    distance_r, indx_r = nbrs.kneighbors(X_hat)
    # hat{r_i} computation
    nbrs_hat = NearestNeighbors(n_neighbors=1).fit(X_syn_hat)
    distance_hat, indx_s = nbrs_hat.kneighbors(X_hat)
    # See which one is bigger
    R_Diff = distance_hat[st.session_state.indiv_index, 0] - distance_r[st.session_state.indiv_index, 1]
    return R_Diff < 0
def dcr_no_prot():
    return st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 0] < (np.mean(st.session_state.dists_real_syn_gower[:, 0])/2)
def hidr_no_prot():
    return st.session_state.indiv_index == st.session_state.idx_real_syn_gower[st.session_state.indiv_index, 0]
    
def metric_applicability(metric_results):
    
    st.session_state.has_continuous = not st.session_state.real_data.select_dtypes(include=['float64']).empty
    st.session_state.cont_cols = st.session_state.real_data.select_dtypes(include=['float64']).columns.tolist() + ['Steps per Day']
    
    with open("sensitive_attributes.txt", "r") as sensitive_file:
        sensitive_attributes = sensitive_file.read().splitlines()
    st.session_state.is_sens_cont = any(st.session_state.real_data[attr].dtype == 'float64' for attr in sensitive_attributes if attr in st.session_state.real_data.columns)
                      
    st.session_state.is_large = st.session_state.real_data.shape[1] > 3
    
    zcap_prob=gcap_prob=mdcr_prob=hitr_prob=mir_prob=nnaa_prob=crp_prob=nsnd_prob=cvp_prob=dvp_prob=auth_prob=dmlp_prob=idS_prob=air_prob=dcr_prob=nndr_prob=hidd_prob = ""
    zcap_sol=gcap_sol=mdcr_sol=hitr_sol=mir_sol=nnaa_sol=crp_sol=nsnd_sol=cvp_sol=dvp_sol=auth_sol=dmlp_sol=idS_sol=air_sol=dcr_sol=nndr_sol=hidd_sol = ""
    
    applicability_column = pd.DataFrame(metric_results['Metric'])
    applicability_column['App.'] = '✅'
    
    problem_column = pd.DataFrame(metric_results['Metric'])
    problem_column['Problem'] = '✅'
    
    solution_column = pd.DataFrame(metric_results['Metric'])
    solution_column['Possible Solution'] = '✅'
    
    applicability_u_column = pd.DataFrame(metric_results['Metric'])
    applicability_u_column['User App.'] = '✅'
    
    user_protected_column = pd.DataFrame(metric_results['Metric'])
    user_protected_column['User Protected?'] = '✅'
    
    shareable_column = pd.DataFrame(metric_results['Metric'])
    shareable_column['Shareable?'] = '✅'
    
    #The shareable column
    st.session_state.air_share = '✅'
    st.session_state.gcap_share = '✅'
    st.session_state.zcap_share = '✅'
    st.session_state.mdcr_share = '✅'
    st.session_state.hitr_share = '✅'
    st.session_state.mir_share = '✅'
    st.session_state.nnaa_share = '✅'
    st.session_state.crp_share = '✅'
    st.session_state.nsnd_share = '✅'
    st.session_state.cvp_share = '✅'
    st.session_state.dvp_share = '✅'
    st.session_state.auth_share = '✅'
    st.session_state.dmlp_share = '✅'
    st.session_state.ids_share = '✅'
    st.session_state.dcr_share = '✅'
    st.session_state.nndr_share = '✅'
    st.session_state.hidr_share = '✅'
    
    if metric_results.loc[metric_results['Metric'] == 'Attribute Inference Risk', 'Result'].iloc[0] > 0.5:
        shareable_column.loc[shareable_column['Metric']=='Attribute Inference Risk', 'Shareable?'] = '⛔️'
        st.session_state.air_share = '⛔️'
    if metric_results.loc[metric_results['Metric']=='GeneralizedCAP', 'Result'].iloc[0] > 0.5:
        shareable_column.loc[shareable_column['Metric']=='GeneralizedCAP', 'Shareable?'] = '⛔️'
        st.session_state.gcap_share = '⛔️'
    if metric_results.loc[metric_results['Metric']=='ZeroCAP', 'Result'].iloc[0] > 0.5:
        shareable_column.loc[shareable_column['Metric']=='ZeroCAP', 'Shareable?'] = '⛔️'
        st.session_state.zcap_share = '⛔️'
    if metric_results.loc[metric_results['Metric']=='Median Distance to Closest Record', 'Result'].iloc[0] >= 0.5:
        shareable_column.loc[shareable_column['Metric']=='Median Distance to Closest Record', 'Shareable?'] = '⛔️'
        st.session_state.mdcr_share = '⛔️'
    if metric_results.loc[metric_results['Metric']=='Median Distance to Closest Record', 'Result'].iloc[0] < 0.5:
        shareable_column.loc[shareable_column['Metric']=='Median Distance to Closest Record', 'Shareable?'] = '⚠️'
        st.session_state.mdcr_share = '⚠️'
    if metric_results.loc[metric_results['Metric']=='Hitting Rate', 'Result'].iloc[0] >= 0.00001:
        shareable_column.loc[shareable_column['Metric']=='Hitting Rate', 'Shareable?'] = '⛔️'
        st.session_state.hitr_share = '⛔️'
    if metric_results.loc[metric_results['Metric']=='Membership Inference Risk', 'Result'].iloc[0] > 0.5:
        shareable_column.loc[shareable_column['Metric']=='Membership Inference Risk', 'Shareable?'] = '⛔️'
        st.session_state.mir_share = '⛔️'
    if metric_results.loc[metric_results['Metric']=='Nearest Neighbour Adversarial Accuracy', 'Result'].iloc[0] > 0.1:
        shareable_column.loc[shareable_column['Metric']=='Nearest Neighbour Adversarial Accuracy', 'Shareable?'] = '⚠️'
        st.session_state.nnaa_share = '⚠️'
    if metric_results.loc[metric_results['Metric']=='Common Row Proportion', 'Result'].iloc[0] >= 0.00001:
        shareable_column.loc[shareable_column['Metric']=='Common Row Proportion', 'Shareable?'] = '⛔️'
        st.session_state.crp_share = '⛔️'
    if metric_results.loc[metric_results['Metric']=='Nearest Synthetic Neighbour Distance', 'Result'].iloc[0] > 0.5:
        shareable_column.loc[shareable_column['Metric']=='Nearest Synthetic Neighbour Distance', 'Shareable?'] = '⚠️'
        st.session_state.nsnd_share = '⚠️'
    if metric_results.loc[metric_results['Metric']=='Close Value Probability', 'Result'].iloc[0] >= 0.00001:
        shareable_column.loc[shareable_column['Metric']=='Close Value Probability', 'Shareable?'] = '⛔️'
        st.session_state.cvp_share = '⛔️'
    if metric_results.loc[metric_results['Metric']=='Distant Value Probability', 'Result'].iloc[0] >= 0.00001:
        shareable_column.loc[shareable_column['Metric']=='Distant Value Probability', 'Shareable?'] = '⛔️'
        st.session_state.dvp_share = '⛔️'
    if metric_results.loc[metric_results['Metric']=='Authenticity', 'Result'].iloc[0] >= 0.00001:
        shareable_column.loc[shareable_column['Metric']=='Authenticity', 'Shareable?'] = '⚠️'
        st.session_state.auth_share = '⚠️'
    if metric_results.loc[metric_results['Metric']=='DetectionMLP', 'Result'].iloc[0] > 0.1:
        shareable_column.loc[shareable_column['Metric']=='DetectionMLP', 'Shareable?'] = '⚠️'
        st.session_state.dmlp_share = '⚠️'
    if metric_results.loc[metric_results['Metric']=='Identifiability Score', 'Result'].iloc[0] >= 0.00001:
        shareable_column.loc[shareable_column['Metric']=='Identifiability Score', 'Shareable?'] = '⚠️'
        st.session_state.ids_share = '⚠️'
    if metric_results.loc[metric_results['Metric']=='Distance to Closest Record', 'Result'].iloc[0] > 0.5:
        shareable_column.loc[shareable_column['Metric']=='Distance to Closest Record', 'Shareable?'] = '⚠️'
        st.session_state.dcr_share = '⚠️'
    if metric_results.loc[metric_results['Metric']=='Nearest Neighbour Distance Ratio', 'Result'].iloc[0] > 0.1:
        shareable_column.loc[shareable_column['Metric']=='Nearest Neighbour Distance Ratio', 'Shareable?'] = '⚠️'
        st.session_state.nndr_share = '⚠️'
    if metric_results.loc[metric_results['Metric']=='Hidden Rate', 'Result'].iloc[0] >= 0.00001:
        shareable_column.loc[shareable_column['Metric']=='Hidden Rate', 'Shareable?'] = '⛔️'
        st.session_state.hidr_share = '⛔️'
    
    #User at risk column
    st.session_state.air_prot = '✅'
    st.session_state.gcap_prot = '✅'
    st.session_state.zcap_prot = '✅'
    st.session_state.mdcr_prot = '✅'
    st.session_state.hitr_prot = '✅'
    st.session_state.mir_prot = '✅'
    st.session_state.nnaa_prot = '✅'
    st.session_state.crp_prot = '✅'
    st.session_state.nsnd_prot = '✅'
    st.session_state.cvp_prot = '✅'
    st.session_state.dvp_prot = '✅'
    st.session_state.auth_prot = '✅'
    st.session_state.dmlp_prot = '✅'
    st.session_state.ids_prot = '✅'
    st.session_state.dcr_prot = '✅'
    st.session_state.nndr_prot = '✅'
    st.session_state.hidr_prot = '✅'
    
    if air_no_prot():
        user_protected_column.loc[user_protected_column['Metric']=='Attribute Inference Risk', 'User Protected?'] = '⛔️'
        st.session_state.air_prot = '⛔️'
    if gcap_no_prot():
        user_protected_column.loc[user_protected_column['Metric']=='GeneralizedCAP', 'User Protected?'] = '⛔️'
        st.session_state.gcap_prot = '⛔️'
    if zcap_no_prot():
        user_protected_column.loc[user_protected_column['Metric']=='ZeroCAP', 'User Protected?'] = '⛔️'
        st.session_state.zcap_prot = '⛔️'
    if mdcr_no_prot():
        user_protected_column.loc[user_protected_column['Metric']=='Median Distance to Closest Record', 'User Protected?'] = '⛔️'
        st.session_state.mdcr_prot = '⛔️'
    if hitr_no_prot():
        user_protected_column.loc[user_protected_column['Metric']=='Hitting Rate', 'User Protected?'] = '⛔️'
        st.session_state.hitr_prot = '⛔️'
    if metric_results.loc[metric_results['Metric']=='Membership Inference Risk', 'Result'].iloc[0] <= 1:
        user_protected_column.loc[user_protected_column['Metric']=='Membership Inference Risk', 'User Protected?'] = '⚠️'
        st.session_state.mir_prot = '⚠️'
    if nnaa_no_prot():
        user_protected_column.loc[user_protected_column['Metric']=='Nearest Neighbour Adversarial Accuracy', 'User Protected?'] = '⛔️'
        st.session_state.nnaa_prot = '⛔️'
    if crp_no_prot():
        user_protected_column.loc[user_protected_column['Metric']=='Common Row Proportion', 'User Protected?'] = '⛔️'
        st.session_state.crp_prot = '⛔️'
    if nsnd_no_prot():
        user_protected_column.loc[user_protected_column['Metric']=='Nearest Synthetic Neighbour Distance', 'User Protected?'] = '⛔️'
        st.session_state.nsnd_prot = '⛔️'
    if cvp_no_prot():
        user_protected_column.loc[user_protected_column['Metric']=='Close Value Probability', 'User Protected?'] = '⛔️'
        st.session_state.cvp_prot = '⛔️'
    if dvp_no_prot():
        user_protected_column.loc[user_protected_column['Metric']=='Distant Value Probability', 'User Protected?'] = '⛔️'
        st.session_state.dvp_prot = '⛔️'
    if auth_no_prot():
        user_protected_column.loc[user_protected_column['Metric']=='Authenticity', 'User Protected?'] = '⛔️'
        st.session_state.auth_prot = '⛔️'
    if metric_results.loc[metric_results['Metric']=='DetectionMLP', 'Result'].iloc[0] <= 1:
        user_protected_column.loc[user_protected_column['Metric']=='DetectionMLP', 'User Protected?'] = '⚠️'
        st.session_state.dmlp_prot = '⚠️'
    if idS_no_prot():
        user_protected_column.loc[user_protected_column['Metric']=='Identifiability Score', 'User Protected?'] = '⛔️'
        st.session_state.ids_prot = '⛔️'
    if dcr_no_prot():
        user_protected_column.loc[user_protected_column['Metric']=='Distance to Closest Record', 'User Protected?'] = '⛔️'
        st.session_state.dcr_prot = '⛔️'
    if metric_results.loc[metric_results['Metric']=='Nearest Neighbour Distance Ratio', 'Result'].iloc[0] <= 1:
        user_protected_column.loc[user_protected_column['Metric']=='Nearest Neighbour Distance Ratio', 'User Protected?'] = '⚠️'
        st.session_state.nndr_prot = '⚠️'
    if hidr_no_prot():
        user_protected_column.loc[user_protected_column['Metric']=='Hidden Rate', 'User Protected?'] = '⛔️'
        st.session_state.hidr_prot = '⛔️'
    
    if st.session_state.has_continuous:
        zcap_prob += f'<br>- Continuous attributes can not be used ({st.session_state.cont_cols}).'
        zcap_sol += '<br>- Remove all continuous attributes.'
        applicability_column.loc[applicability_column['Metric']=='ZeroCAP', 'App.'] = '⛔️'
        applicability_u_column.loc[applicability_u_column['Metric']=='ZeroCAP', 'User App.'] = '⛔️'
        gcap_prob += f'<br>- Key fields contain continuous attributes ({st.session_state.cont_cols}), in the nearest neighbour algorithm, continuous attributes influence the distance measure differently than other attributes.'
        gcap_sol += '<br>- Remove all continuous attributes.'
        applicability_column.loc[applicability_column['Metric']=='GeneralizedCAP', 'App.'] = '⚠️'
        mdcr_prob += '<br>- Distances lose expresivity for continuous attributes.'
        mdcr_sol += f'<br>- Consider removing continuous attributes.'
        applicability_column.loc[applicability_column['Metric']=='Median Distance to Closest Record', 'App.'] = '⚠️'
        hitr_prob += '<br>- The randomness induced by the synthesizer makes finding a match highly unlikely.'
        hitr_sol += f'<br>-Consider removing continuous attributes ({st.session_state.cont_cols})'
        applicability_column.loc[applicability_column['Metric']=='Hitting Rate', 'App.'] = '⛔️'
        applicability_u_column.loc[applicability_u_column['Metric']=='Hitting Rate', 'User App.'] = '⛔️'
        crp_prob += f'<br>- Continuous attributes ({st.session_state.cont_cols}) can not be used, as the noise induced by the synthesizer renders this theoretically impossible.'
        crp_sol += '<br>- Remove all continuous attributes.'
        applicability_column.loc[applicability_column['Metric']=='Common Row Proportion', 'App.'] = '⛔️'
        applicability_u_column.loc[applicability_u_column['Metric']=='Common Row Proportion', 'User App.'] = '⛔️'
        nsnd_prob += f'<br>- Distances lose expresivity and vary much for continuous attributes ({st.session_state.cont_cols}).'
        nsnd_sol += f'<br>- Consider removing continuous attributes ({st.session_state.cont_cols})'
        applicability_column.loc[applicability_column['Metric']=='Nearest Synthetic Neighbour Distance', 'App.'] = '⚠️'
        cvp_prob += f'<br>- Distances lose expresivity and vary much for continuous attributes ({st.session_state.cont_cols}).'
        cvp_sol += '<br>- Remove all continuous attributes.'
        applicability_column.loc[applicability_column['Metric']=='Close Value Probability', 'App.'] = '⚠️'
        applicability_u_column.loc[applicability_u_column['Metric']=='Close Value Probability', 'User App.'] = '⚠️'
        dvp_prob += f'<br>- Distances lose expresivity and vary much for continuous attributes ({st.session_state.cont_cols}).'
        dvp_sol += '<br>- Remove all continuous attributes.'
        applicability_column.loc[applicability_column['Metric']=='Distant Value Probability', 'App.'] = '⚠️'
        applicability_u_column.loc[applicability_u_column['Metric']=='Distant Value Probability', 'User App.'] = '⚠️'
        dcr_prob += f'<br>- Distances lose expresivity and vary much for continuous attributes ({st.session_state.cont_cols}).'
        dcr_sol += '<br>- Consider removing continuous attributes.'
        applicability_column.loc[applicability_column['Metric']=='Distance to Closest Record', 'App.'] = '⚠️'
        air_prob += f'<br>- Key fields contain continuous attributes ({st.session_state.cont_cols}), in the nearest neighbour algorithm, continuous attributes influence the distance measure differently than other attributes.'
        air_sol += '<br>- Remove all continuous attributes.'
        applicability_column.loc[applicability_column['Metric']=='Attribute Inference Risk', 'App.'] = '⚠️'
        nndr_prob += f'<br>- Distances lose expresivity and vary much for continuous attributes ({st.session_state.cont_cols}).'
        nndr_sol += '<br>- Consider removing continuous attributes.'
        applicability_column.loc[applicability_column['Metric']=='Nearest Neighbour Distance Ratio', 'App.'] = '⚠️'
    
    if st.session_state.is_large:
        mdcr_prob += '<br>- Distance can be misleading in high-dimensional spaces.'
        mdcr_sol += f'<br>- Consider decreasing the dataset size.'
        nnaa_prob += '<br>- Distance can be misleading in high-dimensional spaces.'
        nnaa_sol += '<br>- Consider decreasing the size of the dataset.'
        nsnd_prob += '<br>- Distance can be misleading in high-dimensional spaces.'
        nsnd_sol += '<br>- Consider decreasing the size of the dataset.'
        cvp_prob += '<br>- Distance can be misleading in high-dimensional spaces.'
        cvp_sol += '<br>- Consider decreasing the dataset size.'
        dvp_prob += '<br>- Distance can be misleading in high-dimensional spaces.'
        dvp_sol += '<br>- Consider decreasing the dataset size.'
        auth_prob += '<br>- Distance can be misleading in high-dimensional spaces.'
        auth_sol += '<br>- Consider decreasing the dataset size.'
        idS_prob += '<br>- Distance can be misleading in high-dimensional spaces.'
        idS_sol += '<br>- Consider decreasing the dataset size.'
        air_prob += '<br>- Distance can be misleading in high-dimensional spaces.'
        air_sol += '<br>- Consider decreasing the dataset size.'
        dcr_prob += '<br>- Distance can be misleading in high-dimensional spaces.'
        dcr_sol += '<br>- Consider decreasing the dataset size.'
        nndr_prob += '<br>- Distance can be misleading in high-dimensional spaces.'
        nndr_sol += '<br>- Consider decreasing the dataset size.'
        
    if st.session_state.is_sens_cont:
        gcap_prob += '<br>- Your sensitive attributes contain continuous variables, the metric will not work.'
        gcap_sol += '<br>- Remove continuous attributes as sensitive values'
        applicability_column.loc[applicability_column['Metric']=='GeneralizedCAP', 'App.'] = '⛔️'
        applicability_u_column.loc[applicability_u_column['Metric']=='GeneralizedCAP', 'User App.'] = '⛔️'
        air_prob += '<br>- Thresholds are easy to cheat by the synthesizer. Just make sure that sensitive data is at least the threshold away from the original sensitive field.'
        applicability_column.loc[applicability_column['Metric']=='Attribute Inference Risk', 'App.'] = '⚠️'
        applicability_u_column.loc[applicability_u_column['Metric']=='Attribute Inference Risk', 'User App.'] = '⚠️'
    st.session_state.has_prob_syn_neigh = has_problematic_synthetic_neighbors()
    if st.session_state.has_prob_syn_neigh:
        mdcr_prob += '<br>- Non-private data is still be produced.'
        mdcr_sol += '<br>- Invetigate the distances between individual data points.'
        applicability_column.loc[applicability_column['Metric']=='Median Distance to Closest Record', 'App.'] = '⛔️'
        applicability_u_column.loc[applicability_u_column['Metric']=='Median Distance to Closest Record', 'User App.'] = '⚠️'
        mir_prob += '<br>- Classification model is being cheated, and a risk persists.'
        mir_sol += '<br>- Investigate each real datapoint and its 3 nearest neighbours in real and synthetic.'
        applicability_column.loc[applicability_column['Metric']=='Membership Inference Risk', 'App.'] = '⛔️'
        applicability_u_column.loc[applicability_u_column['Metric']=='Membership Inference Risk', 'User App.'] = '⚠️'
        nnaa_prob += '<br>- Non-private data is still be produced.'
        nnaa_sol += '<br>- Investigate each real datapoint and its 3 nearest neighbours in real and synthetic.'
        applicability_column.loc[applicability_column['Metric']=='Nearest Neighbour Adversarial Accuracy', 'App.'] = '⛔️'
        applicability_u_column.loc[applicability_u_column['Metric']=='Nearest Neighbour Adversarial Accuracy', 'User App.'] = '⚠️'
        nsnd_prob += '<br>- Non-private data is still be produced.'
        nsnd_sol += '<br>- Investigate each real datapoint and its 3 nearest neighbours in real and synthetic.'
        applicability_column.loc[applicability_column['Metric']=='Nearest Synthetic Neighbour Distance', 'App.'] = '⛔️'
        applicability_u_column.loc[applicability_u_column['Metric']=='Nearest Synthetic Neighbour Distance', 'User App.'] = '⚠️'
        auth_prob += '<br>- Non-private data is still be produced.'
        auth_sol += '<br>- Investigate each real datapoint and its 3 nearest neighbours in real and synthetic.'
        applicability_column.loc[applicability_column['Metric']=='Authenticity', 'App.'] = '⛔️'
        dmlp_prob += '<br>- Non-private data is still be produced.'
        dmlp_sol += '<br>- Investigate each real datapoint and its 3 nearest neighbours in real and synthetic.'
        applicability_column.loc[applicability_column['Metric']=='DetectionMLP', 'App.'] = '⛔️'
        applicability_u_column.loc[applicability_u_column['Metric']=='DetectionMLP', 'User App.'] = '⚠️'
        idS_prob += '<br>- Non-private data is still be produced.'
        idS_sol += '<br>- Investigate each real datapoint and its 3 nearest neighbours in real and synthetic.'
        applicability_column.loc[applicability_column['Metric']=='Identifiability Score', 'App.'] = '⛔️'
        applicability_u_column.loc[applicability_u_column['Metric']=='Identifiability Score', 'User App.'] = '⚠️'
        dcr_prob += '<br>- Non-private data is still be produced.'
        dcr_sol += '<br>- Investigate each real datapoint and its 3 nearest neighbours in real and synthetic.'
        applicability_column.loc[applicability_column['Metric']=='Distance to Closest Record', 'App.'] = '⛔️'
        applicability_u_column.loc[applicability_u_column['Metric']=='Distance to Closest Record', 'User App.'] = '⚠️'
        nndr_prob += '<br>- Non-private data is still be produced.'
        nndr_sol += '<br>- Investigate each real datapoint and its 3 nearest neighbours in real and synthetic.'
        applicability_column.loc[applicability_column['Metric']=='Nearest Neighbour Distance Ratio', 'App.'] = '⛔️'
        applicability_u_column.loc[applicability_u_column['Metric']=='Nearest Neighbour Distance Ratio', 'User App.'] = '⚠️'
    
    mir_prob += '<br>- Assumes that the adversary has very much knowledge about the real data.'
    mir_sol += '<br>- Try multiple different subset as a training set to get a better evaluation.'
    applicability_column.loc[applicability_column['Metric']=='Membership Inference Risk', 'App.'] = '⚠️'
    applicability_u_column.loc[applicability_u_column['Metric']=='Membership Inference Risk', 'User App.'] = '⚠️'
    cvp_prob += '<br>- Finding a "correct" threshold is a very difficult task.'
    cvp_sol += '<br>- Establish a threshold matching distances in datasets.'
    applicability_column.loc[applicability_column['Metric']=='Close Value Probability', 'App.'] = '⛔️'
    applicability_u_column.loc[applicability_u_column['Metric']=='Close Value Probability', 'User App.'] = '⛔️'
    dvp_prob += '<br>- Finding a "correct" threshold is a very difficult task.'
    dvp_sol += '<br>- Establish a threshold matching distances in datasets.'
    applicability_column.loc[applicability_column['Metric']=='Distant Value Probability', 'App.'] = '⛔️'
    applicability_u_column.loc[applicability_u_column['Metric']=='Distant Value Probability', 'User App.'] = '⛔️'
    dmlp_prob += '<br>- Assumes that the adversary has very much knowledge about the real data.'
    dmlp_sol += '<br>- Try multiple different subset as a training set to get a better evaluation.'
    applicability_column.loc[applicability_column['Metric']=='DetectionMLP', 'App.'] = '⚠️'
    applicability_u_column.loc[applicability_u_column['Metric']=='DetectionMLP', 'User App.'] = '⚠️'
    hidd_prob += '<br>- Synthetic individuals need to be generated from the real individual with same index.'
    hidd_sol += '<br>- Use synthesizer that generates individuals that are based on the individual with the same index in the real data.'
    applicability_column.loc[applicability_column['Metric']=='Hidden Rate', 'App.'] = '⛔️'
    applicability_u_column.loc[applicability_u_column['Metric']=='Hidden Rate', 'User App.'] = '⛔️'
    
    problem_column.loc[problem_column['Metric']=='ZeroCAP', 'Problem'] = zcap_prob
    problem_column.loc[problem_column['Metric']=='GeneralizedCAP', 'Problem'] = gcap_prob
    problem_column.loc[problem_column['Metric']=='Median Distance to Closest Record', 'Problem'] = mdcr_prob
    problem_column.loc[problem_column['Metric']=='Hitting Rate', 'Problem'] = hitr_prob
    problem_column.loc[problem_column['Metric']=='Membership Inference Risk', 'Problem'] = mir_prob
    problem_column.loc[problem_column['Metric']=='Nearest Neighbour Adversarial Accuracy', 'Problem'] = nnaa_prob
    problem_column.loc[problem_column['Metric']=='Common Row Proportion', 'Problem'] = crp_prob
    problem_column.loc[problem_column['Metric']=='Nearest Synthetic Neighbour Distance', 'Problem'] = nsnd_prob
    problem_column.loc[problem_column['Metric']=='Close Value Probability', 'Problem'] = cvp_prob
    problem_column.loc[problem_column['Metric']=='Distant Value Probability', 'Problem'] = dvp_prob
    problem_column.loc[problem_column['Metric']=='Authenticity', 'Problem'] = auth_prob
    problem_column.loc[problem_column['Metric']=='DetectionMLP', 'Problem'] = dmlp_prob
    problem_column.loc[problem_column['Metric']=='Identifiability Score', 'Problem'] = idS_prob
    problem_column.loc[problem_column['Metric']=='Attribute Inference Risk', 'Problem'] = air_prob
    problem_column.loc[problem_column['Metric']=='Distance to Closest Record', 'Problem'] = dcr_prob
    problem_column.loc[problem_column['Metric']=='Nearest Neighbour Distance Ratio', 'Problem'] = nndr_prob
    problem_column.loc[problem_column['Metric']=='Hidden Rate', 'Problem'] = hidd_prob
    
    solution_column.loc[solution_column['Metric']=='ZeroCAP', 'Possible Solution'] = zcap_sol
    solution_column.loc[solution_column['Metric']=='GeneralizedCAP', 'Possible Solution'] = gcap_sol
    solution_column.loc[solution_column['Metric']=='Median Distance to Closest Record', 'Possible Solution'] = mdcr_sol
    solution_column.loc[solution_column['Metric']=='Hitting Rate', 'Possible Solution'] = hitr_sol
    solution_column.loc[solution_column['Metric']=='Membership Inference Risk', 'Possible Solution'] = mir_sol
    solution_column.loc[solution_column['Metric']=='Nearest Neighbour Adversarial Accuracy', 'Possible Solution'] = nnaa_sol
    solution_column.loc[solution_column['Metric']=='Common Row Proportion', 'Possible Solution'] = crp_sol
    solution_column.loc[solution_column['Metric']=='Nearest Synthetic Neighbour Distance', 'Possible Solution'] = nsnd_sol
    solution_column.loc[solution_column['Metric']=='Close Value Probability', 'Possible Solution'] = cvp_sol
    solution_column.loc[solution_column['Metric']=='Distant Value Probability', 'Possible Solution'] = dvp_sol
    solution_column.loc[solution_column['Metric']=='Authenticity', 'Possible Solution'] = auth_sol
    solution_column.loc[solution_column['Metric']=='DetectionMLP', 'Possible Solution'] = dmlp_sol
    solution_column.loc[solution_column['Metric']=='Identifiability Score', 'Possible Solution'] = idS_sol
    solution_column.loc[solution_column['Metric']=='Attribute Inference Risk', 'Possible Solution'] = air_sol
    solution_column.loc[solution_column['Metric']=='Distance to Closest Record', 'Possible Solution'] = dcr_sol
    solution_column.loc[solution_column['Metric']=='Nearest Neighbour Distance Ratio', 'Possible Solution'] = nndr_sol
    solution_column.loc[solution_column['Metric']=='Hidden Rate', 'Possible Solution'] = hidd_sol
    
    protected_df = metric_results.merge(user_protected_column, on='Metric')
    shareable_df = protected_df.merge(shareable_column, on='Metric')
    applicability_df = shareable_df.merge(applicability_column, on='Metric')
    applicability_df.rename(columns={"Result": "Risk"})
    #applicability_u_df = applicability_df.merge(applicability_u_column, on='Metric')
    problem_df = applicability_df.merge(problem_column, on='Metric')
    solution_df = problem_df.merge(solution_column, on='Metric')

    return solution_df

if st.session_state.stage == 0:#User input
    st.title("Privacy Advisor")
    with st.popover("Show storyline"):
        st.subheader("Hi 👋 This is the story elaborating your goal as a researcher trying to generate private synthetic data.")
        st.write("Imagine that you are a doctor that wants to synthesize a dataset that holds your own sensitive data.")
        st.write("Having synthesized your dataset, you want to estimate the risk of publishing said dataset using privacy metrics, but do the metric give the necessary privacy estimation of your data?")
        st.write("This app thereby demonstrates privacy estimation of differentially private synthetic data that includes your own data, and the risks that may be associated with relying on current available metrics to estimate both your and all others individuals' privacy.")
        
    st.write("First, lets generate a profile for you:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        choice_liq = st.selectbox("Do you like liquorice?", ["Yes", "No"])
        like_liquorice = 1 if choice_liq == "Yes" else 0  # Convert selection to 1 or 0
        fav_ice = st.text_input("What is your favorite Icecream?")
        choice = st.selectbox("Is this your first time in London?", ["Yes", "No"])
        first_time = 1 if choice == "Yes" else 0  # Convert selection to 1 or 0
    st.markdown("""
    <style>
    div[data-baseweb="slider"] {
        max-width: 600px; /* Set your desired max width */
    }
    </style>
    """, unsafe_allow_html=True)
    height = float(st.slider("How tall are you (in cm)?", 0, 240, 170))
    st.session_state.like_liquorice = like_liquorice
    query_point = pd.DataFrame({
            "Favorite Icecream": [fav_ice],
            "Like Liquorice": [like_liquorice],
            "First Time London": [first_time],
            "Height": [height]
            })
    st.session_state.real_data = pd.read_csv(f'sample_data_{like_liquorice}.csv', index_col=False)
    input_cols = ["Favorite Icecream","Like Liquorice", "First Time London", "Height"]
    all_user = pd.concat([st.session_state.real_data[input_cols], query_point], ignore_index=True)
    fl_encoder = LabelEncoder()
    r_fl = fl_encoder.fit_transform(all_user['Favorite Icecream'])
    all_labels_user = pd.DataFrame({'Height': all_user['Height'],'Favorite Icecream':r_fl, 'Like Liquorice': all_user['Like Liquorice'], 'First Time London': all_user['First Time London']})
    real_labels_user = all_labels_user[:len(st.session_state.real_data)]
    user_labels = all_labels_user.tail(1)
    nn_user = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn_user.fit(real_labels_user)
    distance_user, index_user = nn_user.kneighbors(user_labels)
    st.session_state.indiv_index = index_user[0][0]
    st.session_state.selected_epsilons = []
    st.session_state.scatter_data = {}
    
    st.button(label="Generate Synthetic Dataset", on_click=set_state, args=[1])

if st.session_state.stage == 1:#Summary statistics
    st.title("Synthetic Data Generation")
    st.subheader("Assume this is you:")
    st.dataframe(st.session_state.real_data.iloc[[st.session_state.indiv_index]], use_container_width=True, hide_index=True)
    st.write("*Based on your input, this person in the dataset is most like you.*")
    st.write("This data is in the real dataset, and using the epsilon you desired, a synthetic dataset has been generated.")
    
    if "selected_epsilons" not in st.session_state:
        st.session_state.selected_epsilons = []
    if "scatter_data" not in st.session_state:
        st.session_state.scatter_data = {}
    st.session_state.epsilon = st.selectbox("What ε-value do you wich to use to synthesize your dataset (lower = more private)?", (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2.5, 5))
    if st.session_state.epsilon not in st.session_state.selected_epsilons:
        st.session_state.selected_epsilons.append(st.session_state.epsilon)
    
    st.title("Real vs. Synthetic Data Comparison")
    if st.session_state.epsilon not in st.session_state.scatter_data:
        with st.spinner("Gathering data..."):
            get_data(st.session_state.like_liquorice, st.session_state.epsilon)
            st.session_state.scatter_data[st.session_state.epsilon] = {
                "real_coords": st.session_state.real_coords_tsne,
                "syn_coords": st.session_state.syn_coords_tsne
            }
    # Ensure only the last two epsilons are stored
    if len(st.session_state.selected_epsilons) > 2:
        oldest_epsilon = st.session_state.selected_epsilons.pop(0)
        st.session_state.scatter_data.pop(oldest_epsilon, None)  # Remove old data safely

    # Access stored data for the new epsilon
    real_coords = st.session_state.scatter_data[st.session_state.epsilon]["real_coords"]
    syn_coords_new = st.session_state.scatter_data[st.session_state.epsilon]["syn_coords"]

    # Get synthetic data for the previous epsilon (if available)
    if len(st.session_state.selected_epsilons) > 1:
        prev_eps = st.session_state.selected_epsilons[0]
        syn_coords_old = st.session_state.scatter_data.get(prev_eps, {}).get("syn_coords", [])
    else:
        syn_coords_old = []  # Empty if only one epsilon has been used
    
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.subheader("Real Dataset")
        st.write(f"#Individuals:", len(st.session_state.real_data))
        num_unique_trans = pd.DataFrame(st.session_state.real_data.nunique()).transpose()
        num_unique = pd.DataFrame(num_unique_trans, columns = st.session_state.real_data.columns)
        st.write(f"#Unique values:")
        st.dataframe(num_unique, use_container_width=True, hide_index=True)
        most_frequent_real = pd.DataFrame(st.session_state.real_data.apply(lambda col: col.value_counts().idxmax()))
        most_frequent_real_df = pd.DataFrame(most_frequent_real.transpose(), columns=st.session_state.real_data.columns).transpose()
        st.write(f"Mode of each column:")
        st.dataframe(most_frequent_real_df, use_container_width=True, column_config={"0":"Mode"})
        st.write(f"Summary Statistics:")
        st.dataframe(round(st.session_state.real_data.describe()[1:], 2), use_container_width=True)
    
    with col2:
        st.subheader(f"Synthetic Dataset using ε={st.session_state.epsilon}")
        st.write(f"#Individuals:",len(st.session_state.syn_data_bin))
        num_unique_trans_syn = pd.DataFrame(st.session_state.syn_data_bin.nunique()).transpose()
        num_unique_syn = pd.DataFrame(num_unique_trans_syn, columns = st.session_state.syn_data_bin.columns)
        st.write(f"#Unique values:")
        st.dataframe(num_unique_syn, use_container_width=True, hide_index=True)
        most_frequent_syn = pd.DataFrame(st.session_state.syn_data_bin.apply(lambda col: col.value_counts().idxmax()))
        most_frequent_syn_df = pd.DataFrame(most_frequent_syn.transpose(), columns=st.session_state.syn_data_bin.columns).transpose()
        st.write(f"Mode of each column:")
        st.dataframe(most_frequent_syn_df, use_container_width=True, column_config={"0":"Mode"})
        st.write(f"Summary Statistics:")
        st.dataframe(round(st.session_state.syn_data_bin.describe()[1:], 2), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Scatter Plot of the Real and Synthetic Datasets")
        if len(syn_coords_old) == 0:
            st.pyplot(scatter_plot_tsne(real_coords, syn_coords_new))
            st.write("*Scatter plot of real (blue), synthetic (red), and your (yellow) data, that has been mapped to 2 dimensions using t-SNE.*")
        else:
            st.pyplot(scatter_plot_tsne_old_new(real_coords, syn_coords_old, syn_coords_new))
            st.write(f"*Scatter plot of real (blue), previous synthetic using ε={st.session_state.selected_epsilons[0]} (green), new synthetic using ε={st.session_state.selected_epsilons[1]} (red), and your (yellow) data, that has been mapped to 2 dimensions using t-SNE.*")
    with st.popover("Continue storyline"):
        st.subheader("Hi again 👋 This is the story elaborating your goal as a researcher trying to generate private synthetic data.")
        st.write("You have now generate a 'hopefully' private synthetic dataset.")
        st.write("You want to publish your synthetic dataset, but what does this mean for the privacy of you and the other individuals in the dataset?")
        st.write("Using privacy metrics, we can measure how private the synthetic dataset is... or can we?")
    st.title(f"What is The Risk When Publishing The Data?")
    st.write("Click the button to estimate the privacy using a variety of privacy metrics ⬇️")
    st.button(label="Measure Privacy", on_click=set_state, args=[2])

if st.session_state.stage == 2:#Metric Reults
    st.title("Synthetic Data Generation")
    st.subheader("Assume this is you:")
    st.dataframe(st.session_state.real_data.iloc[[st.session_state.indiv_index]], use_container_width=True, hide_index=True)
    st.write("*Based on your input, this person in the dataset is most like you.*")
    st.write("This data is in the real dataset, and using the epsilon you desired, a synthetic dataset has been generated.")
    st.session_state.epsilon = st.selectbox("What ε-value do you wich to use to synthesize your dataset (lower = more private)?", (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2.5, 5))
    
    st.title("Real vs. Synthetic Data Comparison")
    with st.spinner("Generating synthetic data..."):
        get_data(st.session_state.like_liquorice, st.session_state.epsilon)
    
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.subheader("Real Dataset")
        st.write(f"#Individuals:", len(st.session_state.real_data))
        num_unique_trans = pd.DataFrame(st.session_state.real_data.nunique()).transpose()
        num_unique = pd.DataFrame(num_unique_trans, columns = st.session_state.real_data.columns)
        st.write(f"#Unique values:")
        st.dataframe(num_unique, use_container_width=True, hide_index=True)
        most_frequent_real = pd.DataFrame(st.session_state.real_data.apply(lambda col: col.value_counts().idxmax()))
        most_frequent_real_df = pd.DataFrame(most_frequent_real.transpose(), columns=st.session_state.real_data.columns).transpose()
        st.write(f"Mode of each variable:")
        st.dataframe(most_frequent_real_df, use_container_width=True, column_config={"0":"Mode"})
        st.write(f"Summary Statistics:")
        st.dataframe(round(st.session_state.real_data.describe()[1:], 2), use_container_width=True)
    
    with col2:
        st.subheader(f"Synthetic Dataset using ε={st.session_state.epsilon}")
        st.write(f"#Individuals:",len(st.session_state.syn_data_bin))
        num_unique_trans_syn = pd.DataFrame(st.session_state.syn_data_bin.nunique()).transpose()
        num_unique_syn = pd.DataFrame(num_unique_trans_syn, columns = st.session_state.syn_data_bin.columns)
        st.write(f"#Unique values:")
        st.dataframe(num_unique_syn, use_container_width=True, hide_index=True)
        most_frequent_syn = pd.DataFrame(st.session_state.syn_data_bin.apply(lambda col: col.value_counts().idxmax()))
        most_frequent_syn_df = pd.DataFrame(most_frequent_syn.transpose(), columns=st.session_state.syn_data_bin.columns).transpose()
        st.write(f"Mode of each column:")
        st.dataframe(most_frequent_syn_df, use_container_width=True, column_config={"0":"Mode"})
        st.write(f"Summary Statistics:")
        st.dataframe(round(st.session_state.syn_data_bin.describe()[1:], 2), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Scatter Plot of the Real and Synthetic Datasets")
        st.pyplot(scatter_plot_tsne(st.session_state.real_coords_tsne, st.session_state.syn_coords_tsne))
        st.write("*Scatter plot of real (blue), synthetic (red), and your (yellow) data, that has been mapped to 2 dimensions using t-SNE.*")
    with st.popover("Continue storyline"):
        st.subheader("Hi again 👋 This is the story elaborating your goal as a researcher trying to generate private synthetic data.")
        st.write("You have now generate a 'hopefully' private synthetic dataset.")
        st.write("You want to publish your synthetic dataset, but what does this mean for the privacy of you and the other individuals in the dataset?")
        st.write("Using privacy metrics, we can measure how private the synthetic dataset is... or can we?")
    
    with open("sensitive_attributes.txt", "r") as sensitive_file:
        sensitive_attributes = sensitive_file.read().splitlines()
    st.session_state.is_sens_cont = any(st.session_state.real_data[attr].dtype == 'float64' for attr in sensitive_attributes if attr in st.session_state.real_data.columns)
    st.session_state.has_continuous = not st.session_state.real_data.select_dtypes(include=['float64']).empty                  
    st.session_state.is_large = st.session_state.real_data.shape[1] > 3
    subtitle = ""
    if st.session_state.has_continuous or st.session_state.is_sens_cont:
        subtitle += "[Continues attributes]"
    if st.session_state.is_large:
        subtitle += "[Dataset size]"
    st.session_state.has_prob_syn_neigh = has_problematic_synthetic_neighbors()
    if st.session_state.has_prob_syn_neigh:
        subtitle += " & [Distance measurement]"
    subtitle += "are causing issues in the estimation of privacy."
    
    if st.session_state.has_continuous or st.session_state.is_sens_cont or st.session_state.is_large or st.session_state.has_prob_syn_neigh:
        title = "Privacy Estimation Is Inadequate For Some Metrics"
    else: title = "Privacy Estimation Succesfull"
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(st.session_state.real_labels)
    st.session_state.dists_real, st.session_state.idxs_real = nn.kneighbors(st.session_state.real_labels)
    st.session_state.dists_syn, st.session_state.idxs_syn = nn.kneighbors(st.session_state.syn_labels)
    st.session_state.dists_real_syn_gower, st.session_state.idx_real_syn_gower = gower_knn(st.session_state.coord_real, st.session_state.syn_coords, 2)
    st.session_state.dists_real_real_gower, st.session_state.idx_real_real_gower = gower_knn(st.session_state.coord_real, st.session_state.coord_real, 2)
    st.session_state.dists_syn_real_gower, st.session_state.idx_syn_real_gower = gower_knn(st.session_state.syn_coords, st.session_state.coord_real, 2)
    st.session_state.dists_syn_syn_gower, st.session_state.idx_syn_syn_gower = gower_knn(st.session_state.syn_coords, st.session_state.syn_coords, 2)
    st.subheader(title)
    st.write(f"**{subtitle}**")
    st.write('''Here, you can see how private your synthetic data is through the use of multiple metric results. 
             These metrics measure the privacy as a risk measure with a score in range [0,1], 
             where a high score means high privacy risk and vice versa.''')
    
    app_desc = {
        "✅: Valid": ["The metric deems you protected.", 
                     "The metric deems the dataset shareable.",
                      "No assumption is compromised."],
        "⚠️: Conflict": ["You may be at risk.", 
                           "There may be conflict if you were to share the data.",
                           "Potentially insufficient measurement of risk."],
        "⛔️: Unreliable": ["Your data is getting leaked.",
                           "Privacy will be leaked if shared.",
                            "Unreliable privacy estimation."]
    }
    app_index = ["User Protected?", "Shareable?", "Applicability"]
    app_df = pd.DataFrame(app_desc, index=app_index)
    with st.popover("Meaning of '✅', '⚠️', and '⛔️' in table"):
        st.write("To elaborate how the assumptions of the metrics influence their ability to measure privacy, a description of possible problems is given. Here, the column User App. shows the metrics ability to measure the privacy of your datapoint.")
        st.write("The applicability in this scenario is elaborated as follows:")
        st.dataframe(app_df)
    solution_df = metric_applicability(st.session_state.metric_results_bin)
    st.write("*If you desire, you can explore how the different metrics are computed, and how their applicability was determined by clicking 'Explore'.*")
    col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1, 1, 1])
    with col1:
        st.write("**Metric**")
    with col2:
        st.write("**Risk**")
    with col3:
        st.write("**User Protected?**")
    with col4:
        st.write("**Shareable?**")
    with col5:
        st.write("**Applicability**")
    for index, row in solution_df.iterrows():
        # Create a set of columns for each row
        col1 = st.columns([1], border=True)[0]
        with col1:
            cols = st.columns([3, 1, 1, 1, 1, 1])  # Adjust column widths (metric, result, button)

            # Inner loop: Iterate through the columns for each row
            for col_index, (col_name, col) in enumerate(zip(row.index, cols)):
                if col_index == 0:  # First column (Metric)
                    with col:
                        st.write(row[col_name])  # Display the metric name
                elif col_index == 1:  # Second column (Result)
                    with col:
                        st.write(f"**{row[col_name]}**")  # Display the metric value
                elif col_index == 2:  # Second column (Result)
                    with col:
                        st.write(row[col_name])  # Display the metric value
                elif col_index == 3:  # Second column (Result)
                    with col:
                        st.write(row[col_name])  # Display the metric value
                elif col_index == 4:  # Second column (Result)
                    with col:
                        st.write(row[col_name])  # Display the metric value
                elif col_index == 5:  # Third column (Button)
                    with col:
                        if st.button(f"Explore", key=f"btn_{index}", on_click=set_state, args=[index+10]):  # Create a button with unique key
                            # Update session state when the button is clicked
                            st.rerun()
    

if st.session_state.stage == 10: #AIR
    tit = "Attribute Inference Risk (AIR)"
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    prob_u_txt = ''
    if st.session_state.is_sens_cont:
        prob_overall += '<br>- Thresholds are easy to cheat by the synthesizer. Just make sure that sensitive data is at least the threshold away from the original sensitive field.'
        sol_overall += '<br>- Remove all continuous sensitive attributes.'
        status_overall = '⚠️'
    if st.session_state.is_large:
        prob_overall += '<br>- Distance can be misleading in high-dimensional spaces.'
        sol_overall += '<br>- Consider decreasing the dataset size.'
    if st.session_state.has_continuous:
        prob_overall += f'<br>- Key fields contain continuous attributes ({st.session_state.cont_cols}), in the nearest neighbour algorithm, continuous attributes influence the distance measure differently than other attributes.'
        sol_overall += '<br>- Remove all continuous attributes.'
        status_overall = '⚠️'
    
    st.title(tit)
    st.subheader(f"User Protected?: {st.session_state.air_prot}, Shareability: {st.session_state.air_share}, Applicability: {status_overall}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        
        if st.session_state.air_prot == '⛔️':
            st.subheader("Your data is at risk!⛔️")
        if st.session_state.air_prot == '⚠️':
            st.subheader("Your data might be at risk!⚠️")
        if st.session_state.air_prot == '✅':
            st.subheader("Your data is safe!✅")
        st.write("AIR measures the risk of inference attacks by assessing how easily an attacker, using public real data and synthetic data, can infer sensitive values. It quantifies this difficulty with the a weighted F1-score.")
        st.subheader("How is your score calculated?")
        st.write("Your sensitive field is whether or not you like liquorice. Lets try to infer it using your key fields.")
        st.write("The key fields are:")
        key_fields = ['First Name', 'Last Name', 'Height', 'Nationality', 'Favorite Icecream', 'Times Been to Italy', 'First Time London', 'Steps per Day']
        ind_vals = st.session_state.real_data[key_fields].iloc[[st.session_state.indiv_index]]
        st.dataframe(ind_vals, use_container_width=True, hide_index=True)
        syndat=st.session_state.syn_data_bin
        dummy_real_cat, dummy_syn_cat = get_dummy_datasets(st.session_state.real_data[key_fields], syndat[key_fields])
        dummy_real, dummy_syn = get_dummy_datasets(st.session_state.real_data, syndat)
        dummy_ind_vals = dummy_real_cat[st.session_state.indiv_index]
        idx = air_nn(dummy_ind_vals, dummy_syn_cat, k=1)
        idx2 = air_nn(dummy_ind_vals, dummy_syn_cat, k=2)[1]
        st.write("The key Fields of nearest synthetic neighbour(s) using a normalized Hamming distance is:")
        st.dataframe(syndat[key_fields].iloc[idx], use_container_width=True, hide_index=True)
        st.write("These are the sensitive fields for both individuals:")
        dummy_real_indv, dummy_syn_indv = get_dummy_datasets(st.session_state.real_data['Like Liquorice'], syndat['Like Liquorice'])
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            st.dataframe(pd.DataFrame({'Your Sensitive Field': st.session_state.real_data['Like Liquorice'].iloc[[st.session_state.indiv_index]]}), use_container_width=True, hide_index=True)
        with col1_2:
            st.dataframe(pd.DataFrame({"Neighbour's Sensitive Field": syndat['Like Liquorice'].iloc[idx]}), use_container_width=True, hide_index=True)
        col1_1_1, col1_2_1 = st.columns(2)
        with col1_1_1:
            st.write("(One-Hot encoded)")
            st.write(dummy_real_indv[[st.session_state.indiv_index]])
            
        with col1_2_1:
            st.write("(One-Hot encoded)")
            st.write(dummy_syn_indv[idx])
            
        real_label = np.array(dummy_real_indv[[st.session_state.indiv_index]])
        pred_label = np.array(dummy_syn_indv[idx])
        match = (real_label == 1) & (pred_label == 1)
        row_counts = st.session_state.real_data[key_fields].value_counts(normalize=True)
        prob_df = st.session_state.real_data[key_fields]
        prob_df['Probability'] = prob_df.apply(lambda row: row_counts[tuple(row)], axis=1)
        safe_probs = np.clip(prob_df['Probability'], 1e-10, None)
        numerator = safe_probs * np.log(safe_probs)
        denominator = numerator.sum()
        prob_df['Weight'] = numerator / denominator
        precision = round(match.sum() / (match.sum()+(len(pred_label)-match.sum())), 2)
        recall = 1
        f_one = round((2*precision*recall) / (precision+recall), 2)
        if np.any(match):
            st.write("⛔️You and your neighbour have matching sensitive fields!⛔️")
            st.write("The metric thereby accurately infered your sensitive data, and you increase the overall risk of AIR.")
        else:
            st.write("✅You and your neighbour do not have matching sensitive fields.✅")
            st.write("The metric thereby can not infer your sensitive data, and you do not increase the overall risk of AIR.")
        
        st.subheader("Risk estimation of the dataset")
        st.write("*To calculate this metric, a one-hot encoding for categorical attributes must be used.*")
        st.write("The attacker follows four steps to guess a sensitive value:")
        st.write("1. Select a row from the real dataset and note its key fields.")
        st.write("2. Find the (k=1) nearest synthetic neighbour(s) using a normalized Hamming distance on the key fields.")
        st.write("3. Evaluate the binary and continuous attributes seperately for infering the sensitive fields.")
        st.write("3.1. Binary attributes: Computes true positives, false positives, false negatives.")
        st.write("3.2. Continuous attributes: Checks if predictions are within ±10% of actual values.")
        st.write("4. Compute the weighted F1-Score")
        st.write("This attack is repeated for all rows in the real dataset, and the score is weighted performance in predicting the sensitive column. The score is an overall probability of guessing the sensitive column correctly.")
        
        st.subheader(f"Shareability problems: {st.session_state.air_share}")
        if st.session_state.air_share == '⛔️':
            st.write("As the sensitive attribute is binary, and the risk is > 0.5, the probability of guessing the sensitive attribute is better than random, and the general population is at risk.")
        if st.session_state.air_share == '✅':
            st.write("As the sensitive attribute is binary, and the risk is < 0.5, the probability of guessing the sensitive attribute is worse than random, and the general population is therefore not at risk.")
        st.subheader(f"Applicability problems: {status_overall}")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
        st.write("The influence of continuous attributes in key fields may be vissible from the key fields of your 2nd nearest neighbour:")
        st.dataframe(syndat[key_fields].iloc[[idx2]], use_container_width=True, hide_index=True)
        
    
    with col2:
        st.subheader("Real Dataset:")
        st.dataframe(st.session_state.real_data, use_container_width=True, hide_index=True)
        st.subheader("Synthetic Dataset:")
        st.dataframe(st.session_state.syn_data_bin, use_container_width=True, hide_index=True)
        st.subheader("Real Dataset (One-Hot Encoded):")
        st.write(dummy_real)
        st.subheader("Synthetic Dataset (One-Hot Encoded):")
        st.write(dummy_syn)
            
    st.button(label="Go Back", on_click=set_state, args=[2])
        
if st.session_state.stage == 11: #GCAP    
    tit="Generalized Correct Attribution Probability (GCAP)"
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    prob_u_txt = ''
    if st.session_state.is_sens_cont:
        prob_overall += '<br>- Your sensitive attributes contain continuous variables, the metric will not work.'
        sol_overall += '<br>- Remove continuous attributes as sensitive values'
        status_overall = '⛔️'
    if st.session_state.is_large:
        prob_overall += '<br>- Distance can be misleading in high-dimensional spaces.'
        sol_overall += '<br>- Consider decreasing the dataset size.'
    if st.session_state.has_continuous:
        prob_overall += f'<br>- Key fields contain continuous attributes ({st.session_state.cont_cols}), in the nearest neighbour algorithm, continuous attributes influence the distance measure differently than other attributes.'
        sol_overall += '<br>- Remove all continuous attributes.'
        status_overall = '⚠️'
    
    st.title(tit)
    st.subheader(f"User Protected?: {st.session_state.gcap_prot}, Shareability: {st.session_state.gcap_share}, Applicability: {status_overall}")
    col1, col2 = st.columns(2, border=True)
    
    with col1:
        if st.session_state.gcap_prot == '⛔️':
            st.subheader("Your data is at risk!⛔️")
        if st.session_state.gcap_prot == '⚠️':
            st.subheader("Your data might be at risk!⚠️")
        if st.session_state.gcap_prot == '✅':
            st.subheader("Your data is safe!✅")
        st.write("GCAP measures the risk of inference attacks by assessing how easily an attacker, using public real data and synthetic data, can infer sensitive values. It quantifies this difficulty with the Correct Attribution Probability (CAP) algorithm.")
        st.subheader("How is your score calculated?")
        st.write("Your sensitive field is whether or not you like liquorice. Lets try to infer it using your key fields.")
        st.write("Your key fields are:")
        key_fields = ['First Name', 'Last Name', 'Height', 'Nationality', 'Favorite Icecream', 'Times Been to Italy', 'First Time London', 'Steps per Day']
        ind_vals = st.session_state.real_data[key_fields].iloc[[st.session_state.indiv_index]]
        st.dataframe(ind_vals, use_container_width=True, hide_index=True)
        syndat=st.session_state.syn_data_bin
        neighbour_index, neighbour, distance = nearest_neighbor_hamming(ind_vals, syndat[key_fields])
        st.write("This row is your nearest synthetic neighbouring key fields:")
        st.dataframe(neighbour, use_container_width=True, hide_index=True)
        st.write("These are the sensitive fields for both individuals:")
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            st.dataframe(pd.DataFrame({'Your Sensitive Field': st.session_state.real_data['Like Liquorice'].iloc[[st.session_state.indiv_index]]}), use_container_width=True, hide_index=True)
        with col1_2:
            st.dataframe(pd.DataFrame({"Neighbour's Sensitive Field": syndat['Like Liquorice'].iloc[[neighbour_index]]}), use_container_width=True, hide_index=True)
        
        if syndat['Like Liquorice'].values[neighbour_index] == st.session_state.real_data['Like Liquorice'].values[st.session_state.indiv_index]:
            st.write("⛔️You and your neighbour have matching sensitive fields!⛔️")
            st.write("The metric thereby accurately infered your sensitive data, and you increase the overall risk.")
        else:
            st.write("✅You and your neighbour do not have matching sensitive fields.✅")
            st.write("The metric thereby can not infer your sensitive data, and you do not increase the overall risk.")
        
        st.subheader("Risk estimation of the dataset")
        st.write("The attacker follows four steps to guess a sensitive value:")
        st.write("1. Select a row from the real dataset and note its key fields.")
        st.write("2. Find all synthetic rows matching these key fields (the synthetic equivalence class).")
        st.write("*If there are no matching key fields, the nearest synthetic neighbours using Hamming distance on the key fields are chosen*")
        st.write("3. Use the sensitive values in to vote on the real row’s sensitive fields.")
        st.write("4. The final score is the proportion of correct votes, ranging from 0 to 1.")
        st.write("This attack is repeated for all rows in the real dataset, and the score is an overall probability of guessing the sensitive column correctly.")
        
        
        st.subheader(f"Shareability problems: {st.session_state.gcap_share}")
        if st.session_state.gcap_share == '⛔️':
            st.write("As the sensitive attribute is binary, and the risk is > 0.5, the probability of guessing the sensitive attribute is better than random, and the general population is at risk.")
        if st.session_state.gcap_share == '✅':
            st.write("As the sensitive attribute is binary, and the risk is <= 0.5, the probability of guessing the sensitive attribute is worse than random, and the general population is therefore not at risk.")
        st.subheader(f"Applicability problems: {status_overall}")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
        st.write("The influence of continuous attributes in key fields may be vissible from the key fields of your 2nd nearest neighbour:")
        neighbour_index1, neighbour1, distance1 = nearest_neighbor_hamming(ind_vals, syndat[key_fields].drop([syndat[key_fields].index[neighbour_index]]))
        st.dataframe(neighbour1, use_container_width=True, hide_index=True)
            
    with col2:
        st.write("Real Dataset:")
        st.dataframe(st.session_state.real_data, use_container_width=True, hide_index=True)
        st.write("Your Synthetic Dataset:")
        st.dataframe(st.session_state.syn_data_bin, use_container_width=True, hide_index=True)
    
    st.button(label="Go Back", on_click=set_state, args=[2])

if st.session_state.stage == 12: #ZCAP
    tit = 'Zero Correct Attribution Probability (ZCAP)'
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    if st.session_state.has_continuous:
        prob_overall += f'<br>- Continuous attributes can not be used ({st.session_state.cont_cols}).'
        sol_overall += '<br>- Remove all continuous attributes.'
        status_overall = '⛔️'
        
    st.title(tit)
    st.subheader(f"User Protected?: {st.session_state.zcap_prot}, Shareability: {st.session_state.zcap_share}, Applicability: {status_overall}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        if st.session_state.zcap_prot == '⛔️':
            st.subheader("Your data is at risk!⛔️")
        if st.session_state.zcap_prot == '⚠️':
            st.subheader("Your data might be at risk!⚠️")
        if st.session_state.zcap_prot == '✅':
            st.subheader("Your data is safe!✅")
        st.write("ZCAP measures the risk of inference attacks by assessing how easily an attacker, using public real data and synthetic data, can infer sensitive values. It quantifies this difficulty with the Correct Attribution Probability (CAP) algorithm.")
        
        st.subheader("How is your score calculated?")
        st.write("Your sensitive field is whether or not you like liquorice. Lets try to infer it using your key fields.")
        st.write("Your key fields are:")
        key_fields = ['First Name', 'Last Name', 'Height', 'Nationality', 'Favorite Icecream', 'Times Been to Italy', 'First Time London', 'Steps per Day']
        ind_vals = st.session_state.real_data[key_fields].iloc[[st.session_state.indiv_index]]
        st.dataframe(ind_vals, hide_index=True)
        syndat=st.session_state.syn_data_bin
        matching_rows = st.session_state.syn_data_bin[st.session_state.syn_data_bin.apply(lambda row: (ind_vals == row[key_fields]).all(axis=1).any(), axis=1)]
        if any((ind_vals == syndat[key_fields].iloc[i]).all(axis=1).any() for i in range(len(syndat[key_fields]))):
            st.write("These rows have matching key fields in the synthetic dataset:")
            st.dataframe(matching_rows, hide_index=True)
        if not matching_rows.empty and matching_rows['Like Liquorice'].eq(st.session_state.real_data.loc[st.session_state.indiv_index, 'Like Liquorice']).any():
            st.write("⛔️You and your neighbour have matching sensitive fields!⛔️")
            st.write("The metric thereby accurately infered your sensitive data, and you increase the overall risk.")
        if not matching_rows.empty and not matching_rows['Like Liquorice'].eq(st.session_state.real_data.loc[st.session_state.indiv_index, 'Like Liquorice']).any():
            st.write("✅You and your matches do not have matching sensitive fields.✅")
            st.write("The metric thereby can not infer your sensitive data, and you do not increase the overall risk.")
        else:
            st.write("✅No match was found!✅")
            st.write("The metric thereby can not infer your sensitive data, and you do not increase the overall risk.")
       
        st.subheader("Risk estimation of the dataset")
        st.write("The attacker follows four steps to guess a sensitive value:")
        st.write("1. Select a row from the real dataset and note its key fields.")
        st.write("2. Find all synthetic rows matching these key fields (the synthetic equivalence class).")
        st.write("*If there are no matching key fields, the row has a score of 0.*")
        st.write("3. Use the sensitive values in to vote on the real row’s sensitive fields.")
        st.write("4. The final score is the proportion of correct votes, ranging from 0 to 1.")
        st.write("This attack is repeated for all rows in the real dataset, and the score is an overall probability of guessing the sensitive column correctly.")
        
        st.subheader(f"Shareability problems: {st.session_state.zcap_share}")
        if st.session_state.zcap_share == '⛔️':
            st.write("As the sensitive attribute is binary, and the risk is > 0.5, the probability of guessing the sensitive attribute is better than random, and the general population is at risk.")
        if st.session_state.zcap_share == '✅':
            st.write("As the sensitive attribute is binary, and the risk is <= 0.5, the probability of guessing the sensitive attribute is worse than random, and the general population is therefore not at risk.")
        st.subheader(f"Applicability problems: {status_overall}")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
        
    with col2:
        st.write("Real Dataset:")
        st.dataframe(st.session_state.real_data, use_container_width=True, hide_index=True)
        st.write("Your Synthetic Dataset:")
        st.dataframe(st.session_state.syn_data_bin, use_container_width=True, hide_index=True)
        
    st.button(label="Go Back", on_click=set_state, args=[2])

if st.session_state.stage == 13: #MDCR
    tit="Median Distance to Closest Record (MDCR)"
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    if st.session_state.is_large:
        prob_overall += '<br>- Distance can be misleading in high-dimensional spaces.'
        sol_overall += '<br>- Consider decreasing the dataset size.'
    if st.session_state.has_continuous:
        prob_overall += '<br>- Distances lose expresivity for continuous attributes.'
        sol_overall += f'<br>- Consider removing continuous attributes.'
        status_overall = '⚠️'
    if st.session_state.has_prob_syn_neigh:
        prob_overall += '<br>- Non-private data is still be produced.'
        sol_overall += '<br>- Invetigate the distances between individual data points.'
        status_overall = '⛔️'

    st.title(tit)
    st.subheader(f"User Protected?: {st.session_state.mdcr_prot}, Shareability: {st.session_state.mdcr_share}, Applicability: {status_overall}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        if st.session_state.mdcr_prot == '⛔️':
            st.subheader("Your data is at risk!⛔️")
        if st.session_state.mdcr_prot == '⚠️':
            st.subheader("Your data might be at risk!⚠️")
        if st.session_state.mdcr_prot == '✅':
            st.subheader("Your data is safe!✅")
        st.write("MDCR measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated.")
        
        st.subheader("How is your score calculated?")
        st.write("Your nearest neighbours are:")
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            st.write("Real neighbour:")
            st.dataframe(st.session_state.real_data.iloc[[st.session_state.idxs_real[st.session_state.indiv_index, 1]]], use_container_width=True, hide_index=True)
            st.write(f"With distance: {round(st.session_state.dists_real[st.session_state.indiv_index, 1], 1)}")
            
        with col1_2:
            st.write("Synthetic neighbour:")
            st.dataframe(st.session_state.syn_data_bin.iloc[[st.session_state.idxs_syn[st.session_state.indiv_index, 0]]], use_container_width=True, hide_index=True)
            st.write(f"With distance: {round(st.session_state.dists_syn[st.session_state.indiv_index, 0], 2)}")
            
        st.write("For your record, the equation would therefore be:")
        st.latex(r'\frac{'f'{round(st.session_state.dists_real[st.session_state.indiv_index, 1], 1)}'r'}{'f'{round(st.session_state.dists_syn[st.session_state.indiv_index, 0], 2)}'r'} = 'f'{round(st.session_state.dists_real[st.session_state.indiv_index, 1] / st.session_state.dists_syn[st.session_state.indiv_index, 0], 2)}')
        if st.session_state.dists_real[st.session_state.indiv_index, 1] > st.session_state.dists_syn[st.session_state.indiv_index, 0]:
            st.write("⛔️You have a neighbour in the synthetic data that is closer to you than your real neighbour.⛔️")
            st.write("The metric thereby accurately re-identified you, and you increase the overall risk.")
        else:
            st.write("✅You have a neighbour in the real data that is closer to you than your synthetic neighbour.✅")
            st.write("The metric thereby can not re-identify you, and you do not increase the overall risk.")
        
        st.subheader("Risk estimation of the dataset")
        st.write("The attacker follows four steps to measure the re-identification risk:")
        st.write("1. For each real individual find the distance to the nearest neighbour in the real dataset.")
        st.write("2. For each real individual find the distance to the nearest neighbour in the synthetic dataset.")
        st.write("3. Calculate the median of distances between real individuals.")
        st.write("4. Calculate the median of distances between real and synthetic individuals.")
        
        st.subheader(f"Shareability problems: {st.session_state.mdcr_share}")
        if st.session_state.mdcr_share == '⛔️':
            st.write("You generally have a lower distance to synthetic neighbours than real neighbours.")
        if st.session_state.mdcr_share == '⚠️':
            st.write("You generally have a lower distance to real neighbours than synthetic neighbours. However, to correctly determine the shareability through this metric, you have to estimate the risk for each individual, as one may be at high risk while the overall risk is still low.")
        if st.session_state.mdcr_share == '✅':
            st.write("You have a lower distance to real neighbours than synthetic neighbours for all datapoints.")
        st.subheader(f"Applicability problems: {status_overall}")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
        st.write("The influence of continuous attributes and high dimensions may be vissible from looking at the two neighbour with the smallest and largest distance between them:")
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            st.write("Minimum distance neighbours in the real dataset:")
            dists = pd.DataFrame(st.session_state.dists_real)[1]
            min_index = np.argmin(dists)
            st.dataframe(st.session_state.real_data.iloc[[min_index]], use_container_width=True, hide_index=True)
            st.dataframe(st.session_state.real_data.iloc[[st.session_state.idxs_real[min_index, 1]]], use_container_width=True, hide_index=True)
            st.write(f"With distance {round(dists[min_index], 2)}")
            st.write("Maximum distance neighbours in the real dataset:")
            max_index = np.argmax(dists)
            st.dataframe(st.session_state.real_data.iloc[[max_index]], use_container_width=True, hide_index=True)
            st.dataframe(st.session_state.real_data.iloc[[st.session_state.idxs_real[max_index, 1]]], use_container_width=True, hide_index=True)
            st.write(f"With distance {round(dists[max_index], 2)}")
        with col1_2:
            st.write("Minimum distance neighbours in the synthetic dataset:")
            distsyn = pd.DataFrame(st.session_state.dists_syn)[1]
            min_indexsyn = np.argmin(distsyn)
            st.dataframe(st.session_state.syn_data_bin.iloc[[min_indexsyn]], use_container_width=True, hide_index=True)
            st.dataframe(st.session_state.syn_data_bin.iloc[[st.session_state.idxs_syn[min_indexsyn, 1]]], use_container_width=True, hide_index=True)
            st.write(f"With distance {round(distsyn[min_indexsyn], 2)}")
            st.write("Maximum distance neighbours in the synthetic dataset:")
            max_indexsyn = np.argmax(distsyn)
            st.dataframe(st.session_state.syn_data_bin.iloc[[max_indexsyn]], use_container_width=True, hide_index=True)
            st.dataframe(st.session_state.syn_data_bin.iloc[[st.session_state.idxs_syn[max_indexsyn, 1]]], use_container_width=True, hide_index=True)
            st.write(f"With distance {round(distsyn[max_indexsyn], 2)}")

    with col2:
        st.write("Scatter plot of real and synthetic data:")
        st.pyplot(scatter_plot_tsne(st.session_state.real_coords_tsne, st.session_state.syn_coords_tsne))
        
        st.session_state.tsne_df_real = pd.DataFrame(st.session_state.real_coords_tsne)
        st.session_state.tsne_df_syn = pd.DataFrame(st.session_state.syn_coords_tsne)
        st.write("Scatter plot of you and your nearest neighbour in the real and synthetic data:")
        st.pyplot(scatter_plot_tsne(pd.concat([st.session_state.tsne_df_real.iloc[[st.session_state.idxs_real[len(st.session_state.tsne_df_real)-1, 1]]], st.session_state.tsne_df_real.iloc[[len(st.session_state.tsne_df_real)-1]]]),
                                                st.session_state.tsne_df_syn.iloc[[st.session_state.idxs_syn[len(st.session_state.tsne_df_syn)-1, 1]]]))
        
    st.button(label="Go Back", on_click=set_state, args=[2])
    
if st.session_state.stage == 14: #Hitting Rate
    tit="Hitting Rate (HitR)"
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    if st.session_state.has_continuous:
        prob_overall += '<br>- The randomness induced by the synthesizer makes finding a match highly unlikely.'
        sol_overall += f'<br>-Consider removing continuous attributes ({st.session_state.cont_cols})'
        status_overall = '⛔️'
    
    st.title(tit)
    st.subheader(f"User Protected?: {st.session_state.hitr_prot}, Shareability: {st.session_state.hitr_share}, Applicability: {status_overall}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        if st.session_state.hitr_prot == '⛔️':
            st.subheader("Your data is at risk!⛔️")
        if st.session_state.hitr_prot == '⚠️':
            st.subheader("Your data might be at risk!⚠️")
        if st.session_state.hitr_prot == '✅':
            st.subheader("Your data is safe!✅")
        st.write("HitR measures the risk of identifying whether an individual contributed their data to the real dataset while having access to the synthetic data.")
        
            
        st.subheader("How is your score calculated?")
        st.write("To identify whether or not you contributed your data, the metrics tries to find individuals with matching categorical and similar continuous attribute values.")
        st.write("Here are your values:")
        ind_vals = st.session_state.real_data.iloc[[st.session_state.indiv_index]]
        st.dataframe(ind_vals, hide_index=True)
        cat_attr = ['First Name', 'Last Name', 'Height', 'Nationality', 'Favorite Icecream', 'Times Been to Italy', 'First Time London', 'Steps per Day']
        if any((ind_vals[cat_attr] == st.session_state.syn_data_bin[cat_attr].iloc[i]).all(axis=1).any() for i in range(len(st.session_state.syn_data_bin[cat_attr]))):
            st.write("⛔️ You have matching synthetic individuals. ⛔️")
            st.write("These rows with matching categorical cloumns in the synthetic dataset:")
            matching_rows = st.session_state.syn_data_bin[st.session_state.syn_data_bin.apply(lambda row: (ind_vals == row[key_fields]).all(axis=1).any(), axis=1)]
            st.dataframe(matching_rows, hide_index=True)
        else:
            st.write("✅There are not any synthetic individuals with 'matching' values, and you are safe.✅")
            st.write("Therefore, the score for your data is 0.")
            
        st.subheader("Risk estimation of the dataset")
        st.write("The attacker follows four steps to guess a sensitive value:")
        st.write("1. Select a row from the real dataset and note its categorical and continuous attributes.")
        st.write("2. Find all synthetic rows with matching categorical values and where the continuous values are within a threshold (0.0333) of the real value.")
        st.write("*If there are no 'matching' individuals, the row has a score of 0.*")
        st.write("This attack is repeated for all rows in the real dataset, and the score is the probability of 'matching' rows in the real and synthetic datasets")
        
        st.subheader(f"Shareability problems: {st.session_state.hitr_share}")
        if st.session_state.hitr_share == '⛔️':
            st.write("There are matching rows in the synthetic data, and minimum one person is therefore at risk.")
        if st.session_state.hitr_share == '✅':
            st.write("There are no matching rows in the synthetic data, and no one is therefore at risk.")
        
        st.subheader(f"Applicability problems: {status_overall}")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
    
    with col2:
        st.write("Real Dataset:")
        st.dataframe(st.session_state.real_data, use_container_width=True, hide_index=True)
        st.write("Your Synthetic Dataset:")
        st.dataframe(st.session_state.syn_data_bin, use_container_width=True, hide_index=True)
    
    st.button(label="Go Back", on_click=set_state, args=[2])
    
if st.session_state.stage == 15: #MIR
    tit="Membership Inference Risk (MIR)"
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    status_u = '✅'
    prob_u_txt = ''
    prob_overall += '<br>- Assumes that the adversary has very much knowledge about the real data.'
    sol_overall += '<br>- Try multiple different subset as a training set to get a better evaluation.'
    status_overall = '⚠️'
    status_u = '⚠️'
    if st.session_state.has_prob_syn_neigh:
        prob_overall += '<br>- Classification model is being cheated, and a risk persists.'
        sol_overall += '<br>- Investigate each real datapoint and its 3 nearest neighbours in real and synthetic.'
        status_overall = '⛔️'
        status_u = '⛔️'
        
    if status_overall == '⚠️' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '✅' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '⚠️' and status_u == '⚠️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
    if status_overall == '⛔️' and status_u == '⛔️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
        
    st.title(tit)
    st.subheader(f"App: {status_overall} User App: {status_u}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.subheader("Overall")
        st.write("MIR estimates the risk of identifying whether an individual contributed their data to the real dataset while having access to the synthetic data and a subset of the real data.")
        st.write("The attacker follows four steps to identify individuals:")
        st.write("1. Make a new dataset that contains both the real and synthetic data as well as a labeling of whether or not the are real.")
        st.write("2. Split the dataset up into a train and test set.")
        st.write("3. Train a LightGBM classifier on the train set.")
        st.write("4. Measure the Recall of the classification task on the test set to get MIR.")
        st.subheader("Problems Overall:")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Problems for you:")
        st.markdown(prob_u_txt, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
        st.write("**The problems that may occur:**")
        st.write(" - Classification models are easy to cheat. If the synthetic data isn’t “too close” to the training data. However, when doing this, non-private data can still be produced:")
        st.image("images/distance_threshold_problem.png", caption="*https://desfontain.es/blog/bad-ugly-good-maybe.html*")
        
    with col2:
        st.write("Scatter plot of real and synthetic data:")
        st.pyplot(scatter_plot_tsne(st.session_state.real_coords_tsne, st.session_state.syn_coords_tsne))
        
    st.button(label="Go Back", on_click=set_state, args=[2])
    
if st.session_state.stage == 16: #NNAA
    tit="Nearest Neighbour Adversarial Accuracy (NNAA)"
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    status_u = '✅'
    prob_u_txt = ''
    if st.session_state.has_prob_syn_neigh:
        prob_overall += '<br>- Non-private data is still be produced.'
        sol_overall += '<br>- Investigate each real datapoint and its 3 nearest neighbours in real and synthetic.'
        status_overall = '⛔️'
        status_u = '⛔️'
    if st.session_state.is_large:
        prob_overall += '<br>- Distance can be misleading in high-dimensional spaces.'
        sol_overall += '<br>- Consider decreasing the size of the dataset.'
        
    if status_overall == '⚠️' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '✅' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '⚠️' and status_u == '⚠️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
    if status_overall == '⛔️' and status_u == '⛔️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
        
    st.title(tit)
    st.subheader(f"App: {status_overall} User App: {status_u}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.subheader("Overall")
        st.write('''NNAA estimates the risk of identifying whether an individual contributed their data to the real dataset while only having access to the synthetic data.''')
        st.write("NNAA is the calculated in four steps:")
        st.write("1. Use PCA, MCA and/or FAMD to map the datasets to 2 dimensions.")
        st.write("2. Use the mapped dataset to determine the nearest neighbour distances from real to synthetic and synthetic to real.")
        st.write("3. Calculate NNAA as the mean probability of:")
        st.write("3.1. from real data points, the probability that the distance to the nearest neighbour in the synthetic dataset is larger than the nearest neighbour in the real dataset.")
        st.write("3.2. from synthetic data points, the probability that the distance to the nearest neighbour in the real dataset is larger than the nearest neighbour in the synthetic dataset.")
        st.write("The score is then subtracted from 1 to match the direction of the risk score.")
        
        st.subheader("For your data")
        
        target_value = st.session_state.indiv_index
        indices = np.argwhere(st.session_state.idx_syn_real_gower[:, 0] == target_value)
        if indices.size > 0:  # Check if the array is not empty
            indice = indices[0, 0]  # Extract the first occurrence
            syn_nb = st.session_state.idx_syn_syn_gower[indice, 0]
        else:
            indice = None
            syn_nb = None
        col1_1, col1_2 = st.columns(2)   
        with col1_1:
            st.write("Your real neighbour:")
            st.dataframe(st.session_state.real_data.iloc[[st.session_state.idx_real_real_gower[st.session_state.indiv_index, 1]]], use_container_width=True, hide_index=True)
            st.write(f"With distance: {round(st.session_state.dists_real_real_gower[st.session_state.indiv_index, 1], 4)}")
            if indice != None:
                st.write("Syn individual with you as neighbour:")
                st.dataframe(st.session_state.syn_data_bin.iloc[[indice]], use_container_width=True, hide_index=True)
                st.write(f"With distance: {round(st.session_state.dists_syn_real_gower[indice, 0], 4)}")
        if indice == None: st.write("No syn individual has you as neighbour.")

        with col1_2:
            st.write("Your syn neighbour:")
            st.dataframe(st.session_state.syn_data_bin.iloc[[st.session_state.idx_real_syn_gower[st.session_state.indiv_index, 0]]], use_container_width=True, hide_index=True)
            st.write(f"With distance: {round(st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 0], 4)}")
            if syn_nb != None:
                st.write("Syn individual's syn neighbour:")
                st.dataframe(st.session_state.syn_data_bin.iloc[[syn_nb]], use_container_width=True, hide_index=True)
                st.write(f"With distance: {round(st.session_state.dists_syn_syn_gower[syn_nb, 1], 4)}")
            
        st.write("Thereby, your data's contribution to the score is:")
        if round(st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 0], 4) > round(st.session_state.dists_real_real_gower[st.session_state.indiv_index, 1], 4):
            score1 = 1
        else: score1 = 0
        if indice != None:
            if round(st.session_state.dists_real_syn_gower[indice, 0], 4) > round(st.session_state.dists_syn_syn_gower[syn_nb, 1], 4):
                score2 = 1
            else: score2 = 0
            st.latex(r'\frac{'f'1[{round(st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 0], 4)}>{round(st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 1], 4)}] + 1[{round(st.session_state.dists_real_syn_gower[indice, 0], 4)}>{round(st.session_state.dists_syn_syn_gower[syn_nb, 1], 4)}]'r'}{'f'{len(st.session_state.dists_real_real_gower)}'r'} = 'f'{round((score1 + score2)/len(st.session_state.dists_real_real_gower), 5)}')
        else: 
            score2 = 0
            st.latex(r'0.5 * \frac{'f'1[{round(st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 0], 4)}>{round(st.session_state.dists_real_real_gower[st.session_state.indiv_index, 1], 4)}] + 1[{0}>{0}]'r'}{'f'{len(st.session_state.dists_real_real_gower)}'r'} = 'f'{round(0.5*(score1 + score2)/len(st.session_state.dists_real_real_gower), 5)}')
        st.subheader("Problems Overall:")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Problems for you:")
        st.markdown(prob_u_txt, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
        st.write("**The problems that may occur:**")
        st.write(" - The metric is easy to cheat. If the synthetic data isn’t “too close” to the training data. However, when doing this, non-private data can still be produced:")
        st.image("images/distance_threshold_problem.png", caption="*https://desfontain.es/blog/bad-ugly-good-maybe.html*")
    
    with col2:
        st.write("Scatter plot of real data mapped using PCA:")
        st.pyplot(scatter_plot_real(st.session_state.coord_real))
        st.write("Scatter plot of real and synthetic data mapped using PCA:")
        st.pyplot(scatter_plot(st.session_state.coord_real, st.session_state.syn_coords))
        if indice != None:
            st.write("Scatter plot of synthetic data point with you as its neighbour and its synthetic neighbour:")
            st.pyplot(scatter_plot(pd.concat([st.session_state.coord_real.iloc[[st.session_state.idx_real_real_gower[len(st.session_state.coord_real)-1, 1]]], st.session_state.coord_real.iloc[[len(st.session_state.coord_real)-1]]]),
                                        pd.concat([st.session_state.syn_coords.iloc[[indice]], st.session_state.syn_coords.iloc[[syn_nb]]])))
        else:
            st.write("Scatter plot of you and your nearest neighbour in the real and synthetic data:")
            st.pyplot(scatter_plot(pd.concat([st.session_state.coord_real.iloc[[st.session_state.idx_real_real_gower[len(st.session_state.coord_real)-1, 1]]], st.session_state.coord_real.iloc[[len(st.session_state.coord_real)-1]]]),
                                            st.session_state.syn_coords.iloc[[st.session_state.idx_real_syn_gower[len(st.session_state.syn_coords)-1, 0]]]))

    st.button(label="Go Back", on_click=set_state, args=[2])
    
if st.session_state.stage == 17: #CRP
    tit="Common Row Proportion (CRP)"
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    status_u = '✅'
    prob_u_txt = ''
    if st.session_state.has_continuous:
        prob_overall += f'<br>- Continuous attributes ({st.session_state.cont_cols}) can not be used, as the noise induced by the synthesizer renders this theoretically impossible.'
        sol_overall += '<br>- Remove all continuous attributes.'
        status_overall = '⛔️'
        status_u = '⛔️'
    if status_overall == '⚠️' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '✅' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '⚠️' and status_u == '⚠️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
    if status_overall == '⛔️' and status_u == '⛔️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
    st.title(tit)
    st.subheader(f"App: {status_overall} User App: {status_u}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.subheader("Overall")
        st.write("CRP measures the risk of re-identification as a probability of a real individual's row being a row in the synthetic data.")
        st.write("The score is therefore calculated as:")
        st.latex(r"\frac{|real \cap synthetic|}{|real|}")
        
        st.subheader("For your data")
        st.write("You have no matching rows. Therefore, you are not at risk, and you contribute with 0 to the score.")
        st.subheader("Problems Overall:")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Problems for you:")
        st.markdown(prob_u_txt, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
    
    with col2:
        st.write("Real Dataset:")
        st.dataframe(st.session_state.real_data, use_container_width=True, hide_index=True, height=150)
        st.write("Your Synthetic Dataset:")
        st.dataframe(st.session_state.syn_data_bin, use_container_width=True, hide_index=True, height=150)
        
    st.button(label="Go Back", on_click=set_state, args=[2])
    
if st.session_state.stage == 18: #NSND
    tit="Nearest Synthetic Neighbour Distance Ratio (NSND)"
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    status_u = '✅'
    prob_u_txt = ''
    if st.session_state.has_continuous:
        prob_overall += f'<br>- Distances lose expresivity and vary much for continuous attributes ({st.session_state.cont_cols}).'
        sol_overall += f'<br>- Consider removing continuous attributes ({st.session_state.cont_cols})'
        status_overall = '⚠️'
    if st.session_state.is_large:
        prob_overall += '<br>- Distance can be misleading in high-dimensional spaces.'
        sol_overall += '<br>- Consider decreasing the dataset size.'
    if st.session_state.has_prob_syn_neigh:
        prob_overall += '<br>- Non-private data is still be produced.'
        sol_overall += '<br>- Investigate each real datapoint and its 3 nearest neighbours in real and synthetic.'
        status_overall = '⛔️'
        status_u = '⛔️'
    if status_overall == '⚠️' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '✅' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '⚠️' and status_u == '⚠️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
    if status_overall == '⛔️' and status_u == '⛔️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
        
    st.title(tit)
    st.subheader(f"App: {status_overall} User App: {status_u}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.subheader("Overall")
        st.write("NSND measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated through a distance measure.")
        st.write("The score is calculated as the mean min-max reduced distance to the nearest synthetic neighbour.")
        st.write("*To make the score fit the risk measure, the score is subtracted from 1.*")
        st.subheader("For your data")
        st.write("Your nearest synthetic neighbour:")
        st.dataframe(st.session_state.syn_data_bin.iloc[[st.session_state.idxs_syn[st.session_state.indiv_index, 0]]], use_container_width=True, hide_index=True)
        st.write(f"With distance: {round(st.session_state.dists_syn[st.session_state.indiv_index, 0], 2)}")
        st.write("For your record, the NSND contribution would therefore be:")
        st.latex(r'\frac{'f'{round(st.session_state.dists_syn[st.session_state.indiv_index, 0], 2)} - {round(min(st.session_state.dists_syn[:, 0]), 2)}(min(dist))'r'}{'f'{round(max(st.session_state.dists_syn[:, 0]), 2)}(max(dist)) - {round(min(st.session_state.dists_syn[:, 0]), 2)}(min(dist)) + 1e-8'r'} = 'f'{round(((st.session_state.dists_syn[st.session_state.indiv_index, 0] - min(st.session_state.dists_syn[:, 0]))) / (max(st.session_state.dists_syn[:, 0]) - min(st.session_state.dists_syn[:, 0]) + 1e-8), 2)}')
        st.subheader("Problems Overall:")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Problems for you:")
        st.markdown(prob_u_txt, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
        st.write("**The problems that may occur:**")
        st.write(" - The metric is easy to cheat. If the synthetic data isn’t “too close” to the training data. However, when doing this, non-private data can still be produced:")
        st.image("images/distance_threshold_problem.png", caption="*https://desfontain.es/blog/bad-ugly-good-maybe.html*")
    
    with col2:
        st.write("Scatter plot of real and synthetic data:")
        st.pyplot(scatter_plot_tsne(st.session_state.real_coords_tsne, st.session_state.syn_coords_tsne))
        
        st.session_state.tsne_df_real = pd.DataFrame(st.session_state.real_coords_tsne)
        st.session_state.tsne_df_syn = pd.DataFrame(st.session_state.syn_coords_tsne)
        st.write("Scatter plot of you and your nearest neighbour in the real and synthetic data:")
        st.pyplot(scatter_plot_tsne(pd.concat([st.session_state.tsne_df_real.iloc[[st.session_state.idxs_real[len(st.session_state.tsne_df_real)-1, 1]]], st.session_state.tsne_df_real.iloc[[len(st.session_state.tsne_df_real)-1]]]),
                                                st.session_state.tsne_df_syn.iloc[[st.session_state.idxs_syn[len(st.session_state.tsne_df_syn)-1, 1]]]))
    
    st.button(label="Go Back", on_click=set_state, args=[2])
    
if st.session_state.stage == 19: #CVP
    tit="Close Value Probability (CVP)"
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    status_u = '✅'
    prob_u_txt = ''
    
    if st.session_state.is_large:
        prob_overall += '<br>- Distance can be misleading in high-dimensional spaces.'
        sol_overall += '<br>- Consider decreasing the dataset size.'
    if st.session_state.has_continuous:
        prob_overall += f'<br>- Distances lose expresivity and vary much for continuous attributes ({st.session_state.cont_cols}).'
        sol_overall += '<br>- Remove all continuous attributes.'
        status_overall = '⚠️'
        status_u = '⚠️'
        
    prob_overall += '<br>- Finding a "correct" threshold is a very difficult task.'
    sol_overall += '<br>- Establish a threshold matching distances in datasets.'
    status_overall = '⛔️'
    status_u = '⛔️'
    
    if status_overall == '⚠️' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '✅' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '⚠️' and status_u == '⚠️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
    if status_overall == '⛔️' and status_u == '⛔️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
    
    st.title(tit)
    st.subheader(f"App: {status_overall} User App: {status_u}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.subheader("Overall")
        st.write("CVP measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated.")
        st.write("The attacker follows two steps to re-identify individuals:")
        st.write("1. Find all instances where the distance to the nearest neighbour in the synthetic dataset is less than a given threshold (in our case 0.2).")
        st.write("2. Calculate the average probability of this happening in the real dataset to get the CVP.")
        st.subheader("For your data")
        st.write("Your nearest synthetic neighbour:")
        st.dataframe(st.session_state.syn_data_bin.iloc[[st.session_state.idxs_syn[st.session_state.indiv_index, 0]]], use_container_width=True, hide_index=True)
        st.write(f"With distance: {round(st.session_state.dists_syn[st.session_state.indiv_index, 0], 2)}")
        if round(st.session_state.dists_syn[st.session_state.indiv_index, 0], 2) < 0.2:
            contribution = 1
        else: contribution = 0
        st.write("For your record, the CVP contribution would therefore be:")
        st.latex(r'\frac{'f'{contribution}'r'}{'f'{len(st.session_state.dists_real)}'r'} = 'f'{round(contribution / len(st.session_state.dists_real))}')
        st.subheader("Problems Overall:")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Problems for you:")
        st.markdown(prob_u_txt, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
        st.write("**The problems that may occur:**")
        st.write(" - Thresholds are easy to cheat by the synthesizer. Just make sure that sensitive data is at least the threshold away from the original sensitive field.")
        st.write(' - Finding a "correct" threshold is a very difficult task.')
        st.write("This may be visible from looking at the minimum distance between real and synthetic nearest neighbour data points, which is:")
        st.write("Real to synthetic:", round(np.min(st.session_state.dists_syn[:, 0]), 2))
        
    with col2:
        st.write("Scatter plot of real and synthetic data:")
        st.pyplot(scatter_plot_tsne(st.session_state.real_coords_tsne, st.session_state.syn_coords_tsne))
        
        st.session_state.tsne_df_real = pd.DataFrame(st.session_state.real_coords_tsne)
        st.session_state.tsne_df_syn = pd.DataFrame(st.session_state.syn_coords_tsne)
        st.write("Scatter plot of you and your nearest neighbour in the real and synthetic data:")
        st.pyplot(scatter_plot_tsne(pd.concat([st.session_state.tsne_df_real.iloc[[st.session_state.idxs_real[len(st.session_state.tsne_df_real)-1, 1]]], st.session_state.tsne_df_real.iloc[[len(st.session_state.tsne_df_real)-1]]]),
                                                st.session_state.tsne_df_syn.iloc[[st.session_state.idxs_syn[len(st.session_state.tsne_df_syn)-1, 1]]]))
    
    st.button(label="Go Back", on_click=set_state, args=[2])
      
if st.session_state.stage == 20: #DVP
    tit="Distant Value Probability (DVP)"
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    status_u = '✅'
    prob_u_txt = ''
    if st.session_state.has_continuous:
        prob_overall += f'<br>- Distances lose expresivity and vary much for continuous attributes ({st.session_state.cont_cols}).'
        sol_overall += '<br>- Remove all continuous attributes.'
        status_overall = '⚠️'
        status_u = '⚠️'
    if st.session_state.is_large:
        prob_overall += '<br>- Distance can be misleading in high-dimensional spaces.'
        sol_overall += '<br>- Consider decreasing the dataset size.'
    prob_overall += '<br>- Finding a "correct" threshold is a very difficult task.'
    sol_overall += '<br>- Establish a threshold matching distances in datasets.'
    status_overall = '⛔️'
    status_u = '⛔️'
    
    if status_overall == '⚠️' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '✅' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '⚠️' and status_u == '⚠️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
    if status_overall == '⛔️' and status_u == '⛔️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
        
    st.title(tit)
    st.subheader(f"App: {status_overall} User App: {status_u}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.subheader("Overall")
        st.write("DVP measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated.")
        st.write("The attacker follows three steps to re-identify individuals:")
        st.write("1. Find all instances where the distance to the nearest neighbour in the synthetic dataset is longer than a given threshold (in our case 0.8).")
        st.write("2. Calculate the average probability of this happening in the real dataset.")
        st.write("3. Subtract this number from 1 to get the DVP.")
        st.subheader("For your data")
        st.write("Your nearest synthetic neighbour:")
        st.dataframe(st.session_state.syn_data_bin.iloc[[st.session_state.idxs_syn[st.session_state.indiv_index, 0]]], use_container_width=True, hide_index=True)
        st.write(f"With distance: {round(st.session_state.dists_syn[st.session_state.indiv_index, 0], 2)}")
        if round(st.session_state.dists_syn[st.session_state.indiv_index, 0], 2) > 0.8:
            contribution = 1
        else: contribution = 0
        st.write("For your record, the DVP contribution would therefore be:")
        st.latex(r'\frac{'f'{-contribution}'r'}{'f'{len(st.session_state.dists_real)}'r'} = 'f'{round(-contribution / len(st.session_state.dists_real), 4)}')
        st.subheader("Problems Overall:")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Problems for you:")
        st.markdown(prob_u_txt, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
        st.write("**The problems that may occur:**")
        st.write(" - Thresholds are easy to cheat by the synthesizer. Just make sure that sensitive data is at least the threshold away from the original sensitive field.")
        st.write(' - Finding a "correct" threshold is a very difficult task.')
        st.write("This may be visible from looking at the minimum distance between real and synthetic nearest neighbour data points, which is:")
        st.write("Real to synthetic:", round(np.min(st.session_state.dists_syn[:, 0]), 2))
    
    with col2:
        st.write("Scatter plot of real and synthetic data:")
        st.pyplot(scatter_plot_tsne(st.session_state.real_coords_tsne, st.session_state.syn_coords_tsne))
        
        st.session_state.tsne_df_real = pd.DataFrame(st.session_state.real_coords_tsne)
        st.session_state.tsne_df_syn = pd.DataFrame(st.session_state.syn_coords_tsne)
        st.write("Scatter plot of you and your nearest neighbour in the real and synthetic data:")
        st.pyplot(scatter_plot_tsne(pd.concat([st.session_state.tsne_df_real.iloc[[st.session_state.idxs_real[len(st.session_state.tsne_df_real)-1, 1]]], st.session_state.tsne_df_real.iloc[[len(st.session_state.tsne_df_real)-1]]]),
                                                st.session_state.tsne_df_syn.iloc[[st.session_state.idxs_syn[len(st.session_state.tsne_df_syn)-1, 1]]]))
      
    st.button(label="Go Back", on_click=set_state, args=[2])
    
if st.session_state.stage == 21: #Authenticity
    tit="Authenticity (Auth)"
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    status_u = '✅'
    prob_u_txt = ''
    if st.session_state.is_large:
        prob_overall += '<br>- Distance can be misleading in high-dimensional spaces.'
        sol_overall += '<br>- Consider decreasing the dataset size.'
    if st.session_state.has_prob_syn_neigh:
        prob_overall += '<br>- Non-private data is still be produced.'
        sol_overall += '<br>- Investigate each real datapoint and its 3 nearest neighbours in real and synthetic.'
        status_overall = '⛔️'
    if status_overall == '⚠️' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '✅' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '⚠️' and status_u == '⚠️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
    if status_overall == '⛔️' and status_u == '⛔️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
        
    st.title(tit)
    st.subheader(f"App: {status_overall} User App: {status_u}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.subheader("Overall")
        st.write("Auth measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated.")
        st.write("The Auth risk is measured as the probability that a synthetic nearest neighbour is closer than a real nearest neighbour over the real dataset.")
        st.subheader("For your data")
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            st.write("Your real neighbour:")
            st.dataframe(st.session_state.real_data.iloc[[st.session_state.idxs_real[st.session_state.indiv_index, 1]]], use_container_width=True, hide_index=True)
            st.write(f"With distance: {round(st.session_state.dists_real[st.session_state.indiv_index, 1], 2)}")
            
        with col1_2:
            st.write("Your syn neighbour:")
            st.dataframe(st.session_state.syn_data_bin.iloc[[st.session_state.idxs_syn[st.session_state.indiv_index, 0]]], use_container_width=True, hide_index=True)
            st.write(f"With distance: {round(st.session_state.dists_syn[st.session_state.indiv_index, 0], 2)}")
        
        if (st.session_state.dists_syn[st.session_state.indiv_index, 0] - st.session_state.dists_real[st.session_state.indiv_index, 1]) < 0:
            ur_score = round(1 / len(st.session_state.real_data), 2)
        else: ur_score = 0
        st.write("For your record, the IdScore contribution would therefore be:")
        st.latex(r"\frac{1["f"{round(st.session_state.dists_syn[st.session_state.indiv_index, 0], 2)} - {round(st.session_state.dists_real[st.session_state.indiv_index, 1], 2)}"r"< 0]}{"f"{len(st.session_state.real_data)}"r"} = "f"{ur_score}")
        st.subheader("Problems Overall:")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Problems for you:")
        st.markdown(prob_u_txt, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
        st.write("**The problems that may occur:**")
        st.write(" - A low score indicates low risk, but a non-zero score means that some individuals are at risk of re-identification")
        col11, col12 = st.columns(2, border=False)
        with col11:
            st.write("One can imagine a case like this ➡️")
            st.write("Now imagine a scenario in which there are more than 1 real individual in the circle.")
            st.write("Here, according to Auth, they are not at risk, however the re-identification risk of such is at least 0.5, and can be larger if an adversary knows any information about the individual at risk.")
        with col12:
            st.image("images/distance_threshold_problem.png", caption="*https://desfontain.es/blog/bad-ugly-good-maybe.html*")
            
    with col2:
        st.write("Scatter plot of real and synthetic data:")
        st.pyplot(scatter_plot_tsne(st.session_state.real_coords_tsne, st.session_state.syn_coords_tsne))
        
        st.session_state.tsne_df_real = pd.DataFrame(st.session_state.real_coords_tsne)
        st.session_state.tsne_df_syn = pd.DataFrame(st.session_state.syn_coords_tsne)
        st.write("Scatter plot of you and your nearest neighbour in the real and synthetic data:")
        st.pyplot(scatter_plot_tsne(pd.concat([st.session_state.tsne_df_real.iloc[[st.session_state.idxs_real[len(st.session_state.tsne_df_real)-1, 1]]], st.session_state.tsne_df_real.iloc[[len(st.session_state.tsne_df_real)-1]]]),
                                                st.session_state.tsne_df_syn.iloc[[st.session_state.idxs_syn[len(st.session_state.tsne_df_syn)-1, 1]]]))
    
    st.button(label="Go Back", on_click=set_state, args=[2])
 
if st.session_state.stage == 22: #DMLP
    tit="DetectionMLP (D-MLP)"
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    status_u = '✅'
    prob_u_txt = ''
    if st.session_state.has_prob_syn_neigh:
        prob_overall += '<br>- Non-private data is still be produced.'
        sol_overall += '<br>- Investigate each real datapoint and its 3 nearest neighbours in real and synthetic.'
        status_overall = '⛔️'
        status_u = '⛔️'
    prob_overall += '<br>- Assumes that the adversary has very much knowledge about the real data.'
    sol_overall += '<br>- Try multiple different subset as a training set to get a better evaluation.'
    status_overall = '⚠️'
    status_u = '⚠️'
    
    if status_overall == '⚠️' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '✅' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '⚠️' and status_u == '⚠️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
    if status_overall == '⛔️' and status_u == '⛔️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
        
    st.title(tit)
    st.subheader(f"App: {status_overall} User App: {status_u}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.subheader("Overall")
        st.write("D-MLP measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated, while having access to a subset of the real data.")
        st.write("The attacker follows four steps to re-identify individuals:")
        st.write("1. Make a new dataset that contains both the real and synthetic data as well as a labeling of whether or not the are real.")
        st.write("2. Split the dataset up into a train and test set.")
        st.write("3. Train a MLP classifier on the train set.")
        st.write("4. Measure the AUC of the classification task on the test set to get D-MLP.")
        st.subheader("Problems Overall:")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Problems for you:")
        st.markdown(prob_u_txt, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
        st.write("**The problems that may occur:**")
        st.write(" - Classification models are easy to cheat. If the synthetic data isn’t “too close” to the training data. However, when doing this, non-private data can still be produced:")
        st.image("images/distance_threshold_problem.png", caption="*https://desfontain.es/blog/bad-ugly-good-maybe.html*")
    with col2:
        st.write("Scatter plot of real and synthetic data:")
        st.pyplot(scatter_plot_tsne(st.session_state.real_coords_tsne, st.session_state.syn_coords_tsne))
    
    st.button(label="Go Back", on_click=set_state, args=[2])
    
if st.session_state.stage == 23: #Identifiability
    tit="Identifiability Score (IdScore)"
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    status_u = '✅'
    prob_u_txt = ''
    if st.session_state.is_large:
        prob_overall += '<br>- Distance can be misleading in high-dimensional spaces.'
        sol_overall += '<br>- Consider decreasing the dataset size.'
    if st.session_state.has_prob_syn_neigh:
        prob_overall += '<br>- Non-private data is still be produced.'
        sol_overall += '<br>- Investigate each real datapoint and its 3 nearest neighbours in real and synthetic.'
        status_overall = '⛔️'
        status_u = '⚠️'
    if status_overall == '⚠️' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '✅' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '⚠️' and status_u == '⚠️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
    if status_overall == '⛔️' and status_u == '⛔️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
        
    st.title(tit)
    st.subheader(f"App: {status_overall} User App: {status_u}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.subheader("Overall")
        st.write('''IdScore estimates the risk of re-identifying any real individual while only having access to the synthetic data. 
                It estimates this as the probability that the distance to the closest synthetic individual is closer than the distance from the closest real individual in weighted versions of the real and synthetic dataset.
                Here, the weight is assigned as a contribution factor for each individual attribute value.
                ''')
        st.write("The IdScore is the calculated as the fraction of weighted individuals where their distance to a synthetic individual is smaller than the distance to a real individual.")
        X_gt_ = st.session_state.real_labels.to_numpy().reshape(len(st.session_state.real_data), -1)
        X_syn_ = st.session_state.syn_labels.to_numpy().reshape(len(st.session_state.syn_data_bin), -1)
        
        def compute_entropy(labels: np.ndarray) -> np.ndarray:
            from scipy.stats import entropy
            value, counts = np.unique(np.round(labels), return_counts=True)
            return entropy(counts)
        no, x_dim = X_gt_.shape
        W = np.zeros(
            [
                x_dim,
            ]
        )
        for i in range(x_dim):
            W[i] = compute_entropy(X_gt_[:, i])
        X_hat = X_gt_
        X_syn_hat = X_syn_
        eps = st.session_state.epsilon
        W = np.ones_like(W)
        for i in range(x_dim):
            X_hat[:, i] = X_gt_[:, i] * 1.0 / (W[i] + eps)
            X_syn_hat[:, i] = X_syn_[:, i] * 1.0 / (W[i] + eps)
        nbrs = NearestNeighbors(n_neighbors=2).fit(X_hat)
        distance_r, indx_r = nbrs.kneighbors(X_hat)
        # hat{r_i} computation
        nbrs_hat = NearestNeighbors(n_neighbors=1).fit(X_syn_hat)
        distance_hat, indx_s = nbrs_hat.kneighbors(X_hat)
        # See which one is bigger
        R_Diff = distance_hat[st.session_state.indiv_index, 0] - distance_r[st.session_state.indiv_index, 1]
        identifiability_value_indiv = np.sum(R_Diff < 0) / float(no)
        st.subheader("For your data")
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            st.write("Your real neighbour:")
            st.dataframe(st.session_state.real_data.iloc[[indx_r[st.session_state.indiv_index, 1]]], use_container_width=True, hide_index=True)
            st.write(f"With weighted distance: {round(distance_r[st.session_state.indiv_index, 1], 2)}")
            
        with col1_2:
            st.write("Your syn neighbour:")
            st.dataframe(st.session_state.syn_data_bin.iloc[[indx_s[st.session_state.indiv_index, 0]]], use_container_width=True, hide_index=True)
            st.write(f"With weighted distance: {round(distance_hat[st.session_state.indiv_index, 0], 2)}")
            
        if R_Diff < 0:
            contribution_id = round(1 / len(st.session_state.real_data), 5)
        else: contribution_id = 0
        st.write("For your record, the IdScore contribution would therefore be:")
        st.latex(r"\frac{1["f"{round(distance_hat[st.session_state.indiv_index, 0], 2)} - {round(distance_r[st.session_state.indiv_index, 1], 2)}"r"< 0]}{"f"{len(st.session_state.real_data)}"r"} = "f"{contribution_id}")
        st.subheader("Problems Overall:")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Problems for you:")
        st.markdown(prob_u_txt, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
        st.write("**The problems that may occur:**")
        st.write(" - A low score indicates low risk, but a non-zero score means that some individuals are at risk of re-identification")
        col11, col12 = st.columns(2, border=False)
        with col11:
            st.write("One can imagine a case like this ➡️")
            st.write("Now imagine a scenario in which there are more than 1 real individual in the circle.")
            st.write("Here, according to IdScore, they are not at risk, however the re-identification risk of such is at least 0.5, and can be larger if an adversary knows any information about the individual at risk.")
        with col12:
            st.image("images/distance_threshold_problem.png", caption="*https://desfontain.es/blog/bad-ugly-good-maybe.html*")

    with col2:
        st.write("Scatter plot of real and synthetic data:")
        st.pyplot(scatter_plot_tsne(st.session_state.real_coords_tsne, st.session_state.syn_coords_tsne))
        
        
        st.write("Scatter plot of you and your nearest neighbour in the real and synthetic data:")
        st.pyplot(scatter_plot_tsne(pd.concat([st.session_state.tsne_df_real.iloc[[indx_r[st.session_state.indiv_index, 1]]], st.session_state.tsne_df_real.iloc[[st.session_state.indiv_index]]]),
                                                st.session_state.tsne_df_syn.iloc[[indx_s[st.session_state.indiv_index, 0]]]))

    st.button(label="Go Back", on_click=set_state, args=[2])
    
if st.session_state.stage == 24: #DCR
    tit="Distance to Closest Record (DCR)"
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    status_u = '✅'
    prob_u_txt = ''
    if st.session_state.has_continuous:
        prob_overall += f'<br>- Distances lose expresivity and vary much for continuous attributes ({st.session_state.cont_cols}).'
        sol_overall += '<br>- Consider removing continuous attributes.'
        status_overall = '⚠️'
    if st.session_state.is_large:
        prob_overall += '<br>- Distance can be misleading in high-dimensional spaces.'
        sol_overall += '<br>- Consider decreasing the dataset size.'
    if st.session_state.has_prob_syn_neigh:
        prob_overall += '<br>- Non-private data is still be produced.'
        sol_overall += '<br>- Investigate each real datapoint and its 3 nearest neighbours in real and synthetic.'
        status_overall = '⛔️'
        status_u = '⛔️'
    if status_overall == '⚠️' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '✅' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '⚠️' and status_u == '⚠️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
    if status_overall == '⛔️' and status_u == '⛔️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
        
    st.title(tit)
    st.subheader(f"App: {status_overall} User App: {status_u}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.subheader("Overall")
        st.write("DCR measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated.")
        st.write("DCR is the calculated in three steps:")
        st.write("1. Use PCA, MCA and/or FAMD to map the datasets to 2 dimensions.")
        st.write("2. Use the mapped dataset to determine the nearest neighbour distances from real to synthetic individuals.")
        st.write("3. Calculate the mean of the distances.")
        st.write("*To make the score fit the risk measure, we take the logarithm of the mean, and apply the sigmoid function. This is then subtracted from 1.")
        st.subheader("For your data")
        st.write("You:")
        st.dataframe(st.session_state.real_data.iloc[[st.session_state.indiv_index]], use_container_width=True, hide_index=True)
        st.write("Your syn neighbour:")
        st.dataframe(st.session_state.syn_data_bin.iloc[[st.session_state.idx_real_syn_gower[st.session_state.indiv_index, 0]]], use_container_width=True, hide_index=True)
        st.write(f"With distance: {round(st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 0], 4)}")
        st.write("Thereby, your contribution to the score is:")
        from math import log
        def sigmoid(x):
            return 1/(1 + np.exp(-x)) 
        res = 1 - sigmoid(log(st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 0], 10))
        st.latex(rf"1 - \sigma(log({round(st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 0], 4)})) = {round(res, 2)}")
        st.subheader("Problems Overall:")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Problems for you:")
        st.markdown(prob_u_txt, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
        st.write("**The problems that may occur:**")
        st.write(" - The metric is easy to cheat. If the synthetic data isn’t “too close” to the training data. However, when doing this, non-private data can still be produced:")
        st.image("images/distance_threshold_problem.png", caption="*https://desfontain.es/blog/bad-ugly-good-maybe.html*")
    
    with col2:
        st.write("Scatter plot of real data mapped using PCA:")
        st.pyplot(scatter_plot_real(st.session_state.coord_real))
        st.write("Scatter plot of real and synthetic data mapped using PCA:")
        st.pyplot(scatter_plot(st.session_state.coord_real, st.session_state.syn_coords))
        st.write("Scatter plot of you and your nearest neighbour in the synthetic data:")
        st.pyplot(scatter_plot(st.session_state.coord_real.iloc[[len(st.session_state.coord_real)-1]],
                            st.session_state.syn_coords.iloc[[st.session_state.idx_real_syn_gower[len(st.session_state.syn_coords)-1, 0]]]))
    
    st.button(label="Go Back", on_click=set_state, args=[2])
    
if st.session_state.stage == 25: #NNDR
    tit="Nearest Neighbour Distance Ratio (NNDR)"
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    status_u = '✅'
    prob_u_txt = ''
    if st.session_state.has_continuous:
        prob_overall += f'<br>- Distances lose expresivity and vary much for continuous attributes ({st.session_state.cont_cols}).'
        sol_overall += '<br>- Consider removing continuous attributes.'
        status_overall = '⚠️'
    if st.session_state.is_large:
        prob_overall += '<br>- Distance can be misleading in high-dimensional spaces.'
        sol_overall += '<br>- Consider decreasing the dataset size.'
    if st.session_state.has_prob_syn_neigh:
        prob_overall += '<br>- Non-private data is still be produced.'
        sol_overall += '<br>- Investigate each real datapoint and its 3 nearest neighbours in real and synthetic.'
        status_overall = '⛔️'
        status_u = '⚠️'
    if status_overall == '⚠️' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '✅' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '⚠️' and status_u == '⚠️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
    if status_overall == '⛔️' and status_u == '⛔️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
        
    st.title(tit)
    st.subheader(f"App: {status_overall} User App: {status_u}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.subheader("Overall")
        st.write("NNDR measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated.")
        st.write("NNDR is the calculated in three steps:")
        st.write("1. Use PCA, MCA and/or FAMD to map the datasets to 2 dimensions.")
        st.write("2. Use the mapped dataset to determine the nearest and 2nd nearest neighbour distances from real to synthetic individuals.")
        st.write("3. Calculate the ratio between the distances of the nearest and 2nd nearest synthetic neighbour.")
        st.write("*To make the score fit the risk measure, the score is subtracted from 1.*")
        st.subheader("For your data")
        st.write("You:")
        st.dataframe(st.session_state.real_data.iloc[[st.session_state.indiv_index]], use_container_width=True, hide_index=True)
        st.write("Your syn neighbour:")
        st.dataframe(st.session_state.syn_data_bin.iloc[[st.session_state.idx_real_syn_gower[st.session_state.indiv_index, 0]]], use_container_width=True, hide_index=True)
        st.write(f"With distance: {round(st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 0], 4)}")
        st.write("Your 2nd syn neighbour:")
        st.dataframe(st.session_state.syn_data_bin.iloc[[st.session_state.idx_real_syn_gower[st.session_state.indiv_index, 1]]], use_container_width=True, hide_index=True)
        st.write(f"With distance: {round(st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 1], 4)}")
        
        st.write("Thereby, your contribution to the score is:")
        ratio = 1 if st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 0] == 0 else (st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 0] / st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 1])
        
        st.latex(r"1 - \frac{"f"{round(st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 0], 4)}"r"}{"f"{round(st.session_state.dists_real_syn_gower[st.session_state.indiv_index, 1], 4)}"r"} = "f"{round(ratio, 4)}")
        st.subheader("Problems Overall:")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Problems for you:")
        st.markdown(prob_u_txt, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
        st.write("**The problems that may occur:**")
        st.write(" - The metric is easy to cheat. If the synthetic data isn’t “too close” to the training data. However, when doing this, non-private data can still be produced:")
        st.image("images/distance_threshold_problem.png", caption="*https://desfontain.es/blog/bad-ugly-good-maybe.html*")
        
    with col2:
        st.write("Scatter plot of real data mapped using PCA:")
        st.pyplot(scatter_plot_real(st.session_state.coord_real))
        st.write("Scatter plot of real and synthetic data mapped using PCA:")
        st.pyplot(scatter_plot(st.session_state.coord_real, st.session_state.syn_coords))
        st.write("Scatter plot of you and your nearest neighbour in the synthetic data:")
        st.pyplot(scatter_plot(st.session_state.coord_real.iloc[[len(st.session_state.coord_real)-1]],
                            pd.concat([st.session_state.syn_coords.iloc[[st.session_state.idx_real_syn_gower[len(st.session_state.syn_coords)-1, 0]]], st.session_state.syn_coords.iloc[[st.session_state.idx_real_syn_gower[len(st.session_state.syn_coords)-1, 1]]]])))
    
    st.button(label="Go Back", on_click=set_state, args=[2])
    
if st.session_state.stage == 26: #Hidden Rate
    tit="Hidden Rate"
    prob_overall = ''
    sol_overall = ''
    status_overall = '✅'
    status_u = '✅'
    prob_u_txt = ''
    prob_overall += '<br>- Synthetic individuals need to be generated from the real individual with same index.'
    sol_overall += '<br>- Use synthesizer that generates individuals that are based on the individual with the same index in the real data.'
    status_overall = '⛔️'
    status_u = '⛔️'
    if status_overall == '⚠️' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '✅' and status_u == '✅':
        prob_u_txt = 'For you, the overall issues still persist. However, your score is calculated correctly.'
    if status_overall == '⚠️' and status_u == '⚠️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
    if status_overall == '⛔️' and status_u == '⛔️':
        prob_u_txt = 'For you, the overall issues still persist, and your score calculation is influenced by this.'
        
    st.title(tit)
    st.subheader(f"App: {status_overall} User App: {status_u}")
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.subheader("Overall")
        st.write('''Hidden Rate estimates the risk of identifying whether an individual contributed their data to the real dataset while only having access to the synthetic data.''')
        st.write("Hidden Rate test for each record if the nearest synthetic individual is the one generated by the real individual itself.")
        st.write("Hidden Rate is the calculated in three steps:")
        st.write("1. Use PCA, MCA and/or FAMD to map the datasets to 2 dimensions.")
        st.write("2. Use the mapped dataset to determine the nearest neighbour distances from real to synthetic.")
        st.write("3. Calculate Hidden Rate as the probability that the synthetic neighbour has the same index as the real individual.")
        st.subheader("For your data")
        st.write("Your index is:", st.session_state.indiv_index)
        st.write("Your syn neighbour:")
        st.dataframe(st.session_state.syn_data_bin.iloc[[st.session_state.idx_real_syn_gower[st.session_state.indiv_index, 0]]], use_container_width=True, hide_index=True)
        st.write("With index:", st.session_state.idx_real_syn_gower[st.session_state.indiv_index, 0])
        if st.session_state.indiv_index == st.session_state.idx_real_syn_gower[st.session_state.indiv_index, 0]:
            st.write("You have matching indexes, and your contribution to the score is therefore:")
            st.latex(r"\frac{1}{"f"{len(st.session_state.real_data)}"r"} = "f"{round(1/len(st.session_state.real_data), 2)}")
        else:
            st.write("You do not have matching indexes, and your contribution to the score is therefore:")
            st.latex(r"\frac{0}{"f"{len(st.session_state.real_data)}"r"} = 0")
        st.subheader("Problems Overall:")
        st.markdown(prob_overall, unsafe_allow_html=True)
        st.subheader("Problems for you:")
        st.markdown(prob_u_txt, unsafe_allow_html=True)
        st.subheader("Solutions:")
        st.markdown(sol_overall, unsafe_allow_html=True)
        
    with col2:
        st.write("Scatter plot of real data mapped using PCA:")
        st.pyplot(scatter_plot_real(st.session_state.coord_real))
        st.write("Scatter plot of real and synthetic data mapped using PCA:")
        st.pyplot(scatter_plot(st.session_state.coord_real, st.session_state.syn_coords))
    
    st.button(label="Go Back", on_click=set_state, args=[2])
        
st.button(label="Start Over", on_click=set_state, args=[0])