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
# Show the page title and description.
st.set_page_config(page_title="Privacy estimation", page_icon="😈", layout='wide')
st.title("Privacy Estimation of Your Data")
#st.session_state.stage = 0
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
        'First_Name': [name_gen.first_name() for _ in range(num_samples)],
        'Last_Name': [name_gen.last_name() for _ in range(num_samples)]
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
            person["Favorite Ice Cream"] = np.random.choice(ice_creams, p=[0.1, 0.1, 0.1, 0.1, 0.2, 0.4])
        else:
            person["Favorite Ice Cream"] = random.choice(ice_creams)

        # Assign liking for liquorice (Nordic countries → Higher probability)
        if person["Country of Origin"] in ["Sweden", "Norway", "Denmark", "Finland"]:
            person["Likes Liquorice"] = np.random.choice([1, 0], p=[0.9, 0.1])  # 70% chance for Nordic countries
        else:
            person["Likes Liquorice"] = np.random.choice([1, 0], p=[0.2, 0.8])  # 20% for others

        # Assign number of times visited Italy (Random integer, but higher if from Europe)
        if person["Country of Origin"] in ["Germany", "France", "UK", "Sweden", "Norway", "Denmark", "Finland", "Italy"]:
            person["Times Visited Italy"] = np.random.poisson(2)  # Higher average visits
        else:
            person["Times Visited Italy"] = np.random.poisson(0.5)  # Lower average visits

        # First time in London (UK residents more likely to say yes)
        person["First Time in London"] = 1 if person["Country of Origin"] == "UK" else np.random.choice([1, 0], p=[0.2, 0.8])

        # Number of steps per day (Normal distribution with realistic values)
        person["Steps per Day"] = max(1000, int(np.random.normal(8000, 3000)))  # Avoids negative steps

        data.append(person)

    # Create DataFrame
    df = pd.DataFrame(data)
    
    full_df = pd.concat([basic_df, df], axis=1)
    
    if liquorice == 0:
        # Sample row: UK resident who does NOT like liquorice
        indiv = {
            "First_Name": "James",
            "Last_Name": "Smith",
            "Height": round(random.gauss(175, 10), 2),
            "Country of Origin": "UK",
            "Favorite Ice Cream": "Strawberry",
            "Likes Liquorice": 0,
            "Times Visited Italy": 2,
            "First Time in London": 0,
            "Steps per Day": 7500
        }

    if liquorice == 1:
        # Sample row: Sweden resident who LIKES liquorice
        indiv = {
            "First_Name": "Lars",
            "Last_Name": "Andersson",
            "Height": round(random.gauss(185, 10), 2), 
            "Country of Origin": "Sweden",
            "Favorite Ice Cream": "Chocolate",
            "Likes Liquorice": 1,
            "Times Visited Italy": 3,
            "First Time in London": 0,
            "Steps per Day": 8000
        }
    full_df = pd.concat([full_df, indiv], ignore_index=True)
        
    # Save to CSV (optional)
    full_df.to_csv("sample_people_data.csv", index=False)

def scatter_plot_real(coord_real):
    your_x = coord_real['Dim. 1'].iloc[len(coord_real)-1]
    your_y = coord_real['Dim. 2'].iloc[len(coord_real)-1]
    # Scatter Plot
    plt.figure()
    # Plot DataFrame 1
    plt.scatter(coord_real['Dim. 1'], coord_real['Dim. 2'], color='royalblue', label='Real', alpha=1)
    plt.scatter(your_x, your_y, marker='x', color='cyan', edgecolors='black', linewidths=2,  label='You', alpha=1)
    # Plot DataFrame 2
    plt.title('Scatter Plot of real data')
    plt.xlabel('X-coord')
    plt.ylabel('Y-coord')
    plt.legend()
    plt.grid(True)

    # Show Plot
    #plt.show()
    return plt
def scatter_plot_real_tsne(coord_real):
    real = pd.DataFrame(coord_real)
    your_x = real[0].iloc[len(coord_real)-1]
    your_y = real[1].iloc[len(coord_real)-1]
    # Scatter Plot
    plt.figure()
    # Plot DataFrame 1
    plt.scatter(real[0], real[1], color='royalblue', label='Real', alpha=0.5)
    plt.scatter(your_x, your_y, marker='x', color='cyan', edgecolors='black', linewidths=2,  label='You', alpha=1)
    # Plot DataFrame 2
    plt.title('Scatter Plot of real data (TSNE)')
    plt.xlabel('X-coord')
    plt.ylabel('Y-coord')
    plt.legend()
    plt.grid(True)

    return plt
def scatter_plot(coord_real, coord_synth):
    your_x = coord_real['Dim. 1'].iloc[len(coord_real)-1]
    your_y = coord_real['Dim. 2'].iloc[len(coord_real)-1]
    # Scatter Plot
    plt.figure()
    # Plot DataFrame 1
    plt.scatter(coord_real['Dim. 1'], coord_real['Dim. 2'], color='royalblue', label='Real', alpha=1)
    # Plot DataFrame 2
    plt.scatter(coord_synth['Dim. 1'], coord_synth['Dim. 2'], color='red', label='Synthetic', alpha=0.5)
    plt.scatter(your_x, your_y, marker='x', color='cyan', edgecolors='black', linewidths=2,  label='You', alpha=1)
    
    plt.title('Scatter Plot of real and synthetic data')
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
    your_x = real[0].iloc[len(coord_real)-1]
    your_y = real[1].iloc[len(coord_real)-1]
    # Scatter Plot
    plt.figure()

    # Plot DataFrame 1
    plt.scatter(real[0], real[1], color='royalblue', label='Real', alpha=0.5)

    # Plot DataFrame 2
    plt.scatter(syn[0], syn[1], color='red', label='Synthetic', alpha=0.5)
    
    #Plot you
    plt.scatter(your_x, your_y, marker='x', color='cyan', edgecolors='black', linewidths=2,  label='You', alpha=1)
    

    plt.title('Scatter Plot of real and synthetic data (TSNE)')
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

def metric_applicability(metric_results):
    #<br>- Distance measures are easy to cheat by the synthesizer. Just make sure that synthetic datapoints are not within a certain distance threshold.
    applicability_column = pd.DataFrame(metric_results['Metric'])
    applicability_column['App.'] = '✅'
    
    applicability_column.loc[applicability_column['Metric']=='ZeroCAP', 'App.'] = '⛔️'
    applicability_column.loc[applicability_column['Metric']=='GeneralizedCAP', 'App.'] = '⚠️'
    applicability_column.loc[applicability_column['Metric']=='Median Distance to Closest Record', 'App.'] = '⚠️'
    applicability_column.loc[applicability_column['Metric']=='Hitting Rate', 'App.'] = '⛔️'
    applicability_column.loc[applicability_column['Metric']=='Membership Inference Risk', 'App.'] = '⚠️'
    applicability_column.loc[applicability_column['Metric']=='Nearest Neighbour Adversarial Accuracy', 'App.'] = '⚠️'
    applicability_column.loc[applicability_column['Metric']=='Common Row Proportion', 'App.'] = '⛔️'
    applicability_column.loc[applicability_column['Metric']=='Nearest Synthetic Neighbour Distance', 'App.'] = '⚠️'
    applicability_column.loc[applicability_column['Metric']=='Close Value Probability', 'App.'] = '⛔️'
    applicability_column.loc[applicability_column['Metric']=='Distant Value Probability', 'App.'] = '⛔️'
    applicability_column.loc[applicability_column['Metric']=='Authenticity', 'App.'] = '⚠️'
    applicability_column.loc[applicability_column['Metric']=='DetectionMLP', 'App.'] = '⚠️'
    applicability_column.loc[applicability_column['Metric']=='Identifiability Score', 'App.'] = '✅'
    applicability_column.loc[applicability_column['Metric']=='Attribute Inference Risk', 'App.'] = '⚠️'
    applicability_column.loc[applicability_column['Metric']=='Distance to Closest Record', 'App.'] = '⚠️'
    applicability_column.loc[applicability_column['Metric']=='Nearest Neighbour Distance Ratio', 'App.'] = '⚠️'
    applicability_column.loc[applicability_column['Metric']=='Hidden Rate', 'App.'] = '⚠️'
    
    problem_column = pd.DataFrame(metric_results['Metric'])
    problem_column['Problem'] = '✅'
    
    problem_column.loc[problem_column['Metric']=='ZeroCAP', 'Problem'] = '- Continuous attributes can not be used.'
    problem_column.loc[problem_column['Metric']=='GeneralizedCAP', 'Problem'] = '- Your sensitive attribute is *Like Liquorice*, however if you sensitive attribute is continuous, the metric will not work. <br>- If key fields are continuous, in the nearest neighbour algorithm, continuous attributes influence the distance measure differently than other attributes.'
    problem_column.loc[problem_column['Metric']=='Median Distance to Closest Record', 'Problem'] = '- Distances lose expresivity for continuous attributes. <br>- Distance measures do not account for patterns in the distribution.<br>- Distance can be misleading in high-dimensional spaces.<br>- Median is not representative of the risk for a single individual.'
    problem_column.loc[problem_column['Metric']=='Hitting Rate', 'Problem'] = '⛔️'
    problem_column.loc[problem_column['Metric']=='Membership Inference Risk', 'Problem'] = '⚠️'
    problem_column.loc[problem_column['Metric']=='Nearest Neighbour Adversarial Accuracy', 'Problem'] = '⚠️'
    problem_column.loc[problem_column['Metric']=='Common Row Proportion', 'Problem'] = '- Continuous attributes can not be used, as the noise induced by the synthesizer renders this theoretically impossible.'
    problem_column.loc[problem_column['Metric']=='Nearest Synthetic Neighbour Distance', 'Problem'] = '⚠️'
    problem_column.loc[problem_column['Metric']=='Close Value Probability', 'Problem'] = '- Distances lose expresivity for continuous attributes.<br>- Distance can be misleading in high-dimensional spaces. <br>- Thresholds are easy to cheat by the synthesizer. Just make sure that sensitive data is at least the threshold away from the original sensitive field. <br>- Finding a "correct" threshold is a very difficult task.'
    problem_column.loc[problem_column['Metric']=='Distant Value Probability', 'Problem'] = '- Distances lose expresivity for continuous attributes.<br>- Distance can be misleading in high-dimensional spaces. <br>- Thresholds are easy to cheat by the synthesizer. Just make sure that sensitive data is at least the threshold away from the original sensitive field. <br>- Finding a "correct" threshold is a very difficult task.'
    problem_column.loc[problem_column['Metric']=='Authenticity', 'Problem'] = '⚠️'
    problem_column.loc[problem_column['Metric']=='DetectionMLP', 'Problem'] = '⚠️'
    problem_column.loc[problem_column['Metric']=='Identifiability Score', 'Problem'] = '- A low score indicates low risk, but a non-zero score means that some individuals are at risk of re-identification'
    problem_column.loc[problem_column['Metric']=='Attribute Inference Risk', 'Problem'] = '- Your key fields contain continuous attributes, in the nearest neighbour algorithm, continuous attributes influence the distance measure differently than other attributes.<br>- Distance measures do not account for patterns in the distribution. <br>- Distance can be misleading in high-dimensional spaces <br>- Thresholds are easy to cheat by the synthesizer. Just make sure that sensitive data is at least the threshold away from the original sensitive field.'
    problem_column.loc[problem_column['Metric']=='Distance to Closest Record', 'Problem'] = '⚠️'
    problem_column.loc[problem_column['Metric']=='Nearest Neighbour Distance Ratio', 'Problem'] = '⚠️'
    problem_column.loc[problem_column['Metric']=='Hidden Rate', 'Problem'] = '⚠️'
    
    solution_column = pd.DataFrame(metric_results['Metric'])
    solution_column['Possible Solution'] = '✅'
    
    solution_column.loc[solution_column['Metric']=='ZeroCAP', 'Possible Solution'] = 'Remove all continuous attributes.'
    solution_column.loc[solution_column['Metric']=='GeneralizedCAP', 'Possible Solution'] = 'Remove all continuous attributes.'
    solution_column.loc[solution_column['Metric']=='Median Distance to Closest Record', 'Possible Solution'] = 'Invetigate the distances between individual data points.'
    solution_column.loc[solution_column['Metric']=='Hitting Rate', 'Possible Solution'] = '⛔️'
    solution_column.loc[solution_column['Metric']=='Membership Inference Risk', 'Possible Solution'] = '⚠️'
    solution_column.loc[solution_column['Metric']=='Nearest Neighbour Adversarial Accuracy', 'Possible Solution'] = '⚠️'
    solution_column.loc[solution_column['Metric']=='Common Row Proportion', 'Possible Solution'] = 'Remove all continuous attributes.'
    solution_column.loc[solution_column['Metric']=='Nearest Synthetic Neighbour Distance', 'Possible Solution'] = '⚠️'
    solution_column.loc[solution_column['Metric']=='Close Value Probability', 'Possible Solution'] = '- Remove all continuous attributes. <br>- Establish a "correct" threshold.'
    solution_column.loc[solution_column['Metric']=='Distant Value Probability', 'Possible Solution'] = '- Remove all continuous attributes. <br>- Establish a "correct" threshold.'
    solution_column.loc[solution_column['Metric']=='Authenticity', 'Possible Solution'] = '⚠️'
    solution_column.loc[solution_column['Metric']=='DetectionMLP', 'Possible Solution'] = '⚠️'
    solution_column.loc[solution_column['Metric']=='Identifiability Score', 'Possible Solution'] = 'Investigate individuals contributing positively to the risk.'
    solution_column.loc[solution_column['Metric']=='Attribute Inference Risk', 'Possible Solution'] = 'Remove all continuous attributes in the key fields.'
    solution_column.loc[solution_column['Metric']=='Distance to Closest Record', 'Possible Solution'] = '⚠️'
    solution_column.loc[solution_column['Metric']=='Nearest Neighbour Distance Ratio', 'Possible Solution'] = '⚠️'
    solution_column.loc[solution_column['Metric']=='Hidden Rate', 'Possible Solution'] = '⚠️'
        
    applicability_df = metric_results.merge(applicability_column, on='Metric')
    applicability_df.rename(columns={"Result": "Risk"})
    problem_df = applicability_df.merge(problem_column, on='Metric')
    solution_df = problem_df.merge(solution_column, on='Metric')
    return solution_df

def set_state(i):
    st.session_state.stage = i

if st.session_state.stage == 0:
    st.write("This app demonstrates privacy estimation of differentially private synthetic data, and the risks that may be associated with relying on current available metrics.")
    st.write("You are one of the individuals contributing to a dataset which you want to synthesize.")
    st.write("Please input your contribution to the dataset:")
    like_liquorice = st.selectbox("Do you like liquorice", (0, 1))
    fav_ice = st.text_input("What is your favorite Icecream?")
    first_time = st.selectbox("Is this your first time in London?", (0, 1))
    height = st.slider("How tall are you (in cm)?", 0.0, 240.0, 170.0)
    epsilon = st.selectbox("What ε-value do you wich to use to synthesize your dataset (lower = more private)?", (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2.5, 5))
    st.session_state.real_data = pd.read_csv(f'sample_data_{like_liquorice}.csv', index_col=False)
    st.session_state.syn_data_bin = pd.read_csv(f'demo_syn/syn_no_{like_liquorice}_{epsilon}.csv', index_col=False).drop(columns=['Unnamed: 0']) #They got switched around during synthesis
    st.session_state.syn_data_no_bin = pd.read_csv(f'demo_syn/syn_bin_{like_liquorice}_{epsilon}.csv', index_col=False).drop(columns=['Unnamed: 0'])#They got switched around during synthesis
    st.session_state.metric_results_bin = pd.read_csv(f'metric_results/syn_no_{like_liquorice}_{epsilon}.csv', index_col=False).drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])#They got switched around during synthesis
    st.session_state.metric_results_no_bin = pd.read_csv(f'metric_results/syn_bin_{like_liquorice}_{epsilon}.csv', index_col=False).drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])#They got switched around during synthesis
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
    st.button(label="Submit", on_click=set_state, args=[1])

if st.session_state.stage >= 1:
    st.write("Based on your input, this person is most likely equivalent to you, and is therefore assigned to you:")
    st.dataframe(st.session_state.real_data.iloc[[len(st.session_state.real_data)-1]], use_container_width=True, hide_index=True)
    st.write("Your data is a part of a real dataset, and using the epsilon you desired, a synthetic dataset has been generated. Below is a description of both the real dataset and the synthetic one.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Real Dataset")
        st.write(f"Number of individuals in dataset: {len(st.session_state.real_data)}")
        num_unique_trans = pd.DataFrame(st.session_state.real_data.nunique()).transpose()
        num_unique = pd.DataFrame(num_unique_trans, columns = st.session_state.real_data.columns)
        st.write(f"Number of unique values for each column:")
        st.dataframe(num_unique, use_container_width=True, hide_index=True)
        most_frequent_real = pd.DataFrame(st.session_state.real_data.apply(lambda col: col.value_counts().idxmax()))
        most_frequent_real_df = pd.DataFrame(most_frequent_real.transpose(), columns=st.session_state.real_data.columns)
        st.write(f"These are the most frequent values for each column:")
        st.dataframe(most_frequent_real_df, use_container_width=True, hide_index=True)
        st.write(f"Description of the numeric data:")
        st.dataframe(round(st.session_state.real_data.describe()[1:], 2), use_container_width=True)
        st.write(f"Scatter plot of real data:")
        st.pyplot(scatter_plot_real_tsne(st.session_state.real_coords_tsne), use_container_width = True)
    with col2:
        st.subheader("Synthetic Dataset")
        st.write(f"Number of individuals in dataset: {len(st.session_state.syn_data_bin)}")
        num_unique_trans_syn = pd.DataFrame(st.session_state.syn_data_bin.nunique()).transpose()
        num_unique_syn = pd.DataFrame(num_unique_trans_syn, columns = st.session_state.syn_data_bin.columns)
        st.write(f"Number of unique values for each column:")
        st.dataframe(num_unique_syn, use_container_width=True, hide_index=True)
        most_frequent_syn = pd.DataFrame(st.session_state.syn_data_bin.apply(lambda col: col.value_counts().idxmax()))
        most_frequent_syn_df = pd.DataFrame(most_frequent_syn.transpose(), columns=st.session_state.syn_data_bin.columns)
        st.write(f"These are the most frequent values for each column:")
        st.dataframe(most_frequent_syn_df, use_container_width=True, hide_index=True)
        st.write(f"Description of the numeric data:")
        st.dataframe(round(st.session_state.syn_data_bin.describe()[1:], 2), use_container_width=True)
        st.write(f"Scatter plot of real and synthetic data:")
        st.pyplot(scatter_plot_tsne(st.session_state.real_coords_tsne, st.session_state.syn_coords_tsne), use_container_width = True)
    st.write("You want to publish your synthetic dataset, but what does this mean for you and the other individuals' privacy?")
    st.write("Using privacy metrics, we can measure how private the synthetic dataset is. Click the button to estimate the privacy:")
if st.session_state.stage == 1:
    st.button(label="Measure Privacy", on_click=set_state, args=[2])

if st.session_state.stage >= 2:
    st.write('''Here, you can see how private your synthetic data is through the use of multiple metric results. 
             These metrics measure the privacy as a risk measure with a score in range [0,1], 
             where a high score means high privacy risk and vice versa.''')
    st.write("Furthermore, to elaborate how the assumptions of the metrics influence their ability to measure privacy, a description of possible problems is given.")
    st.write("The applicability in this scenario is elaborated as follows:")
    st.write("✅: The metric is good, and no assumption is missing")
    st.write("⚠️: The metric requires some assumption which is potentially not met")
    st.write("⛔️: The metric is not reliable in any sense.")
    st.markdown(metric_applicability(st.session_state.metric_results_bin).to_html(escape=False, index=False), unsafe_allow_html=True)
    st.write("If you desire, you can explore how the different metrics are computed, and how there may be issues when using them below ⬇️")
    idScore_tab, dmlp_tab, auth_tab, dvp_tab, cvp_tab, nsnd_tab, crp_tab, nnaa_tab, mir_tab, hitr_tab, mdcr_tab, zcap_tab, gcap_tab, air_tab, dcr_tab, nndr_tab, hidd_tab = st.tabs(
        ["IdScore", "D-MLP", "Auth", "DVP", "CVP", "NSND", "CRP", "NNAA", "MIR", "HitR", "MDCR", "ZCAP", "GCAP", "AIR", "DCR", "NNDR", "Hidden Rate"])
    
    with idScore_tab:
        st.subheader("Identifiability Score (IdScore):")
        col1, col2 = st.columns(2, border=True)
        with col1:
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
            eps = 1e-16
            W = np.ones_like(W)
            for i in range(x_dim):
                X_hat[:, i] = X_gt_[:, i] * 1.0 / (W[i] + eps)
                X_syn_hat[:, i] = X_syn_[:, i] * 1.0 / (W[i] + eps)
            nbrs = NearestNeighbors(n_neighbors=2).fit(X_hat)
            distance, _ = nbrs.kneighbors(X_hat)
            # hat{r_i} computation
            nbrs_hat = NearestNeighbors(n_neighbors=1).fit(X_syn_hat)
            distance_hat, _ = nbrs_hat.kneighbors(X_hat)
            # See which one is bigger
            R_Diff = distance_hat[len(st.session_state.real_data)-1, 0] - distance[len(st.session_state.real_data)-1, 1]
            identifiability_value_indiv = np.sum(R_Diff < 0) / float(no)
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(st.session_state.real_labels)
            dists_real, idxs_real = nn.kneighbors(st.session_state.real_labels)
            dists_syn, idxs_syn = nn.kneighbors(st.session_state.syn_labels)
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.write("Real neighbour:")
                st.dataframe(st.session_state.real_data.iloc[[idxs_real[len(st.session_state.real_data)-1, 1]]], use_container_width=True, hide_index=True)
                st.write(f"With weighted distance: {round(distance[len(st.session_state.real_data)-1, 1], 2)}")
                
            with col1_2:
                st.write("Synthetic neighbour:")
                st.dataframe(st.session_state.syn_data_bin.iloc[[idxs_syn[len(st.session_state.real_data)-1, 1]]], use_container_width=True, hide_index=True)
                st.write(f"With weighted distance: {round(distance_hat[len(st.session_state.real_data)-1, 0], 2)}")
                
            st.write("For your record, the equation would therefore be:")
            if R_Diff < 0:
                contribution_id = round(1 / len(st.session_state.real_data), 2)
            else: contribution_id = 0
            st.write("For your record, the IdScore contribution would therefore be:")
            st.latex(r"\frac{"f"{round(distance_hat[len(st.session_state.real_data)-1, 0], 2)} - {round(distance[len(st.session_state.real_data)-1, 1], 2)}"r"< 0}{"f"{len(st.session_state.real_data)}"r"} = "f"{contribution_id}")

        with col2:
            st.write("Scatter plot of real and synthetic data:")
            st.pyplot(scatter_plot_tsne(st.session_state.real_coords_tsne, st.session_state.syn_coords_tsne))
            
            st.session_state.tsne_df_real = pd.DataFrame(st.session_state.real_coords_tsne)
            st.session_state.tsne_df_syn = pd.DataFrame(st.session_state.syn_coords_tsne)
            st.write("Scatter plot of you and your nearest neighbour in the real and synthetic data:")
            st.pyplot(scatter_plot_tsne(pd.concat([st.session_state.tsne_df_real.iloc[[idxs_real[len(st.session_state.tsne_df_real)-1, 1]]], st.session_state.tsne_df_real.iloc[[len(st.session_state.tsne_df_real)-1]]]),
                                                    st.session_state.tsne_df_syn.iloc[[idxs_syn[len(st.session_state.tsne_df_syn)-1, 1]]]))
        
        st.write("**The problems that occur:**")
        st.write(" - A low score indicates low risk, but a non-zero score means that some individuals are at risk of re-identification")
        st.write(" - Distance can be misleading in high-dimensional spaces.")
        
    with dmlp_tab:
        st.subheader("DetectionMLP (D-MLP):")
        col1, col2 = st.columns(2, border=True)
        with col1:
            st.write("D-MLP measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated, while having access to a subset of the real data.")
            st.write("The attacker follows four steps to re-identify individuals:")
            st.write("1. Make a new dataset that contains both the real and synthetic data as well as a labeling of whether or not the are real.")
            st.write("2. Split the dataset up into a train and test set.")
            st.write("3. Train a MLP classifier on the train set.")
            st.write("4. Measure the AUC of the classification task on the test set to get D-MLP.")
        with col2:
            st.write("Scatter plot of real and synthetic data:")
            st.pyplot(scatter_plot_tsne(st.session_state.real_coords_tsne, st.session_state.syn_coords_tsne))
        
        st.write("**The problems that occur:**")
        st.write(" - classification models are easy to cheat.If the synthetic data isn’t “too close” to the training data. However, when doing this, non-private data can still be produced:")
        st.image("images/distance_threshold_problem.png")
        
    with auth_tab:
        st.subheader("Authenticity (Auth):")
        st.write("Auth measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated.")
        st.write("The Auth risk is measured as the probability that a synthetic nearest neighbour is closer than a real nearest neighbour over the real dataset.")
        
    with dvp_tab:        
        st.subheader("Distant Value Probability (DVP):")
        col1, col2 = st.columns(2, border=True)
        with col1:
            st.write("DVP measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated.")
            st.write("The attacker follows three steps to re-identify individuals:")
            st.write("1. Find all instances where the distance to the nearest neighbour in the synthetic dataset is longer than a given threshold (in our case 0.8).")
            st.write("2. Calculate the average probability of this happening in the real dataset.")
            st.write("3. Subtract this number from 1 to get the DVP.")
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(st.session_state.real_labels)
            dists_real, idxs_real = nn.kneighbors(st.session_state.real_labels)
            dists_syn, idxs_syn = nn.kneighbors(st.session_state.syn_labels)
            st.write("Your nearest synthetic neighbour:")
            st.dataframe(st.session_state.syn_data_bin.iloc[[idxs_syn[len(st.session_state.real_data)-1, 1]]], use_container_width=True, hide_index=True)
            st.write(f"With distance: {round(dists_syn[len(st.session_state.real_data)-1, 1], 2)}")
            if round(dists_syn[len(st.session_state.real_data)-1, 1], 2) > 0.8:
                contribution = 1
            else: contribution = 0
            st.write("For your record, the DVP contribution would therefore be:")
            st.latex(r'\frac{'f'{contribution}'r'}{'f'{len(dists_real)}'r'} = 'f'{round(contribution / len(dists_real), 4)}')
        
        with col2:
            st.write("Scatter plot of real and synthetic data:")
            st.pyplot(scatter_plot_tsne(st.session_state.real_coords_tsne, st.session_state.syn_coords_tsne))
            
            st.session_state.tsne_df_real = pd.DataFrame(st.session_state.real_coords_tsne)
            st.session_state.tsne_df_syn = pd.DataFrame(st.session_state.syn_coords_tsne)
            st.write("Scatter plot of you and your nearest neighbour in the real and synthetic data:")
            st.pyplot(scatter_plot_tsne(pd.concat([st.session_state.tsne_df_real.iloc[[idxs_real[len(st.session_state.tsne_df_real)-1, 1]]], st.session_state.tsne_df_real.iloc[[len(st.session_state.tsne_df_real)-1]]]),
                                                    st.session_state.tsne_df_syn.iloc[[idxs_syn[len(st.session_state.tsne_df_syn)-1, 1]]]))
        st.write("**The problems that occur:**")
        st.write(" - Distances lose expresivity for high distributions as well as continuous attributes.")
        st.write(" - Distance can be misleading in high-dimensional spaces.")
        st.write(" - Thresholds are easy to cheat by the synthesizer. Just make sure that sensitive data is at least the threshold away from the original sensitive field.")
        st.write(' - Finding a "correct" threshold is a very difficult task.')
        st.write("This may be visible from looking at the minimum distance between real and synthetic nearest neighbour data points, which is:")
        st.write("Real to synthetic:", round(np.min(dists_syn[1]), 2))
    
    with cvp_tab:        
        st.subheader("Close Value Probability (CVP):")
        col1, col2 = st.columns(2, border=True)
        with col1:
            st.write("CVP measures the risk of re-identification by assessing how easily an attacker, using the synthetic data, can infer the individual from which it was generated.")
            st.write("The attacker follows two steps to re-identify individuals:")
            st.write("1. Find all instances where the distance to the nearest neighbour in the synthetic dataset is less than a given threshold (in our case 0.2).")
            st.write("2. Calculate the average probability of this happening in the real dataset to get the CVP.")
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(st.session_state.real_labels)
            dists_real, idxs_real = nn.kneighbors(st.session_state.real_labels)
            dists_syn, idxs_syn = nn.kneighbors(st.session_state.syn_labels)
            st.write("Your nearest synthetic neighbour:")
            st.dataframe(st.session_state.syn_data_bin.iloc[[idxs_syn[len(st.session_state.real_data)-1, 1]]], use_container_width=True, hide_index=True)
            st.write(f"With distance: {round(dists_syn[len(st.session_state.real_data)-1, 1], 2)}")
            if round(dists_syn[len(st.session_state.real_data)-1, 1], 2) < 0.2:
                contribution = 1
            else: contribution = 0
            st.write("For your record, the CVP contribution would therefore be:")
            st.latex(r'\frac{'f'{contribution}'r'}{'f'{len(dists_real)}'r'} = 'f'{round(contribution / len(dists_real))}')
        
        with col2:
            st.write("Scatter plot of real and synthetic data:")
            st.pyplot(scatter_plot_tsne(st.session_state.real_coords_tsne, st.session_state.syn_coords_tsne))
            
            st.session_state.tsne_df_real = pd.DataFrame(st.session_state.real_coords_tsne)
            st.session_state.tsne_df_syn = pd.DataFrame(st.session_state.syn_coords_tsne)
            st.write("Scatter plot of you and your nearest neighbour in the real and synthetic data:")
            st.pyplot(scatter_plot_tsne(pd.concat([st.session_state.tsne_df_real.iloc[[idxs_real[len(st.session_state.tsne_df_real)-1, 1]]], st.session_state.tsne_df_real.iloc[[len(st.session_state.tsne_df_real)-1]]]),
                                                    st.session_state.tsne_df_syn.iloc[[idxs_syn[len(st.session_state.tsne_df_syn)-1, 1]]]))
        st.write("**The problems that occur:**")
        st.write(" - Distances lose expresivity for high distributions as well as continuous attributes.")
        st.write(" - Distance can be misleading in high-dimensional spaces.")
        st.write(" - Thresholds are easy to cheat by the synthesizer. Just make sure that sensitive data is at least the threshold away from the original sensitive field.")
        st.write(' - Finding a "correct" threshold is a very difficult task.')
        st.write("This may be visible from looking at the minimum distance between real and synthetic nearest neighbour data points, which is:")
        st.write("Real to synthetic:", round(np.min(dists_syn[1]), 2))
        
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
            st.write("The attacker follows four steps to measure the re-identification risk:")
            st.write("1. For each real individual find the distance to the nearest neighbour in the real dataset.")
            st.write("2. For each real individual find the distance to the nearest neighbour in the synthetic dataset.")
            st.write("3. Calculate the median of distances between real individuals.")
            st.write("4. Calculate the median of distances between real and synthetic individuals.")
            st.write("The MDCR is then calculated as:")
            st.latex(r'''\frac{\mu(dists\phantom{i}real\phantom{i}to\phantom{i}real)}
                                  {\mu(dists\phantom{i}real\phantom{i}to\phantom{i}synthetic)}''')
            st.write("Your nearest neighbours for this metric is:")
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(st.session_state.real_labels)
            dists_real, idxs_real = nn.kneighbors(st.session_state.real_labels)
            dists_syn, idxs_syn = nn.kneighbors(st.session_state.syn_labels)
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.write("Real neighbour:")
                st.dataframe(st.session_state.real_data.iloc[[idxs_real[len(st.session_state.real_data)-1, 1]]], use_container_width=True, hide_index=True)
                st.write(f"With distance: {round(dists_real[len(st.session_state.real_data)-1, 1], 2)}")
                
            with col1_2:
                st.write("Synthetic neighbour:")
                st.dataframe(st.session_state.syn_data_bin.iloc[[idxs_syn[len(st.session_state.real_data)-1, 1]]], use_container_width=True, hide_index=True)
                st.write(f"With distance: {round(dists_syn[len(st.session_state.real_data)-1, 1], 2)}")
                
            st.write("For your record, the equation would therefore be:")
            st.latex(r'\frac{'f'{round(dists_real[len(st.session_state.real_data)-1, 1], 2)}'r'}{'f'{round(dists_syn[len(st.session_state.real_data)-1, 1], 2)}'r'} = 'f'{round(round(dists_real[len(st.session_state.real_data)-1, 1], 2) / round(dists_syn[len(st.session_state.real_data)-1, 1], 2), 2)}')
        
        with col2:
            st.write("Scatter plot of real and synthetic data:")
            st.pyplot(scatter_plot_tsne(st.session_state.real_coords_tsne, st.session_state.syn_coords_tsne))
            
            st.session_state.tsne_df_real = pd.DataFrame(st.session_state.real_coords_tsne)
            st.session_state.tsne_df_syn = pd.DataFrame(st.session_state.syn_coords_tsne)
            st.write("Scatter plot of you and your nearest neighbour in the real and synthetic data:")
            st.pyplot(scatter_plot_tsne(pd.concat([st.session_state.tsne_df_real.iloc[[idxs_real[len(st.session_state.tsne_df_real)-1, 1]]], st.session_state.tsne_df_real.iloc[[len(st.session_state.tsne_df_real)-1]]]),
                                                    st.session_state.tsne_df_syn.iloc[[idxs_syn[len(st.session_state.tsne_df_syn)-1, 1]]]))
            
        
        st.write("**The problems that occur:**")
        st.write(" - Distances lose expresivity for high distributions as well as continuous attributes.")
        st.write(" - Distance can be misleading in high-dimensional spaces.")
        st.write("This may be visible from looking at the two neighbour with the smallest and largest distance between them:")
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            st.write("Minimum distance neighbour in the real dataset:")
            dists = pd.DataFrame(dists_real)[1]
            min_index = np.argmin(dists)
            st.dataframe(st.session_state.real_data.iloc[[min_index]], use_container_width=True, hide_index=True)
            st.dataframe(st.session_state.real_data.iloc[[idxs_real[min_index, 1]]], use_container_width=True, hide_index=True)
            st.write(f"With distance {round(dists[min_index], 2)}")
            st.write("Maximum distance neighbour in the real dataset:")
            max_index = np.argmax(dists)
            st.dataframe(st.session_state.real_data.iloc[[max_index]], use_container_width=True, hide_index=True)
            st.dataframe(st.session_state.real_data.iloc[[idxs_real[max_index, 1]]], use_container_width=True, hide_index=True)
            st.write(f"With distance {round(dists[max_index], 2)}")
        with col1_2:
            st.write("Minimum distance neighbour in the synthetic dataset:")
            distsyn = pd.DataFrame(dists_syn)[1]
            min_indexsyn = np.argmin(distsyn)
            st.dataframe(st.session_state.syn_data_bin.iloc[[min_indexsyn]], use_container_width=True, hide_index=True)
            st.dataframe(st.session_state.syn_data_bin.iloc[[idxs_syn[min_indexsyn, 1]]], use_container_width=True, hide_index=True)
            st.write(f"With distance {round(distsyn[min_indexsyn], 2)}")
            st.write("Maximum distance neighbour in the synthetic dataset:")
            max_indexsyn = np.argmax(distsyn)
            st.dataframe(st.session_state.syn_data_bin.iloc[[max_indexsyn]], use_container_width=True, hide_index=True)
            st.dataframe(st.session_state.syn_data_bin.iloc[[idxs_syn[max_indexsyn, 1]]], use_container_width=True, hide_index=True)
            st.write(f"With distance {round(distsyn[max_indexsyn], 2)}")
             
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
            key_fields = ['First Name', 'Last Name', 'Height', 'Nationality', 'Favorite Icecream', 'Times Been to Italy', 'First Time London', 'Steps per Day']
            ind_vals = st.session_state.real_data[key_fields].iloc[[len(st.session_state.real_data)-1]]
            st.dataframe(ind_vals, hide_index=True)
            syndat=st.session_state.syn_data_bin
            if any((ind_vals == syndat[key_fields].iloc[i]).all(axis=1).any() for i in range(len(syndat[key_fields]))):
                st.write("These rows with matching key fields in the synthetic dataset:")
                matching_rows = st.session_state.syn_data_bin[st.session_state.syn_data_bin.apply(lambda row: (ind_vals == row[key_fields]).all(axis=1).any(), axis=1)]
                st.dataframe(matching_rows, hide_index=True)
                st.write("Your row contributes:")
                st.latex(r'''\frac{|rows\phantom{i}with\phantom{i}same\phantom{i}key\phantom{i}and\phantom{i}sensitive\phantom{i}fields|}
                                  {|rows\phantom{i}with\phantom{i}same\phantom{i}key\phantom{i}fields|}''')
            
            if not any((ind_vals == syndat[key_fields].iloc[i]).all(axis=1).any() for i in range(len(syndat[key_fields]))):
                st.write("There are not any synthetic rows with matching key fields.")
                st.write("Therefore, the score for your data is 0.")
        st.write("**The problems that occur:**")
        st.write("You have continuous attributes inthe key fields. Therefore, the randomness induced by the synthesizer makes finding a match highly unlikely.")
            
        with col2:
            st.write("Real Dataset:")
            st.dataframe(st.session_state.real_data, use_container_width=True, hide_index=True)
            st.write("Your Synthetic Dataset:")
            st.dataframe(st.session_state.syn_data_bin, use_container_width=True, hide_index=True)
            #st.session_state.real_data
        
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
            key_fields = ['First Name', 'Last Name', 'Height', 'Nationality', 'Favorite Icecream', 'Times Been to Italy', 'First Time London', 'Steps per Day']
            ind_vals = st.session_state.real_data[key_fields].iloc[[len(st.session_state.real_data)-1]]
            st.dataframe(ind_vals, use_container_width=True, hide_index=True)
            syndat=st.session_state.syn_data_bin
            if any((ind_vals == syndat[key_fields].iloc[i]).all(axis=1).any() for i in range(len(syndat[key_fields]))):
                st.write("These rows have matching key fields in the synthetic dataset:")
                matching_rows = st.session_state.syn_data_bin[st.session_state.syn_data_bin.apply(lambda row: (ind_vals == row[key_fields]).all(axis=1).any(), axis=1)]
                st.dataframe(matching_rows, hide_index=True, use_container_width=True)
                st.write("Your row contributes:")
                st.latex(r'''\frac{|rows\phantom{i}with\phantom{i}same\phantom{i}key\phantom{i}and\phantom{i}sensitive\phantom{i}fields|}
                                  {|rows\phantom{i}with\phantom{i}same\phantom{i}key\phantom{i}fields|}''')
            
            else:
                neighbour_index, neighbour, distance = nearest_neighbor_hamming(ind_vals, syndat[key_fields])
                st.write("This row is your nearest synthetic neighbouring key fields:")
                st.dataframe(neighbour, use_container_width=True, hide_index=True)
                st.write("These are the sensitive fields for both individuals:")
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    st.dataframe(pd.DataFrame({'Your Sensitive Field': st.session_state.real_data['Like Liquorice'].iloc[[len(st.session_state.real_data)-1]]}), use_container_width=True, hide_index=True)
                with col1_2:
                    st.dataframe(pd.DataFrame({"Neighbour's Sensitive Field": syndat['Like Liquorice'].iloc[[neighbour_index]]}), use_container_width=True, hide_index=True)
                
                if syndat['Like Liquorice'].values[neighbour_index] == st.session_state.real_data['Like Liquorice'].values[len(st.session_state.real_data)-1]:
                    st.write("You have the same sensitive field, and your privacy is therefore in jeopardy.")
                    st.write("You therefore contribute a score of 1 to the metric calculation.")
                else:
                    st.write("You do not have the same sensitive field, and your privacy is maintained.")
                    st.write("You therefore contribute a score of 0 to the metric calculation.")
        st.write("**The problems that may occur:**")
        st.write("There are continuous attributes in the key fields. Therefore, finding a neighbour is influenced differently for these. This may be vissible from the key fields of your 2nd nearest neighbour:")
        neighbour_index1, neighbour1, distance1 = nearest_neighbor_hamming(ind_vals, syndat[key_fields].drop([syndat[key_fields].index[neighbour_index]]))
        st.dataframe(neighbour1, use_container_width=True, hide_index=True)
                    
                    
        with col2:
            st.write("Real Dataset:")
            st.dataframe(st.session_state.real_data, use_container_width=True, hide_index=True)
            st.write("Your Synthetic Dataset:")
            st.dataframe(st.session_state.syn_data_bin, use_container_width=True, hide_index=True)

    with air_tab:        
            st.subheader("Attribute Inference Risk (AIR):")
            col1, col2 = st.columns(2, border=True)
            with col1:
                st.write("AIR measures the risk of inference attacks by assessing how easily an attacker, using public real data and synthetic data, can infer sensitive values. It quantifies this difficulty with the a weighted F1-score.")
                st.write("*To calculate this metric, a one-hot encoding for categorical attributes must be used.*")
                st.write("The attacker follows four steps to guess a sensitive value:")
                st.write("1. Select a row from the real dataset and note its key fields.")
                st.write("2. Find the (k=1) nearest synthetic neighbour(s) using a normalized Hamming distance on the key fields.")
                st.write("3. Evaluate the binary and continuous attributes seperately for infering the sensitive fields.")
                st.write("i. Binary attributes: Computes true positives, false positives, false negatives.")
                st.write("ii. Continuous attributes: Checks if predictions are within ±10% of actual values.")
                st.write("4. Compute the weighted F1-Score")
                st.write("This attack is repeated for all rows in the real dataset, and the score is weighted performance in predicting the sensitive column. The score is an overall probability of guessing the sensitive column correctly.")
                st.write("**For your data**")
                st.write("The key fields are:")
                key_fields = ['First Name', 'Last Name', 'Height', 'Nationality', 'Favorite Icecream', 'Times Been to Italy', 'First Time London', 'Steps per Day']
                ind_vals = st.session_state.real_data[key_fields].iloc[[len(st.session_state.real_data)-1]]
                st.dataframe(ind_vals, use_container_width=True, hide_index=True)
                syndat=st.session_state.syn_data_bin
                dummy_real_cat, dummy_syn_cat = get_dummy_datasets(st.session_state.real_data[key_fields], syndat[key_fields])
                dummy_real, dummy_syn = get_dummy_datasets(st.session_state.real_data, syndat)
                dummy_ind_vals = dummy_real_cat[len(st.session_state.real_data)-1]
                idx = air_nn(dummy_ind_vals, dummy_syn_cat, k=1)
                idx2 = air_nn(dummy_ind_vals, dummy_syn_cat, k=2)[1]
                st.write("The key Fields of nearest synthetic neighbour(s) using a normalized Hamming distance is:")
                st.dataframe(syndat[key_fields].iloc[idx], use_container_width=True, hide_index=True)
                st.write("These are the sensitive fields for both individuals:")
                dummy_real_indv, dummy_syn_indv = get_dummy_datasets(st.session_state.real_data['Like Liquorice'], syndat['Like Liquorice'])
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    st.dataframe(pd.DataFrame({'Your Sensitive Field': st.session_state.real_data['Like Liquorice'].iloc[[len(st.session_state.real_data)-1]]}), use_container_width=True, hide_index=True)
                with col1_2:
                    st.dataframe(pd.DataFrame({"Neighbour's Sensitive Field": syndat['Like Liquorice'].iloc[idx]}), use_container_width=True, hide_index=True)
                col1_1_1, col1_2_1 = st.columns(2)
                with col1_1_1:
                    st.write("(One-Hot encoded)")
                    st.write(dummy_real_indv[[len(st.session_state.real_data)-1]])
                    
                with col1_2_1:
                    st.write("(One-Hot encoded)")
                    st.write(dummy_syn_indv[idx])
                    
                real_label = np.array(dummy_real_indv[[len(st.session_state.real_data)-1]])
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
                    st.write("You and your neighbour have matching sensitive fields!")
                    st.write("Your contribution to the score is therefore:")
                else:
                    st.write("You and your neighbour do not have matching sensitive fields.")
                    st.write("Your contribution to the score is therefore:")
                st.latex(r"F_1 = \frac{2*"rf"{precision}"r"*"rf"{recall}"r"}{"rf"{precision}"r"+"rf"{recall}"r"} = "rf"{f_one}")
                st.latex(r"weight = \frac{"rf"{abs(round(numerator[len(st.session_state.real_data)-1], 2))}"r"}{"rf"{abs(round(denominator, 2))}"r"} = {"rf"{round(prob_df['Weight'].iloc[len(st.session_state.real_data)-1], 2)}"r"}")
                st.latex(rf'''AIR = {round((2*precision*recall) / (1+recall), 2)}*{round(prob_df['Weight'].iloc[len(st.session_state.real_data)-1], 2)}= {round(f_one*(prob_df['Weight'].iloc[len(st.session_state.real_data)-1]), 2)}''')
            with col2:
                st.write("Real Dataset:")
                st.dataframe(st.session_state.real_data, use_container_width=True, hide_index=True)
                st.write("Your Synthetic Dataset:")
                st.dataframe(st.session_state.syn_data_bin, use_container_width=True, hide_index=True)
                st.write("Real Dataset (One-Hot Encoded):")
                st.write(dummy_real)
                st.write("Your Synthetic Dataset (One-Hot Encoded):")
                st.write(dummy_syn)
            st.write("**The problems that may occur:**")
            st.write(" - There are continuous attributes in either the key fields. Therefore, finding a neighbour is influenced differently for these. This may be vissible from the key fields of your 2nd nearest neighbour:")
            st.dataframe(syndat[key_fields].iloc[[idx2]], use_container_width=True, hide_index=True)
            
    with dcr_tab:
        st.subheader("Distance to Closest Record (DCR):")
        
    with nndr_tab:
        st.subheader("Nearest Neighbour Distance Ratio (NNDR):")
        
    with hidd_tab:
        st.subheader("Hidden Rate:")

    
st.button(label="Start Over", on_click=set_state, args=[0])