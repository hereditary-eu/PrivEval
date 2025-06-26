from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from get_metric_results import get_metric_results

df = pd.read_csv("Data/real.csv", index_col=False)
syn_df = pd.read_csv("Data/tabsyn.csv", index_col=False)

# df = df.sample(100, random_state=42, ignore_index=True)
# syn_df = syn_df.sample(100, random_state=42, ignore_index=True)

# Combine real and synthetic data
all_data = pd.concat([df, syn_df], ignore_index=True)

# Identify categorical columns
cat_cols = all_data.select_dtypes(include=['object', 'bool']).columns

# Initialize a dictionary to hold encoded data
encoded_data = {}
# Encode categorical columns
for col in cat_cols:
    if all_data[col].dtype == 'bool':
        # Directly convert bool to int 0/1
        encoded_data[col] = all_data[col].astype(int)
    else:
        le = LabelEncoder()
        encoded_data[col] = le.fit_transform(all_data[col].astype(str))

# Keep numeric columns as-is
num_cols = all_data.select_dtypes(exclude=['object', 'bool']).columns
for col in num_cols:
    encoded_data[col] = all_data[col]

# Create final DataFrame
all_labels = pd.DataFrame(encoded_data)
# Split real and synthetic portions
real_len = len(df)
real_labels = all_labels[:real_len]
syn_labels = all_labels[real_len:]

# Run t-SNE
tsne = TSNE(n_components=2)
real_coords_tsne = tsne.fit_transform(real_labels)
syn_coords_tsne = tsne.fit_transform(syn_labels)



# Store TSNE results
tsne_df_real = pd.DataFrame(real_coords_tsne)
tsne_df_syn = pd.DataFrame(syn_coords_tsne)

metric_results = get_metric_results(df, syn_df, real_labels, syn_labels, sensitive_attributes=['Revenue'])
print("Metric Results:")
print(metric_results)

metric_results.to_csv("Data/metric_results.csv", index=False)
