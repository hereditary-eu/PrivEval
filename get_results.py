from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.manifold import TSNE
from get_metric_results import get_metric_results

df = pd.read_csv("Data/real.csv", index_col=False)
syn_df = pd.read_csv("Data/tabsyn.csv", index_col=False)

all_data = pd.concat([df, syn_df], ignore_index=True)

cat_cols = all_data.select_dtypes(include=['object', 'bool']).columns

# Initialize a dictionary to hold encoded data
encoded_data = {}
for col in cat_cols:
    if all_data[col].dtype == 'bool':
        encoded_data[col] = all_data[col].astype(int)
    else:
        le = LabelEncoder()
        encoded_data[col] = le.fit_transform(all_data[col].astype(str))

num_cols = all_data.select_dtypes(exclude=['object', 'bool']).columns
for col in num_cols:
    encoded_data[col] = all_data[col]

all_labels = pd.DataFrame(encoded_data)
real_len = len(df)
real_labels = all_labels[:real_len]
syn_labels = all_labels[real_len:]

metric_results = get_metric_results(df, syn_df, real_labels, syn_labels, sensitive_attributes=['Revenue'])
print("Metric Results:")
print(metric_results)

metric_results.to_csv("metric_results/tabsyn_metric_results.csv", index=False)