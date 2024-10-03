import argparse
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import entropy
import numpy as np


def load_csv(file_path):
    # Load the TSV file into a DataFrame
    return pd.read_csv(file_path)

def kl_seq_len_compare(file1, file2):
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)

    # Make sure both distributions have the same support (i.e., same sequence lengths)
    # Merge the two datasets to align the sequence lengths
    merged_data = pd.merge(data1, data2, on='sequence_lengths', how='outer').fillna(0)

    # Recompute the normalized counts after merging
    p = merged_data['counts_x'] / merged_data['counts_x'].sum()
    q = merged_data['counts_y'] / merged_data['counts_y'].sum()

    # Avoid division by zero by adding a small value
    q = np.where(q == 0, 1e-100, q)

    # Compute KL Divergence
    kl_divergence = entropy(p, q)
    return kl_divergence

# Function to read sequence lengths from file
def read_sequence_lengths(file):
    data = pd.read_csv(file)
    return set(data['sequence_lengths'])

# Compute Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def get_error_bars(data):
    df = pd.read_csv(data)


def plot_seq_len_distribution(file1, file2, image_file):
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)

    df_combine = {"file1": data1, "file2": data2}
    df_combine = pd.concat(df_combine, names=["dataset"])
    df_combine = df_combine.reset_index(level=0)

    figure = px.bar(df_combine, x="sequence_lengths", y="counts", color='dataset')
    figure.update_layout(barmode='group', xaxis=dict(tickmode='array', tickvals=df_combine["sequence_lengths"]),
                         yaxis=dict(tickmode='array', tickvals=df_combine["counts"]),
                         template="plotly_white", title="Sequence Length Distributions")
    #figure.show()
    #figure.write_html(image_file)


def main():
    parser = argparse.ArgumentParser(description='Compute KL Divergence between two sequence length distribution CSV files.')
    parser.add_argument('file1', type=str, help='Path to the first CSV file.')
    parser.add_argument('file2', type=str, help='Path to the second CSV file.')
    parser.add_argument('output_file', type=str, default='.', help='Output directory for the results.')

    args = parser.parse_args()

    # Compute KL divergence
    kl_divergence = kl_seq_len_compare(args.file1, args.file2)

    # Compute Jaccard similarity
    seq_lens1 = read_sequence_lengths(args.file1)
    seq_lens2 = read_sequence_lengths(args.file2)
    jaccard_sim = jaccard_similarity(seq_lens1, seq_lens2)
    #print(f"Jaccard similarity for {args.file1}, {args.file2}: {jaccard_sim}")

    # Plot the sequence length distribution
    plot_seq_len_distribution(args.file1, args.file2, "seq_len_dist.html")

    # Output the results
    with open(args.output_file, 'w') as f:
        f.write(str(kl_divergence))


if __name__ == "__main__":
    main()
