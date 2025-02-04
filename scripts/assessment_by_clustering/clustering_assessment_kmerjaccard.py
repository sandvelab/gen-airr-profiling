import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import entropy
from sklearn.cluster import KMeans


def read_sequence_set(file_path):
    return pd.read_csv(file_path, sep="\t")["junction_aa"]


def generate_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def sequences_to_kmer_sets(sequences, k=3):
    kmer_sets = sequences.apply(lambda seq: set(generate_kmers(seq, k)))
    return kmer_sets


def compute_jaccard_similarity(sequence_set1, sequence_set2):
    kmers1 = sequences_to_kmer_sets(sequence_set1)
    kmers2 = sequences_to_kmer_sets(sequence_set2)
    union1 = set().union(*kmers1)
    union2 = set().union(*kmers2)
    union = set().union(*kmers1).union(*kmers2)
    intersection = union1.intersection(union2)
    return len(intersection) / len(union) if len(union) > 0 else 0.0


def plot_jaccard_similarity_heatmap(jaccard_similarities_matrix):
    sns.clustermap(jaccard_similarities_matrix, annot=True, method='average', metric='euclidean')
    plt.title("Kmer Jaccard similarity")
    plt.tight_layout()
    plt.show()


def plot_js_divergences_kmeans_clustering(js_divergence_df, n_clusters=3):
    row_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    row_clusters = row_kmeans.fit_predict(js_divergence_df)

    # Apply K-means clustering on columns (transpose for clustering)
    col_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    col_clusters = col_kmeans.fit_predict(js_divergence_df.T)

    # Get sorted order based on cluster labels
    row_order = np.argsort(row_clusters)
    col_order = np.argsort(col_clusters)

    # Reorder data
    df_clustered = js_divergence_df.iloc[row_order, col_order]

    # Rename rows and columns to include cluster ID
    df_clustered.index = [f"{name} (C{row_clusters[i] + 1})" for i, name in zip(row_order, js_divergence_df.index)]
    df_clustered.columns = [f"{name} (C{col_clusters[i] + 1})" for i, name in zip(col_order, js_divergence_df.columns)]

    # Create the heatmap
    heatmap = go.Heatmap(
        z=df_clustered.values,
        x=df_clustered.columns,
        y=df_clustered.index,
        colorscale="Viridis",
        showscale=True
    )

    # Plot
    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title="K-Means Clustered Heatmap (Cluster Labels in Names)",
        xaxis_title="Columns (Clusters)",
        yaxis_title="Rows (Clusters)",
        xaxis_tickangle=-45
    )

    fig.show()


def run_sequence_overlap_analysis(generated1, generated2):
    sequence_set1 = read_sequence_set(generated1)
    sequence_set2 = read_sequence_set(generated2)

    jaccard_similarity = compute_jaccard_similarity(sequence_set1, sequence_set2)
    return jaccard_similarity


def main():
    # Test on epitope-specific data
    all_datasets = [
        '../../results_experiments/IEDB_AVFDRKSDAK_human_TCR_filtered_sample1/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_AVFDRKSDAK_human_TCR_filtered_sample1/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
        '../../results_experiments/IEDB_AVFDRKSDAK_human_TCR_filtered_sample2/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_AVFDRKSDAK_human_TCR_filtered_sample2/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
        '../../results_experiments/IEDB_AVFDRKSDAK_human_TCR_filtered_sample3/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_AVFDRKSDAK_human_TCR_filtered_sample3/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
        '../../results_experiments/IEDB_GILGFVFTL_human_TCR_filtered_sample1/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GILGFVFTL_human_TCR_filtered_sample1/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
        '../../results_experiments/IEDB_GILGFVFTL_human_TCR_filtered_sample2/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GILGFVFTL_human_TCR_filtered_sample2/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
        '../../results_experiments/IEDB_GILGFVFTL_human_TCR_filtered_sample3/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GILGFVFTL_human_TCR_filtered_sample3/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
        '../../results_experiments/IEDB_GLCTLVAML_human_TCR_filtered_sample1/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GLCTLVAML_human_TCR_filtered_sample1/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
        '../../results_experiments/IEDB_GLCTLVAML_human_TCR_filtered_sample2/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GLCTLVAML_human_TCR_filtered_sample2/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
        '../../results_experiments/IEDB_GLCTLVAML_human_TCR_filtered_sample3/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GLCTLVAML_human_TCR_filtered_sample3/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',

        ]
    all_datasets_names = ['AVFDRKSDAK_1', 'AVFDRKSDAK_2', 'AVFDRKSDAK_3', 'GILGFVFTL_1', 'GILGFVFTL_2', 'GILGFVFTL_3',
                          'GLCTLVAML_1', 'GLCTLVAML_2', 'GLCTLVAML_3']

    jaccard_similarities_matrix = []
    for i in range(len(all_datasets)):
        jaccard_similarities = []
        for j in range(len(all_datasets)):
            generated1 = all_datasets[i]
            generated2 = all_datasets[j]

            jaccard_similarity = run_sequence_overlap_analysis(generated1, generated2)
            jaccard_similarities.append(jaccard_similarity)
        jaccard_similarities_matrix.append(jaccard_similarities)

    jaccard_similarities_matrix = pd.DataFrame(jaccard_similarities_matrix, index=all_datasets_names, columns=all_datasets_names)
    plot_jaccard_similarity_heatmap(jaccard_similarities_matrix)

    plot_js_divergences_kmeans_clustering(jaccard_similarities_matrix, n_clusters=3)


if __name__ == "__main__":
    main()
