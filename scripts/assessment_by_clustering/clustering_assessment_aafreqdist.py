import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import entropy
from sklearn.cluster import KMeans


def read_sequence_set(file_path):
    return pd.read_csv(file_path, sep="\t")["junction_aa"]


# Define standard amino acids
amino_acids = "ACDEFGHIKLMNPQRSTVWY"


def compute_aa_frequencies(sequences):
    """
    Compute the amino acid frequency distribution for a set of sequences.
    """
    freq_dict = {aa: 0 for aa in amino_acids}
    total_count = 0

    for seq in sequences:
        for aa in seq:
            freq_dict[aa] += 1
            total_count += 1

    frequencies = np.array([freq_dict[aa] / total_count for aa in amino_acids])
    return frequencies


def compute_js_divergence(sequence_set1, sequence_set2):
    """
    Compute Jensen-Shannon divergence between two probability distributions.
    """
    p = compute_aa_frequencies(sequence_set1)
    q = compute_aa_frequencies(sequence_set2)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def plot_js_divergences_heatmap(js_divergence_matrix):
    sns.clustermap(js_divergence_matrix, annot=True, method='average', metric='euclidean')
    plt.title("Jensen-Shannon Divergence")
    plt.tight_layout()
    plt.savefig('JS_divergence_epi.png')


def plot_js_divergences_hierarchical_clustering(js_divergence_df):
    row_linkage = linkage(pdist(js_divergence_df, metric="euclidean"), method="average")
    col_linkage = linkage(pdist(js_divergence_df.T, metric="euclidean"), method="average")
    row_order = dendrogram(row_linkage, no_plot=True)["leaves"]
    col_order = dendrogram(col_linkage, no_plot=True)["leaves"]

    df_clustered = js_divergence_df.iloc[row_order, col_order]

    heatmap = go.Heatmap(
        z=df_clustered.values,
        x=df_clustered.columns,
        y=df_clustered.index,
        colorscale="Viridis"
    )

    # Plot
    fig = go.Figure(data=[heatmap])
    fig.update_layout(title="Clustered Heatmap", xaxis_title="Columns", yaxis_title="Rows")
    #fig.show()
    fig.write_html("JS_divergence_hierarchical_clustering_epi.html")


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

    #fig.show()
    fig.write_html("JS_divergence_kmeans_clustering_epi.html")


def run_sequence_overlap_analysis(generated1, generated2):
    sequence_set1 = read_sequence_set(generated1)
    sequence_set2 = read_sequence_set(generated2)

    js_divergence = compute_js_divergence(sequence_set1, sequence_set2)
    return js_divergence


def main():
    # test on phenotype data
    # all_datasets = ['../results_experiments/6323_Spleen_CD8/data_immuneml_format_with_junction/datasets/dataset/dataset.tsv',
    #             '../results_experiments/6323_Spleen_CD8/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_6323_Spleen_CD8/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
    #             '../results_experiments/6323_Spleen_Tconv/data_immuneml_format_with_junction/datasets/dataset/dataset.tsv',
    #              '../results_experiments/6323_Spleen_Tconv/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_6323_Spleen_Tconv/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
    #             '../results_experiments/6323_Spleen_Treg/data_immuneml_format_with_junction/datasets/dataset/dataset.tsv',
    #              '../results_experiments/6323_Spleen_Treg/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_6323_Spleen_Treg/gen_model/exported_gen_dataset/SoNNiaDataset.tsv']
    #
    # all_datasets_names = ['Spleen_CD8_train', 'Spleen_CD8_SoNNia', 'Spleen_Tconv_train', 'Spleen_Tconv_SoNNia', 'Spleen_Treg_train', 'Spleen_Treg_SoNNia']

    # all_datasets = [
    #     '../../results_experiments/6323_intra-islet_CD4/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_6323_intra-islet_CD4/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
    #     '../../results_experiments/6323_intra-islet_CD8/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_6323_intra-islet_CD8/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
    #     '../../results_experiments/6323_SPL_B_cell/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_6323_SPL_B_cell/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
    #     '../../results_experiments/6323_Spleen_CD8/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_6323_Spleen_CD8/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
    #     '../../results_experiments/6323_Spleen_Tconv/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_6323_Spleen_Tconv/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
    #     '../../results_experiments/6323_Spleen_Treg/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_6323_Spleen_Treg/gen_model/exported_gen_dataset/SoNNiaDataset.tsv']
    #
    # all_datasets_names = ['intra-islet_CD4_SoNNia', 'intra-islet_CD8_SoNNia', 'SPL_B_cell_SoNNia','Spleen_CD8_SoNNia', 'Spleen_Tconv_SoNNia',
    #                       'Spleen_Treg_SoNNia']

    # Test on epitope-specific data
    all_datasets = ['../../results_experiments/IEDB_AVFDRKSDAK_human_TCR_filtered_sample1/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_AVFDRKSDAK_human_TCR_filtered_sample1/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                    '../../results_experiments/IEDB_AVFDRKSDAK_human_TCR_filtered_sample2/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_AVFDRKSDAK_human_TCR_filtered_sample2/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                    '../../results_experiments/IEDB_AVFDRKSDAK_human_TCR_filtered_sample3/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_AVFDRKSDAK_human_TCR_filtered_sample3/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                    '../../results_experiments/IEDB_GILGFVFTL_human_TCR_filtered_sample1/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GILGFVFTL_human_TCR_filtered_sample1/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                    '../../results_experiments/IEDB_GILGFVFTL_human_TCR_filtered_sample2/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GILGFVFTL_human_TCR_filtered_sample2/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                    '../../results_experiments/IEDB_GILGFVFTL_human_TCR_filtered_sample3/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GILGFVFTL_human_TCR_filtered_sample3/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                    '../../results_experiments/IEDB_GLCTLVAML_human_TCR_filtered_sample1/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GLCTLVAML_human_TCR_filtered_sample1/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                    '../../results_experiments/IEDB_GLCTLVAML_human_TCR_filtered_sample2/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GLCTLVAML_human_TCR_filtered_sample2/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                    '../../results_experiments/IEDB_GLCTLVAML_human_TCR_filtered_sample3/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GLCTLVAML_human_TCR_filtered_sample3/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',

                    ]
    all_datasets_names = ['AVFDRKSDAK_1', 'AVFDRKSDAK_2', 'AVFDRKSDAK_3', 'GILGFVFTL_1', 'GILGFVFTL_2', 'GILGFVFTL_3', 'GLCTLVAML_1', 'GLCTLVAML_2', 'GLCTLVAML_3']

    js_divergences_matrix = []
    for i in range(len(all_datasets)):
        js_divergences = []
        for j in range(len(all_datasets)):
            generated1 = all_datasets[i]
            generated2 = all_datasets[j]

            js_divergence = run_sequence_overlap_analysis(generated1, generated2)
            js_divergences.append(js_divergence)
        js_divergences_matrix.append(js_divergences)

    # Convert matrix to dataframe with all_datasets_names as index and columns
    js_divergences_matrix = pd.DataFrame(js_divergences_matrix, index=all_datasets_names, columns=all_datasets_names)

    plot_js_divergences_heatmap(js_divergences_matrix)
    plot_js_divergences_hierarchical_clustering(js_divergences_matrix)
    plot_js_divergences_kmeans_clustering(js_divergences_matrix, n_clusters=3)


if __name__ == "__main__":
    main()
