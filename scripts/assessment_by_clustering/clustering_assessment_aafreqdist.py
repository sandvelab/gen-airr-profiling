import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from scipy.stats import entropy



def read_sequence_set(file_path):
    return pd.read_csv(file_path, sep="\t")["junction_aa"]


# Define standard amino acids
amino_acids = "ACDEFGHIKLMNPQRSTVWY"


def compute_aa_frequencies(sequences):
    """
    Compute the normalized amino acid frequency distribution for a set of sequences.
    """
    freq_dict = {aa: 0 for aa in amino_acids}
    total_count = 0

    for seq in sequences:
        for aa in seq:
            if aa in freq_dict:
                freq_dict[aa] += 1
                total_count += 1

    # Normalize frequencies
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
    plt.savefig('JS_divergence.png')


def run_sequence_overlap_analysis(generated1, generated2):
    sequence_set1 = read_sequence_set(generated1)
    sequence_set2 = read_sequence_set(generated2)

    js_divergence = compute_js_divergence(sequence_set1, sequence_set2)
    return js_divergence


def main():
    # all_datasets = ['../results_experiments/6323_Spleen_CD8/data_immuneml_format_with_junction/datasets/dataset/dataset.tsv',
    #             '../results_experiments/6323_Spleen_CD8/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_6323_Spleen_CD8/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
    #             '../results_experiments/6323_Spleen_Tconv/data_immuneml_format_with_junction/datasets/dataset/dataset.tsv',
    #              '../results_experiments/6323_Spleen_Tconv/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_6323_Spleen_Tconv/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
    #             '../results_experiments/6323_Spleen_Treg/data_immuneml_format_with_junction/datasets/dataset/dataset.tsv',
    #              '../results_experiments/6323_Spleen_Treg/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_6323_Spleen_Treg/gen_model/exported_gen_dataset/SoNNiaDataset.tsv']
    #
    # all_datasets_names = ['Spleen_CD8_train', 'Spleen_CD8_SoNNia', 'Spleen_Tconv_train', 'Spleen_Tconv_SoNNia', 'Spleen_Treg_train', 'Spleen_Treg_SoNNia']

    # Test on epitope-specific data
    all_datasets = ['../results_experiments/IEDB_AVFDRKSDAK_human_TCR_filtered_sample1/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_AVFDRKSDAK_human_TCR_filtered_sample1/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                    '../results_experiments/IEDB_AVFDRKSDAK_human_TCR_filtered_sample2/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_AVFDRKSDAK_human_TCR_filtered_sample2/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                    '../results_experiments/IEDB_AVFDRKSDAK_human_TCR_filtered_sample3/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_AVFDRKSDAK_human_TCR_filtered_sample3/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                    '../results_experiments/IEDB_GILGFVFTL_human_TCR_filtered_sample1/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GILGFVFTL_human_TCR_filtered_sample1/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                    '../results_experiments/IEDB_GILGFVFTL_human_TCR_filtered_sample2/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GILGFVFTL_human_TCR_filtered_sample2/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                    '../results_experiments/IEDB_GILGFVFTL_human_TCR_filtered_sample3/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GILGFVFTL_human_TCR_filtered_sample3/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                    '../results_experiments/IEDB_GLCTLVAML_human_TCR_filtered_sample1/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GLCTLVAML_human_TCR_filtered_sample1/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                    '../results_experiments/IEDB_GLCTLVAML_human_TCR_filtered_sample2/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GLCTLVAML_human_TCR_filtered_sample2/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                    '../results_experiments/IEDB_GLCTLVAML_human_TCR_filtered_sample3/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GLCTLVAML_human_TCR_filtered_sample3/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',

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


if __name__ == "__main__":
    main()
