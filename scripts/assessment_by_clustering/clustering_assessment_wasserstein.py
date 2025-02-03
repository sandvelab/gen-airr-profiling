import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
from Bio.Align import substitution_matrices
import matplotlib.pyplot as plt


blosum62 = substitution_matrices.load('BLOSUM62')

def read_sequence_set(file_path):
    return pd.read_csv(file_path, sep="\t")["junction_aa"]


def blosum62_score(sequence):
    """
    Compute the sum of BLOSUM62 substitution scores for adjacent amino acids in a sequence.
    If a pair is not found in the matrix, assign a default low score (-4).
    """
    score = 0
    for i in range(len(sequence) - 1):  # Use adjacent amino acids for scoring
        pair = (sequence[i], sequence[i+1])
        rev_pair = (sequence[i+1], sequence[i])
        score += blosum62.get(pair, blosum62.get(rev_pair, -4))  # Default score if not found
    return score


def compute_sequence_scores(sequence_set):
    """
    Compute a list of BLOSUM62 scores for a set of sequences.
    """
    return np.array([blosum62_score(seq) for seq in sequence_set])


def compute_wasserstein_distance(sequence_set1, sequence_set2):
    """
    Compute Wasserstein distance between two sets of sequence scores.
    """
    scores1 = compute_sequence_scores(sequence_set1)
    scores2 = compute_sequence_scores(sequence_set2)

    return stats.wasserstein_distance(scores1, scores2)


def plot_wasserstein_distance_heatmap(wasserstein_distance_matrix):
    sns.clustermap(wasserstein_distance_matrix, annot=True)
    plt.title("Wasserstein distance")
    plt.tight_layout()
    plt.savefig('wasserstein_distance.png')


def run_sequence_overlap_analysis(generated1, generated2):
    sequence_set1 = read_sequence_set(generated1)
    sequence_set2 = read_sequence_set(generated2)

    wasserstein_distance = compute_wasserstein_distance(sequence_set1, sequence_set2)
    return wasserstein_distance


def main():
    all_datasets = ['../results_experiments/6323_Spleen_CD8/data_immuneml_format_with_junction/datasets/dataset/dataset.tsv',
                '../results_experiments/6323_Spleen_CD8/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_6323_Spleen_CD8/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                '../results_experiments/6323_Spleen_Tconv/data_immuneml_format_with_junction/datasets/dataset/dataset.tsv',
                 '../results_experiments/6323_Spleen_Tconv/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_6323_Spleen_Tconv/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
                '../results_experiments/6323_Spleen_Treg/data_immuneml_format_with_junction/datasets/dataset/dataset.tsv',
                 '../results_experiments/6323_Spleen_Treg/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_6323_Spleen_Treg/gen_model/exported_gen_dataset/SoNNiaDataset.tsv']

    all_datasets_names = ['Spleen_CD8_train', 'Spleen_CD8_SoNNia', 'Spleen_Tconv_train', 'Spleen_Tconv_SoNNia', 'Spleen_Treg_train', 'Spleen_Treg_SoNNia']

    wasserstein_distances_matrix = []
    for i in range(len(all_datasets)):
        wasserstein_distances = []
        for j in range(len(all_datasets)):
            generated1 = all_datasets[i]
            generated2 = all_datasets[j]

            wasserstein_distance = run_sequence_overlap_analysis(generated1, generated2)
            wasserstein_distances.append(wasserstein_distance)
        wasserstein_distances_matrix.append(wasserstein_distances)

    # Convert matrix to dataframe with all_datasets_names as index and columns
    wasserstein_distances_matrix = pd.DataFrame(wasserstein_distances_matrix, index=all_datasets_names, columns=all_datasets_names)
    plot_wasserstein_distance_heatmap(wasserstein_distances_matrix)


if __name__ == "__main__":
    main()
