import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_sequence_set(file_path):
    return pd.read_csv(file_path, sep="\t")["junction_aa"]


def generate_kmers(sequence, k=4):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def sequences_to_kmer_sets(sequences, k=4):
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


def run_sequence_overlap_analysis(generated1, generated2):
    sequence_set1 = read_sequence_set(generated1)
    sequence_set2 = read_sequence_set(generated2)

    jaccard_similarity = compute_jaccard_similarity(sequence_set1, sequence_set2)
    return jaccard_similarity


def main():
    # Test on epitope-specific data
    all_datasets = [
        '../results_experiments/IEDB_AVFDRKSDAK_human_TCR_filtered_sample1/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_AVFDRKSDAK_human_TCR_filtered_sample1/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
        '../results_experiments/IEDB_AVFDRKSDAK_human_TCR_filtered_sample2/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_AVFDRKSDAK_human_TCR_filtered_sample2/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
        '../results_experiments/IEDB_AVFDRKSDAK_human_TCR_filtered_sample3/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_AVFDRKSDAK_human_TCR_filtered_sample3/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
        '../results_experiments/IEDB_GILGFVFTL_human_TCR_filtered_sample1/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GILGFVFTL_human_TCR_filtered_sample1/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
        '../results_experiments/IEDB_GILGFVFTL_human_TCR_filtered_sample2/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GILGFVFTL_human_TCR_filtered_sample2/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
        '../results_experiments/IEDB_GILGFVFTL_human_TCR_filtered_sample3/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GILGFVFTL_human_TCR_filtered_sample3/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
        '../results_experiments/IEDB_GLCTLVAML_human_TCR_filtered_sample1/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GLCTLVAML_human_TCR_filtered_sample1/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
        '../results_experiments/IEDB_GLCTLVAML_human_TCR_filtered_sample2/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GLCTLVAML_human_TCR_filtered_sample2/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',
        '../results_experiments/IEDB_GLCTLVAML_human_TCR_filtered_sample3/models/soNNia_ngs50000_epoch50/soNNia_ngs50000_epoch50_IEDB_GLCTLVAML_human_TCR_filtered_sample3/gen_model/exported_gen_dataset/SoNNiaDataset.tsv',

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


if __name__ == "__main__":
    main()
