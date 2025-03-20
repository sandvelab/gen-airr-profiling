import os

import pandas as pd
#TODO: We need to find more elegant solution for setting the backend
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.compairr_utils import process_and_save_sequences, run_compairr


def run_phenotype_analysis(analysis_config: AnalysisConfig):
    print(f"Running phenotype analysis for {analysis_config}")
    if len(analysis_config.model_names) != 1:
        raise ValueError("Phenotype analysis only supports one model")
    model_name = analysis_config.model_names[0]
    compairr_sequences_dir = f"{analysis_config.root_output_dir}/generated_compairr_sequences/{model_name}"

    similarities_matrix, dataset_names = calculate_similarities_matrix(compairr_sequences_dir,
                                                                       analysis_config.analysis_output_dir,
                                                                       model_name)

    similarities_df = pd.DataFrame(similarities_matrix, index=dataset_names, columns=dataset_names)
    plot_cluster_heatmap(analysis_config.analysis_output_dir, similarities_df, model_name)


# TODO: There's a lot of file operations here, we should consider refactoring this function a bit more
def calculate_similarities_matrix(sequences_dir, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)

    generated_datasets = os.listdir(sequences_dir)
    generated_datasets_names = [dataset.strip('.tsv') for dataset in generated_datasets]

    similarities_matrix = []

    for dataset1 in generated_datasets:
        similarities = []
        for dataset2 in generated_datasets:
            dataset1_path, dataset2_path = f"{sequences_dir}/{dataset1}", f"{sequences_dir}/{dataset2}"
            dataset1_name, dataset2_name = dataset1.strip('.tsv'), dataset2.strip('.tsv')

            helper_dir = f"{output_dir}/compairr_helper_files"
            os.makedirs(helper_dir, exist_ok=True)
            file_name = f"{dataset1_name}_{dataset2_name}"
            unique_sequences_path = f"{helper_dir}/{file_name}_unique.tsv"
            concat_sequences_path = f"{helper_dir}/{file_name}_concat.tsv"
            process_and_save_sequences(dataset1_path, dataset2_path, unique_sequences_path, concat_sequences_path)

            compairr_output_dir = f"{output_dir}/compairr_output"
            run_compairr(compairr_output_dir, unique_sequences_path, concat_sequences_path, file_name, model_name)

            overlap_df = pd.read_csv(f"{compairr_output_dir}/{file_name}_overlap.tsv", sep='\t')
            n_nonzero_rows = overlap_df[(overlap_df['dataset_1'] != 0) & (overlap_df['dataset_2'] != 0)].shape[0]

            union = pd.read_csv(unique_sequences_path, sep='\t').shape[0]
            jaccard_similarity = n_nonzero_rows / union
            similarities.append(jaccard_similarity)

        similarities_matrix.append(similarities)

    return similarities_matrix, generated_datasets_names


# def plot_cluster_heatmap(output_dir, similarities_matrix, model_name):
#     sns.clustermap(similarities_matrix, annot=True, method='average', metric='euclidean')
#     plt.title(f"Jaccard similarity: {model_name}")
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/cluster_heatmap.png")


def plot_cluster_heatmap(output_dir, similarities_matrix, model_name):
    g = sns.clustermap(similarities_matrix, annot=True, method='average', metric='euclidean')
    g.fig.suptitle(f"Jaccard Similarity: {model_name}", fontsize=14, fontweight='bold', y=1.0)
    #g.fig.suptitle(f"Jaccard similarity: {model_name}")
    g.savefig(f"{output_dir}/cluster_heatmap.png")
    plt.close(g.fig)
