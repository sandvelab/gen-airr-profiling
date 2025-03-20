import os

import pandas as pd
#TODO: We need to find more elegant solution for setting the backend
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from gen_airr_bm.core.analysis_config import AnalysisConfig


def run_phenotype_analysis(analysis_config: AnalysisConfig):
    print(f"Running phenotype analysis for {analysis_config}")
    if len(analysis_config.model_names) != 1:
        raise ValueError("Phenotype analysis only supports one model")
    model_name = analysis_config.model_names[0]
    generated_sequences_dir = f"{analysis_config.root_output_dir}/generated_sequences/{model_name}"
    compairr_sequences_dir = f"{analysis_config.root_output_dir}/generated_compairr_sequences/{model_name}"
    preprocess_files_for_compairr(generated_sequences_dir, compairr_sequences_dir)
    similarities_matrix, dataset_names = calculate_similarities_matrix(compairr_sequences_dir,
                                                                       analysis_config.analysis_output_dir,
                                                                       model_name)

    similarities_df = pd.DataFrame(similarities_matrix, index=dataset_names, columns=dataset_names)
    plot_cluster_heatmap(analysis_config.analysis_output_dir, similarities_df, model_name)


def preprocess_files_for_compairr(generated_sequences_dir, compairr_sequences_dir):
    generated_datasets = os.listdir(generated_sequences_dir)
    os.makedirs(f"{compairr_sequences_dir}", exist_ok=True)
    for dataset in generated_datasets:
        data = pd.read_csv(f"{generated_sequences_dir}/{dataset}", sep='\t')
        data.replace({'duplicate_count': {-1: 1}}, inplace=True)
        data.to_csv(f"{compairr_sequences_dir}/{dataset}", sep='\t', index=False)


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

            analyses_sequences_dir = f"{output_dir}/generated_sequences"
            os.makedirs(analyses_sequences_dir, exist_ok=True)
            file_name = f"{dataset1_name}_{dataset2_name}"
            unique_sequences_path = f"{analyses_sequences_dir}/{file_name}_unique.tsv"
            concat_sequences_path = f"{analyses_sequences_dir}/{file_name}.tsv"
            process_and_save_sequences(dataset1_path, dataset2_path, unique_sequences_path, concat_sequences_path)

            compairr_output_dir = f"{output_dir}/compairr"
            run_compairr(compairr_output_dir, unique_sequences_path, concat_sequences_path, file_name, model_name)

            overlap_df = pd.read_csv(f"{compairr_output_dir}/{file_name}_overlap.tsv", sep='\t')
            n_nonzero_rows = overlap_df[(overlap_df['gen_model_1'] != 0) & (overlap_df['gen_model_2'] != 0)].shape[0]

            union = pd.read_csv(unique_sequences_path, sep='\t').shape[0]
            jaccard_similarity = n_nonzero_rows / union
            similarities.append(jaccard_similarity)

        similarities_matrix.append(similarities)

    return similarities_matrix, generated_datasets_names


def run_compairr(compairr_output_dir, unique_sequences_path, concat_sequences_path, file_name, model_name):
    os.makedirs(compairr_output_dir, exist_ok=True)
    #TODO: For ImmunoHub execution we might need to use binaries instead of the command line
    compairr_command = (f"./compairr-1.13.0-linux-x86_64 -x {unique_sequences_path} {concat_sequences_path} -d 1 -f -t 8 -o "
                        f"{compairr_output_dir}/{file_name}_overlap.tsv -p {compairr_output_dir}/{file_name}_pairs.tsv "
                        f"--log {compairr_output_dir}/{file_name}_log.txt --indels")

    # TODO: Add better support for PWM model
    if model_name == "pwm":
        compairr_command += " -g"
    os.system(compairr_command)


def process_and_save_sequences(generated1_path, generated2_path, output_file_unique, output_file_concat):
    generated1_data = pd.read_csv(generated1_path, sep='\t')
    generated2_data = pd.read_csv(generated2_path, sep='\t')
    generated1_data['sequence_id'] = [f"gen_model_1_{i + 1}" for i in range(len(generated1_data))]
    generated2_data['sequence_id'] = [f"gen_model_2_{i + 1}" for i in range(len(generated2_data))]

    unique_sequences = pd.concat([generated1_data, generated2_data]).drop_duplicates(subset=['junction_aa'])
    unique_sequences.to_csv(output_file_unique, sep='\t', index=False)

    generated1_data['repertoire_id'] = "gen_model_1"
    generated2_data['repertoire_id'] = "gen_model_2"

    concat_data = pd.concat([generated1_data, generated2_data])
    concat_data.to_csv(output_file_concat, sep='\t', index=False)


def plot_cluster_heatmap(output_dir, similarities_matrix, model_name):
    sns.clustermap(similarities_matrix, annot=True, method='average', metric='euclidean')
    plt.title(f"Jaccard similarity: {model_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_heatmap.png")
