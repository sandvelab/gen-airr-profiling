import os
import pandas as pd
import plotly.express as px

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.compairr_utils import (deduplicate_and_merge_two_datasets, run_compairr_existence,
                                              setup_directories)


def run_memorization_analysis(analysis_config: AnalysisConfig):
    """Runs the memorization analysis."""
    print("Running memorization analysis")
    output_dir = analysis_config.analysis_output_dir
    os.makedirs(output_dir, exist_ok=True)

    memorization_results = pd.DataFrame(columns=['dataset', 'model_name', 'jaccard_similarity'])

    compairr_train_dir, train_datasets = setup_directories(analysis_config, "train")
    compairr_test_dir, test_datasets = setup_directories(analysis_config, "test")

    for model_name in analysis_config.model_names:
        compairr_model_dir = f"{analysis_config.root_output_dir}/generated_compairr_sequences/{model_name}"
        generated_datasets = os.listdir(compairr_model_dir)

        for train_file, test_file, gen_file in zip(train_datasets, test_datasets, generated_datasets):
            dataset_name = os.path.splitext(os.path.basename(gen_file))[0]
            memorization_results = compute_and_store_jaccard(memorization_results, dataset_name, model_name,
                                                             f"{compairr_train_dir}/{train_file}",
                                                             f"{compairr_model_dir}/{gen_file}",
                                                             output_dir, "train")

    plot_results(memorization_results, output_dir, "train_memorization.png", "train")


def compute_and_store_jaccard(results_df, dataset_name, model_name, ref_path, gen_path, output_dir, comparison_type):
    """Computes Jaccard similarity and stores results in DataFrame."""
    helper_dir = f"{output_dir}/compairr_helper_files"
    os.makedirs(helper_dir, exist_ok=True)

    file_name = f"{dataset_name}_{comparison_type}_{model_name}"
    jaccard_score = compute_jaccard_similarity(helper_dir, ref_path, gen_path, file_name, output_dir, model_name)

    results_df.loc[len(results_df)] = [dataset_name, model_name, jaccard_score]
    return results_df


def compute_jaccard_similarity(compairr_helper_files, reference_path, model_path, file_name, output_dir, model_name):
    unique_sequences_path = f"{compairr_helper_files}/{file_name}_unique.tsv"
    concat_sequences_path = f"{compairr_helper_files}/{file_name}_concat.tsv"
    deduplicate_and_merge_two_datasets(reference_path, model_path, unique_sequences_path, concat_sequences_path)

    compairr_output_dir = f"{output_dir}/compairr_output"
    run_compairr_existence(compairr_output_dir, unique_sequences_path, concat_sequences_path, file_name, model_name)

    overlap_df = pd.read_csv(f"{compairr_output_dir}/{file_name}_overlap.tsv", sep='\t')
    n_nonzero_rows = overlap_df[(overlap_df['dataset_1'] != 0) & (overlap_df['dataset_2'] != 0)].shape[0]

    union = pd.read_csv(unique_sequences_path, sep='\t').shape[0]
    jaccard_similarity = n_nonzero_rows / union
    return jaccard_similarity


def plot_results(results, fig_dir, file_name, reference):
    fig = px.bar(results,
                 x="dataset",
                 y="jaccard_similarity",
                 color="model_name",
                 title=f"Jaccard Similarities between {reference} and model",
                 labels={"jaccard_similarity": "Jaccard Similarity", "dataset": "Dataset"},
                 barmode="group")

    fig.update_xaxes(tickangle=45)

    png_path = os.path.join(fig_dir, file_name)
    fig.write_image(png_path)
    print(f"Plot saved as PNG at: {png_path}")
