import os
import plotly.express as px
import pandas as pd

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.compairr_utils import preprocess_files_for_compairr

from gen_airr_bm.utils.compairr_utils import process_and_save_sequences, run_compairr


def run_novelty_analysis(analysis_config: AnalysisConfig):
    """Runs the novelty analysis."""
    print("Running novelty analysis")

    output_dir = analysis_config.analysis_output_dir

    # assess memorization
    results = pd.DataFrame(columns=['reference', 'dataset', 'model_name', 'jaccard_similarity'])
    train_sequences_dir = f"{analysis_config.root_output_dir}/train_sequences"
    compairr_train_sequences_dir = f"{analysis_config.root_output_dir}/train_compairr_sequences"
    preprocess_files_for_compairr(train_sequences_dir, compairr_train_sequences_dir)
    train_datasets = os.listdir(compairr_train_sequences_dir)

    idx = 0
    for model_name in analysis_config.model_names:
        model_sequences_dir = f"{analysis_config.root_output_dir}/generated_sequences/{model_name}"
        compairr_model_sequences_dir = f"{analysis_config.root_output_dir}/generated_compairr_sequences/{model_name}"
        preprocess_files_for_compairr(model_sequences_dir, compairr_model_sequences_dir)
        generated_datasets = os.listdir(compairr_model_sequences_dir)

        for train_dataset, generated_dataset in zip(train_datasets, generated_datasets):
            train_data_name = train_dataset.strip('.tsv')
            train_data_path = f"{compairr_train_sequences_dir}/{train_dataset}"
            generated_data_path = f"{compairr_model_sequences_dir}/{generated_dataset}"

            compairr_helper_files = f"{output_dir}/compairr_helper_files"
            os.makedirs(compairr_helper_files, exist_ok=True)
            file_name = f"{train_data_name}_train_{model_name}"
            unique_sequences_path = f"{compairr_helper_files}/{file_name}_unique.tsv"
            concat_sequences_path = f"{compairr_helper_files}/{file_name}_concat.tsv"
            process_and_save_sequences(train_data_path, generated_data_path, unique_sequences_path, concat_sequences_path)

            compairr_output_dir = f"{output_dir}/compairr_output"
            run_compairr(compairr_output_dir, unique_sequences_path, concat_sequences_path, file_name, model_name)

            overlap_df = pd.read_csv(f"{compairr_output_dir}/{file_name}_overlap.tsv", sep='\t')
            n_nonzero_rows = overlap_df[(overlap_df['dataset_1'] != 0) & (overlap_df['dataset_2'] != 0)].shape[0]

            union = pd.read_csv(unique_sequences_path, sep='\t').shape[0]
            jaccard_similarity = n_nonzero_rows / union

            results.loc[idx] = ['train', train_data_name, model_name, jaccard_similarity]
            idx += 1

    plot_results(results, output_dir, "train_memorization.png")

    # assess novelty
    test_sequences_dir = f"{analysis_config.root_output_dir}/test_sequences"


def plot_results(results, fig_dir, file_name):
    fig = px.bar(results,
                 x="dataset",
                 y="jaccard_similarity",
                 color="model_name",
                 title="Memorization: Jaccard Similarities between train and model",
                 labels={"jaccard_similarity": "Jaccard Similarity", "dataset": "Dataset"},
                 barmode="group")

    fig.update_xaxes(tickangle=45)

    png_path = os.path.join(fig_dir, file_name)
    fig.write_image(png_path)
    print(f"Plot saved as PNG at: {png_path}")
