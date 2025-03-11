import os
from pathlib import Path

import plotly.express as px
import pandas as pd

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.compairr_utils import preprocess_files_for_compairr
from gen_airr_bm.utils.compairr_utils import process_and_save_sequences, run_compairr


def run_novelty_analysis(analysis_config: AnalysisConfig):
    """Runs the novelty analysis."""
    print("Running novelty analysis")
    output_dir = Path(analysis_config.analysis_output_dir)
    os.makedirs(output_dir, exist_ok=True)

    memorization_results = pd.DataFrame(columns=['dataset', 'model_name', 'jaccard_similarity'])
    novelty_results = pd.DataFrame(columns=['dataset', 'model_name', 'jaccard_similarity'])

    compairr_train_dir, train_datasets = setup_directories(analysis_config, "train")
    compairr_test_dir, test_datasets = setup_directories(analysis_config, "test")

    for model_name in analysis_config.model_names:
        model_dir = Path(analysis_config.root_output_dir) / "generated_sequences" / f"{model_name}"
        compairr_model_dir = Path(analysis_config.root_output_dir) / "generated_compairr_sequences" / f"{model_name}"
        preprocess_files_for_compairr(model_dir, compairr_model_dir)
        generated_datasets = os.listdir(compairr_model_dir)

        for train_file, test_file, gen_file in zip(train_datasets, test_datasets, generated_datasets):
            dataset_name = Path(gen_file).stem
            memorization_results = compute_and_store_jaccard(memorization_results, dataset_name, model_name,
                                                             compairr_train_dir / train_file, compairr_model_dir / gen_file,
                                                             output_dir, "train")
            novelty_results = compute_and_store_jaccard(novelty_results, dataset_name, model_name,
                                                        compairr_test_dir / test_file, compairr_model_dir / gen_file,
                                                        output_dir, "test")

    plot_results(memorization_results, output_dir, "train_memorization.png", "train")
    plot_results(novelty_results, output_dir, "test_novelty.png", "test")


def setup_directories(analysis_config, dataset_type):
    """Creates preprocessed directories for train/test/generated sequences."""
    raw_dir = Path(analysis_config.root_output_dir) / f"{dataset_type}_sequences"
    compairr_dir = Path(analysis_config.root_output_dir) / f"{dataset_type}_compairr_sequences"
    preprocess_files_for_compairr(raw_dir, compairr_dir)
    return compairr_dir, os.listdir(compairr_dir)


def compute_and_store_jaccard(results_df, dataset_name, model_name, ref_path, gen_path, output_dir, comparison_type):
    """Computes Jaccard similarity and stores results in DataFrame."""
    helper_dir = Path(output_dir) / "compairr_helper_files"
    os.makedirs(helper_dir, exist_ok=True)

    file_name = f"{dataset_name}_{comparison_type}_{model_name}"
    jaccard_score = compute_jaccard_similarity(helper_dir, ref_path, gen_path, file_name, output_dir, model_name)

    results_df.loc[len(results_df)] = [dataset_name, model_name, jaccard_score]
    return results_df

def compute_jaccard_similarity(compairr_helper_files, reference_path, model_path, file_name, output_dir, model_name):
    unique_sequences_path = compairr_helper_files/ f"{file_name}_unique.tsv"
    concat_sequences_path = compairr_helper_files / f"{file_name}_concat.tsv"
    process_and_save_sequences(reference_path, model_path, unique_sequences_path, concat_sequences_path)

    compairr_output_dir = output_dir / "compairr_output"
    run_compairr(compairr_output_dir, unique_sequences_path, concat_sequences_path, file_name, model_name)

    overlap_df = pd.read_csv(compairr_output_dir / f"{file_name}_overlap.tsv", sep='\t')
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
