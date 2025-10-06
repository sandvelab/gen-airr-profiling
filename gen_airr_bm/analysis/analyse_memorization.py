import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.compairr_utils import deduplicate_and_merge_two_datasets, run_compairr_existence
from gen_airr_bm.utils.file_utils import get_sequence_files, get_reference_files


def run_memorization_analysis(analysis_config: AnalysisConfig) -> None:
    """ Runs the memorization analysis.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        None
    """
    print("Running memorization analysis...")

    output_dir = analysis_config.analysis_output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_reference = "train" if "train" in analysis_config.reference_data else None
    test_reference = "test" if "test" in analysis_config.reference_data else None
    if train_reference is None or test_reference is None:
        raise ValueError("Train and test data must be included in reference_data for memorization analysis.")

    model_memorization_scores = get_model_memorization_scores(analysis_config, output_dir, train_reference)
    mean_reference_memorization_score = get_mean_reference_memorization_score(analysis_config, output_dir)

    plot_results(model_memorization_scores, mean_reference_memorization_score, output_dir, "memorization")


def get_model_memorization_scores(analysis_config: AnalysisConfig, output_dir: str, train_reference: str) -> dict:
    """ Get memorization scores (Jaccard similarities) for each model.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        output_dir (str): Directory to save intermediate and final results.
        train_reference (str): Reference data split to use for training (e.g., "train").
    Returns:
        dict: A dictionary with model names as keys and lists of memorization scores as values.
    """
    model_memorization_scores = defaultdict(list)
    for model_name in analysis_config.model_names:
        comparison_files_dir = get_sequence_files(analysis_config, model_name, train_reference)
        model_memorization_scores[model_name] = []
        for ref_file, gen_files in comparison_files_dir.items():
            model_memorization_scores[model_name].extend(get_memorization_scores(ref_file, gen_files,
                                                                                 output_dir, model_name))
    return model_memorization_scores


def get_mean_reference_memorization_score(analysis_config: AnalysisConfig, output_dir: str) -> float:
    """ Get mean memorization score (Jaccard similarity) for the reference data.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        output_dir (str): Directory to save intermediate and final results.
    Returns:
        float: Mean memorization score between the reference datasets (average train vs. test).
    """
    ref_scores = []
    reference_comparison_files = get_reference_files(analysis_config)
    for train_file, test_file in reference_comparison_files:
        ref_score = get_memorization_scores(train_file, [test_file], output_dir, "reference")
        ref_scores.append(ref_score[0])
    mean_ref_memorization_score = np.mean(ref_scores)

    return mean_ref_memorization_score


def get_memorization_scores(train_file: str, test_or_gen_files: list[str], output_dir: str, name: str) -> list:
    """ Compute memorization scores (Jaccard similarities) between a train reference file and generated files or between
    train and the corresponding test set.
    Args:
        train_file (str): Path to the train file.
        test_or_gen_files (list[str]): List of paths to generated files or list of one test file.
        output_dir (str): Directory to save intermediate and final results.
        name (str): Name of the model being evaluated or "reference".
    Returns:
        list: A list of memorization scores (Jaccard similarities) for each model.
    """
    memorization_scores = []
    compairr_helper_dir = f"{output_dir}/compairr_helper_files"
    os.makedirs(compairr_helper_dir, exist_ok=True)
    for file in test_or_gen_files:
        score = compute_jaccard_similarity(compairr_helper_dir, train_file, file, output_dir, name)
        memorization_scores.append(score)

    return memorization_scores


def compute_jaccard_similarity(compairr_helper_dir: str, train_file: str, test_or_gen_file: str, output_dir: str,
                               name: str) -> float:
    """ Compute Jaccard similarity between two datasets using CompAIRR.
    Args:
        compairr_helper_dir (str): Directory to save helper files for CompAIRR.
        train_file (str): Path to the reference train set.
        test_or_gen_file (str): Path to the model-generated set or to corresponding test set.
        output_dir (str): Directory to save CompAIRR output.
        name (str): Name of the model being evaluated or "reference".
    Returns:
        float: Jaccard similarity between the two datasets.
    """
    dataset_name = os.path.splitext(os.path.basename(test_or_gen_file))[0]
    file_identifier = f"{dataset_name}_{name}"
    unique_sequences_path = f"{compairr_helper_dir}/{file_identifier}_unique.tsv"
    concat_sequences_path = f"{compairr_helper_dir}/{file_identifier}_concat.tsv"
    if os.path.exists(unique_sequences_path) and os.path.exists(concat_sequences_path):
        print(f"Deduplicated and merged files already exist for {file_identifier}. Skipping this step.")
    else:
        deduplicate_and_merge_two_datasets(train_file, test_or_gen_file, unique_sequences_path, concat_sequences_path)

    compairr_output_dir = f"{output_dir}/compairr_output"
    run_compairr_existence(compairr_output_dir, unique_sequences_path, concat_sequences_path, file_identifier,
                           allowed_mismatches=0, indels=False)

    overlap_df = pd.read_csv(f"{compairr_output_dir}/{file_identifier}_overlap.tsv", sep='\t')
    n_nonzero_rows = overlap_df[(overlap_df['dataset_1'] != 0) & (overlap_df['dataset_2'] != 0)].shape[0]

    union = pd.read_csv(unique_sequences_path, sep='\t').shape[0]
    jaccard_similarity = n_nonzero_rows / union

    return jaccard_similarity


def plot_results(model_scores: dict, mean_reference_score: float, fig_dir: str, file_name: str) -> None:
    """ Plot memorization scores for each model with std error bars and reference line.
    Args:
        model_scores (dict): Dictionary with model names as keys and lists of memorization scores as values.
        mean_reference_score (float): Mean memorization score for the reference data.
        fig_dir (str): Directory to save the plot.
        file_name (str): Name of the output plot and tsv files (without extension).
    Returns:
        None
    """
    os.makedirs(fig_dir, exist_ok=True)
    png_path = os.path.join(fig_dir, file_name) + ".png"
    tsv_path = os.path.join(fig_dir, file_name) + ".tsv"

    means = {k: np.mean(v) for k, v in model_scores.items()}
    stds = {k: np.std(v) for k, v in model_scores.items()}

    models, scores = zip(*sorted(means.items(), key=lambda x: x[1], reverse=True))
    errors = [stds[model] for model in models]

    if not os.path.exists(tsv_path):
        pd.DataFrame({"model": models, "mean_jaccard_similarity": scores, "std": errors}).to_csv(
            tsv_path, sep="\t", index=False)
        with open(os.path.join(fig_dir, file_name) + "_mean_ref.tsv", "w") as f:
            f.write(f"{mean_reference_score}\n")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=models,
        y=scores,
        error_y=dict(type='data', array=errors, visible=True),
        marker=dict(color='skyblue'),
    ))

    fig.update_layout(
        title=f"Average Memorization Scores Across Models",
        xaxis_title="Models",
        yaxis_title=f"Mean Jaccard Similarity",
        xaxis_tickangle=-45,
        template="plotly_white"
    )

    fig.add_hline(
        y=mean_reference_score,
        line=dict(color="black", dash="dash"),
        annotation_text=f"reference={mean_reference_score:.3f}",
        annotation_position="top right"
    )

    fig.write_image(png_path)

    print(f"Plot saved as PNG at: {png_path}")
