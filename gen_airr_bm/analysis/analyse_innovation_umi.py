import os
import re

import numpy as np
import pandas as pd

from collections import defaultdict
from dataclasses import dataclass, field
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_sequence_files_for_no_train_overlap
from gen_airr_bm.utils.compairr_utils import run_compairr_existence, run_sequence_deduplication
from gen_airr_bm.utils.plotting_utils import plot_avg_innovation_scores


@dataclass
class InnovationScores:
    """ Class to store innovation scores for different models and datasets. """
    mean_innovation: dict = field(default_factory=lambda: defaultdict(dict))
    std_innovation: dict = field(default_factory=lambda: defaultdict(dict))
    innovation_all: dict = field(default_factory=lambda: defaultdict(dict))
    mean_innovation_normalized: dict = field(default_factory=lambda: defaultdict(dict))
    std_innovation_normalized: dict = field(default_factory=lambda: defaultdict(dict))
    innovation_normalized_all: dict = field(default_factory=lambda: defaultdict(dict))


def run_innovation_umi_analysis(analysis_config: AnalysisConfig) -> None:
    """ Runs innovation analysis on the generated and reference sequences.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        None
    """
    print("Running innovation umi analysis")

    output_dir = analysis_config.analysis_output_dir
    compairr_output_dir = f"{output_dir}/compairr_output"

    for directory in [output_dir, compairr_output_dir]:
        os.makedirs(directory, exist_ok=True)

    compute_and_plot_innovation_scores(analysis_config, compairr_output_dir)


def compute_and_plot_innovation_scores(analysis_config: AnalysisConfig, compairr_output_dir: str) -> None:
    """ Compute innovation scores and plot them.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        compairr_output_dir (str): Directory to store CompAIRR output files.
    Returns:
        None
    """
    scores = InnovationScores()
    preprocess_test_for_innovation(analysis_config)
    preprocess_gen_for_normalized_innovation(analysis_config)

    for model in analysis_config.model_names:
        collect_model_scores(analysis_config, model, "test_only", compairr_output_dir, scores)

    plot_innovation_scores(analysis_config, scores)


def collect_model_scores(analysis_config: AnalysisConfig, model: str, test_reference: str, compairr_output_dir: str,
                         scores: InnovationScores) -> None:
    """ Collect nnovation scores for a given model.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        model (str): Name of the model to analyze.
        test_reference (str): Reference dataset for testing.
        compairr_output_dir (str): Directory to store CompAIRR output files.
        scores (InnovationScores): Storage for innovation scores.
    Returns:
        None
    """
    comparison_files_dir = get_sequence_files_for_no_train_overlap(analysis_config, model, test_reference)

    for ref_file, gen_files in comparison_files_dir.items():
        dataset_name = os.path.splitext(os.path.basename(ref_file))[0]

        innovation_scores, innovation_scores_normalized = get_innovation_scores(analysis_config, ref_file, gen_files, compairr_output_dir, model)

        mean_ratio, std_ratio = np.mean(innovation_scores), np.std(innovation_scores)
        mean_ratio_normalized, std_ratio_normalized = np.mean(innovation_scores_normalized), np.std(innovation_scores_normalized)

        scores.mean_innovation[dataset_name][model] = mean_ratio
        scores.std_innovation[dataset_name][model] = std_ratio

        scores.mean_innovation_normalized[dataset_name][model] = mean_ratio_normalized
        scores.std_innovation_normalized[dataset_name][model] = std_ratio_normalized

        scores.innovation_all[dataset_name][model] = innovation_scores
        scores.innovation_normalized_all[dataset_name][model] = innovation_scores_normalized


def preprocess_test_for_innovation(analysis_config: AnalysisConfig) -> str:
    test_dir = f"{analysis_config.root_output_dir}/test_compairr_sequences"
    train_dir = f"{analysis_config.root_output_dir}/train_compairr_sequences"

    helper_dir = f"{analysis_config.root_output_dir}/test_only_compairr_sequences"
    os.makedirs(helper_dir, exist_ok=True)

    for file_name in os.listdir(test_dir):
        test_df = pd.read_csv(f"{test_dir}/{file_name}", sep='\t')
        train_df = pd.read_csv(f"{train_dir}/{file_name}", sep='\t')

        test_unique_df = test_df.drop_duplicates(subset=["junction_aa"])
        train_unique_df = train_df.drop_duplicates(subset=["junction_aa"])

        test_only_df = test_unique_df[~test_unique_df["junction_aa"].isin(train_unique_df["junction_aa"])]
        test_only_df.to_csv(f"{helper_dir}/{file_name}", sep='\t', index=False)


def preprocess_gen_for_normalized_innovation(analysis_config: AnalysisConfig) -> str:
    gen_dir = f"{analysis_config.root_output_dir}/generated_compairr_sequences_split"
    train_dir = f"{analysis_config.root_output_dir}/train_compairr_sequences"

    helper_dir = f"{analysis_config.root_output_dir}/test_only_compairr_sequences"
    helper_dir_gen = f"{analysis_config.root_output_dir}/generated_only_compairr_sequences_split"
    os.makedirs(helper_dir, exist_ok=True)
    os.makedirs(helper_dir_gen, exist_ok=True)

    for model in os.listdir(gen_dir):
        os.makedirs(f"{helper_dir_gen}/{model}", exist_ok=True)
        for file_name in os.listdir(f"{gen_dir}/{model}"):
            gen_df = pd.read_csv(f"{gen_dir}/{model}/{file_name}", sep='\t')
            train_file_name = re.sub(r'_\d+\.tsv$', '.tsv', file_name)
            train_df = pd.read_csv(f"{train_dir}/{train_file_name}", sep='\t')

            gen_unique_df = gen_df.drop_duplicates(subset=["junction_aa"])
            train_unique_df = train_df.drop_duplicates(subset=["junction_aa"])

            gen_only_df = gen_unique_df[~gen_unique_df["junction_aa"].isin(train_unique_df["junction_aa"])]
            gen_only_df.to_csv(f"{helper_dir_gen}/{model}/{file_name}", sep='\t', index=False)


def get_innovation_scores(analysis_config: AnalysisConfig, ref_file: str, gen_files: list,
                          compairr_output_dir: str, model: str) -> list:
    """ Get innovation scores for the generated files compared to the reference file.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        ref_file (str): Path to the reference file.
        gen_files (list): List of paths to generated files.
        compairr_output_dir (str): Directory to store CompAIRR output files.
        model (str): Name of the model used for generation.
    Returns:
        tuple: Lists of innovation scores for the generated files.
    """
    innovation_scores = []
    innovation_scores_normalized = []
    for gen_file in gen_files:
        innovation, innovation_normalized = compute_compairr_overlap_ratio(analysis_config, ref_file, gen_file, compairr_output_dir,
                                                model, "innovation")

        innovation_scores.append(innovation)
        innovation_scores_normalized.append(innovation_normalized)

    return innovation_scores, innovation_scores_normalized


def compute_compairr_overlap_ratio(analysis_config: AnalysisConfig, search_for_file: str, search_in_file: str,
                                   compairr_output_dir: str, name: str, metric: str) -> float:
    """ Compute the overlap ratio between two sequence sets using CompAIRR for innovation.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        search_for_file (str): Path to the file of sequences for which to search for existence in another sequence set.
        search_in_file (str): Path to the file to search for existence in.
        compairr_output_dir (str): Directory to store CompAIRR output files.
        name (str): Name of the model used for generation, or "upper_reference" for the upper reference.
        metric (str): Metric type, either "innovation".
    Returns:
        float: Ratio of non-zero overlap counts to total counts.
    """
    file_name = f"{os.path.splitext(os.path.basename(search_in_file))[0]}_{name}_{metric}"

    if analysis_config.deduplicate:
        search_for_file, search_in_file = run_sequence_deduplication(analysis_config, search_for_file, search_in_file)

    run_compairr_existence(compairr_output_dir, search_for_file, search_in_file, file_name,
                           allowed_mismatches=analysis_config.allowed_mismatches, indels=analysis_config.indels)
    compairr_result = pd.read_csv(f"{compairr_output_dir}/{file_name}_overlap.tsv", sep='\t',
                                  names=['sequence_id', 'overlap_count'], header=0)
    n_nonzero_rows = compairr_result[(compairr_result['overlap_count'] != 0)].shape[0]
    ratio = n_nonzero_rows / len(compairr_result)
    gen_only_df = pd.read_csv(search_in_file, sep='\t')
    ratio_normalized = n_nonzero_rows / len(gen_only_df)

    return ratio, ratio_normalized


# TODO: Refactor innovation plotting hack
def plot_innovation_scores(analysis_config: AnalysisConfig, scores: InnovationScores) -> None:
    """ Plot innovation scores for each dataset and model.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores (InnovationScores): Storage class for innovation scores.
    Returns:
        None
    """
    for dataset in scores.mean_innovation:
        plot_avg_innovation_scores(analysis_config, scores.mean_innovation[dataset], scores.std_innovation[dataset],
                                   analysis_config.analysis_output_dir, "innovation",
                                   f"{dataset}_innovation", "innovation",
                                   scoring_method="innovation")

        plot_avg_innovation_scores(analysis_config, scores.mean_innovation_normalized[dataset], scores.std_innovation_normalized[dataset],
                                   analysis_config.analysis_output_dir, "innovation",
                                   f"{dataset}_innovation_normalized", "innovation",
                                   scoring_method="innovation")

    mean_innovation, std_innovation = collapse_mean_std_across_datasets(scores.mean_innovation, scores.std_innovation)
    mean_innovation_normalized, std_innovation_normalized = collapse_mean_std_across_datasets(scores.mean_innovation_normalized, scores.std_innovation_normalized)

    plot_avg_innovation_scores(analysis_config, mean_innovation, std_innovation,
                               analysis_config.analysis_output_dir, "innovation",
                               f"innovation_{analysis_config.receptor_type.replace(' ', '_')}", "innovation",
                               scoring_method="innovation")

    plot_avg_innovation_scores(analysis_config, mean_innovation_normalized, std_innovation_normalized,
                               analysis_config.analysis_output_dir, "innovation",
                               f"innovation_{analysis_config.receptor_type.replace(' ', '_')}_normalized", "innovation",
                               scoring_method="innovation")


def collapse_mean_std_across_datasets(mean_dict, std_dict):
    """
    mean_dict: {dataset: {model: mean_value}}
    std_dict:  {dataset: {model: std_value}}

    Returns:
        final_mean: {model: float}
        final_std:  {model: float}
    """

    # Collect values across datasets
    mean_values = defaultdict(list)
    std_values = defaultdict(list)

    for dataset in mean_dict:
        for model in mean_dict[dataset]:
            mean_values[model].append(mean_dict[dataset][model])
            std_values[model].append(std_dict[dataset][model])

    # Compute final aggregated mean + std
    final_mean = {model: float(np.mean(vals)) for model, vals in mean_values.items()}
    final_std = {model: float(np.mean(vals)) for model, vals in std_values.items()}

    return final_mean, final_std
