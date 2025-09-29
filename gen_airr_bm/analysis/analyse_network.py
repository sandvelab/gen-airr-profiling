import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.constants.dataset_split import DatasetSplit
from gen_airr_bm.utils.file_utils import get_sequence_files, get_reference_files
from gen_airr_bm.utils.plotting_utils import plot_avg_scores, plot_degree_distribution, plot_grouped_avg_scores
from gen_airr_bm.utils.compairr_utils import run_compairr_existence, deduplicate_single_dataset


def run_network_analysis(analysis_config: AnalysisConfig) -> None:
    """Run network analysis to compute connectivity plots.

    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.

    Returns:
        None
    """
    print("Running network analysis")

    output_dir = analysis_config.analysis_output_dir
    compairr_output_helper_dir = f"{output_dir}/compairr_helper_files"
    compairr_output_dir = f"{output_dir}/compairr_output"

    for directory in [output_dir, compairr_output_helper_dir, compairr_output_dir]:
        os.makedirs(directory, exist_ok=True)

    compute_and_plot_connectivity(analysis_config, compairr_output_dir, compairr_output_helper_dir)


def compute_and_plot_connectivity(analysis_config: AnalysisConfig, compairr_output_dir: str,
                                  compairr_helper_dir: str) -> None:
    """Compute and plot connectivity scores and distributions for the given analysis configuration.

    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        compairr_output_dir (str): Directory to store Compairr output files.
        compairr_helper_dir (str): Directory to store helper files for Compairr.
    Returns:
        None
    """
    validate_references(analysis_config.reference_data)

    divergence_scores_all = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for reference in analysis_config.reference_data:

        for model_name in analysis_config.model_names:
            comparison_files = get_sequence_files(analysis_config, model_name, reference)

            for ref_file, gen_files in comparison_files.items():
                dataset_name, ref_degree_dist, gen_degree_dists = (
                    get_connectivity_distributions_by_dataset(ref_file, gen_files, compairr_helper_dir,
                                                              compairr_output_dir, model_name, reference,
                                                              analysis_config.analysis_output_dir))
                divergence_scores = calculate_divergence_scores(ref_degree_dist, gen_degree_dists, reference, model_name)
                divergence_scores_all[reference][dataset_name][model_name].extend(divergence_scores)

        for dataset_name, divergence_scores in divergence_scores_all[reference].items():
            summarize_and_plot_dataset_connectivity(dataset_name, divergence_scores,
                                                    analysis_config.analysis_output_dir, reference)

    mean_reference_divergence_score = get_mean_reference_divergence_score(analysis_config, compairr_helper_dir,
                                                                          compairr_output_dir)
    summarize_and_plot_all(divergence_scores_all,
                           analysis_config.analysis_output_dir,
                           analysis_config.reference_data,
                           mean_reference_divergence_score)


def get_connectivity_distributions_by_dataset(ref1_file: str, ref2_or_gen_files: list[str], helper_dir: str,
                                              output_dir: str, name: str, reference: str, analysis_output_dir: str) -> (
        tuple)[str, pd.Series, list[pd.Series]]:
    """ For a given dataset, this function computes connectivity distributions of reference set 1 (train or test) and
    either reference set 2 (test) or model generated sets. Connectivity distributions are then plotted as histograms.
    Args:
        ref1_file (str): Path to the reference file 1. E.g. path to train or test file.
        ref2_or_gen_files (list[str]): List of one test file or list of generated sequence files.
        helper_dir (str): Directory for Compairr helper files.
        output_dir (str): Directory for Compairr output files.
        reference (str): Reference data identifier (train or test).
        name (str): Name of generative model or name of second reference set.
        analysis_output_dir (str): Directory to save output plots.
    Returns:
        tuple: Dataset name, reference1 degree distribution, list of reference 2 or generated degree distributions.
    """
    dataset_name = os.path.splitext(os.path.basename(ref1_file))[0]

    ref1_degree_dist, ref2_or_gen_degree_dists = get_node_degree_distributions(ref1_file, ref2_or_gen_files, helper_dir,
                                                                               output_dir, name, reference)

    plot_degree_distribution(ref1_degree_dist, ref2_or_gen_degree_dists, analysis_output_dir, name, reference,
                             dataset_name)

    return dataset_name, ref1_degree_dist, ref2_or_gen_degree_dists


def calculate_divergence_scores(ref1_degree_dist: pd.Series, ref2_or_gen_degree_dists: list[pd.Series],
                                dist1_name: str, dist2_name: str) -> list[float]:
    """ Calculate divergence scores (JSD) between reference set 1 (train or test) and generated degree distributions or
    between reference set 1 (train) and reference set 2 (test) degree distributions.
    Args:
        ref1_degree_dist (pd.Series): Node degree distribution for reference sequences. Either train or test sequences.
        ref2_or_gen_degree_dists (list[pd.Series]): List of node degree distributions for reference set 2 or generated
        sequences.
        dist1_name (str): Name of the first distribution.
        dist2_name (str): Name of the second distribution.
    Returns:
        list: List of divergence scores (JSD) for each generated distribution compared to the reference or list or one
        score for one reference set 1 distribution (train) compared to corresponding reference set 2 distribution (test).
    """
    divergence_scores = []
    for dist in ref2_or_gen_degree_dists:
        jsd_score = calculate_jsd(dist, ref1_degree_dist, dist2_name, dist1_name)
        divergence_scores.extend(jsd_score)
    return divergence_scores


def get_node_degree_distributions(ref1_file: str, ref2_or_gen_files: list, compairr_output_helper_dir: str,
                                  compairr_output_dir: str, name: str, dataset_split: str) -> tuple:
    """
    Compute node degree distributions for reference set 1 (train or test) and the corresponding generated files or
    reference set 2 (test).
    Args:
        ref1_file (str): Path to the reference file (train or test).
        ref2_or_gen_files (list): List of generated sequence files or list of one corresponding test file.
        compairr_output_helper_dir (str): Directory for Compairr helper files.
        compairr_output_dir (str): Directory for Compairr output files.
        name (str): Name of generative model or name of second reference set.
        dataset_split (str): Reference data identifier (train or test).
    Returns:
        tuple: Reference1 node degree distribution and list of generated node degree distributions or list of one
        reference 2 node degree distribution.
    """
    ref2_or_gen_degree_dists = []
    for file in ref2_or_gen_files:
        connectivity = compute_connectivity_with_compairr(file, compairr_output_helper_dir, compairr_output_dir, name)
        degree_dist = get_node_degree_from_compairr_output(connectivity)
        ref2_or_gen_degree_dists.append(degree_dist)

    ref1_connectivity = compute_connectivity_with_compairr(ref1_file, compairr_output_helper_dir, compairr_output_dir,
                                                           dataset_split)
    ref1_degree_dist = get_node_degree_from_compairr_output(ref1_connectivity)

    return ref1_degree_dist, ref2_or_gen_degree_dists


def get_mean_reference_divergence_score(analysis_config: AnalysisConfig, compairr_output_helper_dir: str,
                                        compairr_output_dir: str) -> float:
    """ Get mean divergence score (JSD) for the reference data.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        compairr_output_helper_dir (str): Directory for Compairr helper files.
        compairr_output_dir (str): Directory for Compairr output files.
    Returns:
        float: Mean divergence score for the reference data.
    """
    ref_scores = []
    reference_comparison_files = get_reference_files(analysis_config)
    for train_file, test_file in reference_comparison_files:
        dataset_name, train_node_degree, test_node_degree = (
            get_connectivity_distributions_by_dataset(train_file, [test_file], compairr_output_helper_dir,
                                                      compairr_output_dir, DatasetSplit.TEST.value,
                                                      DatasetSplit.TRAIN.value, analysis_config.analysis_output_dir))
        divergence_scores = calculate_divergence_scores(train_node_degree, test_node_degree, DatasetSplit.TRAIN.value,
                                                        DatasetSplit.TEST.value)
        ref_scores.extend(divergence_scores)
    mean_ref_divergence_score = np.mean(ref_scores)

    return mean_ref_divergence_score


def compute_connectivity_with_compairr(input_sequences_path: str, compairr_output_helper_dir: str,
                                       compairr_output_dir: str, dataset_type: str) -> pd.DataFrame:
    """
    Compute connectivity using Compairr for a single dataset (either train, test, or generated).

    Args:
        input_sequences_path (str): Path to the input sequences file.
        compairr_output_helper_dir (str): Directory for Compairr helper files.
        compairr_output_dir (str): Directory for Compairr output files.
        dataset_type (str): Type of dataset (train or test or generated with a specific model).

    Returns:
        pd.DataFrame: DataFrame containing sequence IDs and their overlap counts.
    """
    if not os.path.exists(input_sequences_path):
        raise FileNotFoundError(f"Input sequences file not found: {input_sequences_path}")
    file_name = f"{os.path.splitext(os.path.basename(input_sequences_path))[0]}_{dataset_type}"
    unique_sequences_path = f"{compairr_output_helper_dir}/{file_name}_unique.tsv"
    if os.path.exists(unique_sequences_path):
        print(f"Unique sequences already exist for {file_name}. Skipping execution.")
    else:
        deduplicate_single_dataset(input_sequences_path, unique_sequences_path)
    run_compairr_existence(compairr_output_dir, unique_sequences_path, unique_sequences_path, file_name,
                           allowed_mismatches=1, indels=True)
    compairr_result = pd.read_csv(f"{compairr_output_dir}/{file_name}_overlap.tsv", sep='\t',
                                  names=['sequence_id', 'overlap_count'], header=0)
    return compairr_result


def get_node_degree_from_compairr_output(compairr_result: pd.DataFrame) -> pd.Series:
    """
    Extract node degree distribution from Compairr overlap results.

    Args:
        compairr_result (pd.DataFrame): DataFrame containing sequence IDs and their overlap counts.

    Returns:
        pd.Series: Node degree distribution, where index is the degree and values are counts.
    """
    if 'overlap_count' not in compairr_result.columns:
        raise ValueError("Compairr result DataFrame must contain 'overlap_count' column.")

    # Adjust overlap count to exclude self-loops
    compairr_result['overlap_count'] -= 1
    node_degree_distribution = compairr_result['overlap_count'].value_counts()
    return node_degree_distribution


def calculate_jsd(node_degree_dist1: pd.Series, node_degree_dist2: pd.Series, dist1_name: str, dist2_name: str) -> list:
    """Compute divergence between two node degree distributions.
    Args:
        node_degree_dist1 (pd.Series): Node degree distribution for set 1.
        node_degree_dist2 (pd.Series): Node degree distribution for set 2.
        dist1_name (str): Name of the first distribution.
        dist2_name (str): Name of the second distribution.
    Returns:
        list: List containing Jensen-Shannon divergence score.
        """
    suffixes = (f"_{dist1_name}", f"_{dist2_name}")
    merged_df = pd.merge(node_degree_dist1, node_degree_dist2, how='outer',
                         suffixes=suffixes,
                         left_index=True, right_index=True).fillna(0)

    p = merged_df[f'count{suffixes[0]}'] / merged_df[f'count{suffixes[0]}'].sum()
    q = merged_df[f'count{suffixes[1]}'] / merged_df[f'count{suffixes[1]}'].sum()

    jsd = jensenshannon(p, q, base=2)

    return [jsd]


def summarize_and_plot_dataset_connectivity(dataset_name: str, divergence_scores: dict[str, list[float]],
                                            output_dir: str, reference: str) -> None:
    """ Compute mean/std and plot dataset-level connectivity scores.
    Args:
        dataset_name (str): Name of the dataset being analyzed.
        divergence_scores (dict): Dictionary with divergence scores for each model.
        output_dir (str): Directory to save the plot.
        reference (str): Reference data identifier (train or test).
    Returns:
        None
    """
    mean_scores = {m: float(np.mean(scores)) for m, scores in divergence_scores.items()}
    std_scores = {m: float(np.std(scores)) for m, scores in divergence_scores.items()}

    plot_avg_scores(
        mean_scores_dict=mean_scores,
        std_scores_dict=std_scores,
        output_dir=output_dir,
        reference_data=reference,
        distribution_type="connectivity",
        file_name=f"{dataset_name}_connectivity",
    )


def summarize_and_plot_all(divergence_scores_all: dict[str, dict[str, dict[str, list]]], output_dir: str,
                           reference_datasets: list[str], mean_reference_score: float) -> None:
    """ Aggregate scores across references and plot grouped averages.
    Args:
        divergence_scores_all (dict(dict(dict[list]))): Double nested dictionary with divergence scores by reference,
        dataset and model.
        output_dir (str): Directory to save the plot.
        reference_datasets (list[str]): List of reference dataset identifiers.
        mean_reference_score (float): Mean divergence score for the reference data.
    Returns:
        None
    """
    mean_scores, std_scores = {}, {}
    for reference, dataset_scores in divergence_scores_all.items():
        model_scores = {model: [score for sample in dataset_scores.values() for score in sample[model]]
                        for model in next(iter(dataset_scores.values())).keys()}
        mean_scores[reference] = {m: float(np.mean(scores)) for m, scores in model_scores.items()}
        std_scores[reference] = {m: float(np.std(scores)) for m, scores in model_scores.items()}

    plot_grouped_avg_scores(
        mean_scores_by_ref=mean_scores,
        std_scores_by_ref=std_scores,
        output_dir=output_dir,
        reference_data=reference_datasets,
        distribution_type="connectivity",
        file_name="all_datasets_connectivity",
        scoring_method="JSD",
        reference_score=mean_reference_score
    )


def validate_references(reference_datasets: list[str]) -> None:
    """ Validate that reference datasets are either 'train' or 'test'.
    Args:
        reference_datasets (list[str]): List of reference dataset identifiers.
    Returns:
        None
    """
    for ref in reference_datasets:
        if ref not in [DatasetSplit.TRAIN.value, DatasetSplit.TEST.value]:
            raise ValueError("Network analysis only supports 'train' or 'test' as reference data.")

