import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_sequence_files
from gen_airr_bm.utils.plotting_utils import plot_avg_scores, plot_degree_distribution
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
                                  compairr_output_helper_dir: str) -> None:
    """Compute and plot connectivity scores for the given analysis configuration.

    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        compairr_output_dir (str): Directory to store Compairr output files.
        compairr_output_helper_dir (str): Directory to store helper files for Compairr.

    Returns:
        None
    """
    dataset_split = analysis_config.reference_data
    if not isinstance(dataset_split, str):
        raise ValueError("Network analysis only supports a single dataset split (train or test).")

    mean_scores = defaultdict(lambda: defaultdict(list))
    std_scores = defaultdict(lambda: defaultdict(list))
    for model in analysis_config.model_names:
        comparison_files_dir = get_sequence_files(analysis_config, model, dataset_split)

        for ref_file, gen_files in comparison_files_dir.items():
            dataset_name = os.path.splitext(os.path.basename(ref_file))[0]
            divergence_scores = calculate_degree_divergence_scores(ref_file, gen_files, compairr_output_helper_dir,
                                                                   compairr_output_dir, model, dataset_split,
                                                                   analysis_config.analysis_output_dir, dataset_name,
                                                                   analysis_config.n_unique_samples)

            mean_scores[dataset_name][model] = np.mean(divergence_scores)
            std_scores[dataset_name][model] = np.std(divergence_scores)

    for dataset in mean_scores:
        plot_connectivity_scores(mean_scores[dataset], std_scores[dataset], analysis_config.analysis_output_dir,
                                 dataset_split, "connectivity", f"{dataset}_connectivity.png")


def calculate_degree_divergence_scores(ref_file: str, gen_files: list, compairr_output_helper_dir: str,
                                       compairr_output_dir: str, model: str, dataset_split: str, output_dir: str,
                                       dataset_name: str, n_unique_samples: int | None) -> list:
    """Calculate divergence scores based on node degree distributions. Additionally, plots the degree distributions.

    Args:
        ref_file (str): Path to the reference file.
        gen_files (list): List of generated sequence files.
        compairr_output_helper_dir (str): Directory for Compairr helper files.
        compairr_output_dir (str): Directory for Compairr output files.
        model (str): Model name used for analysis.
        dataset_split (str): Reference data identifier (train or test).
        output_dir (str): Directory to save output plots.
        dataset_name (str): Name of the dataset being analyzed.
        n_unique_samples (int, optional): Maximum number of unique samples to consider. Defaults to None (all considered).

    Returns:
        list: List of divergence scores between generated and reference node degree distributions.
    """
    divergence_scores = []

    ref_degree_dist, gen_degree_dists = get_node_degree_distributions(ref_file, gen_files, compairr_output_helper_dir,
                                                                      compairr_output_dir, model, dataset_split,
                                                                      n_unique_samples)

    plot_degree_distribution(ref_degree_dist, gen_degree_dists, output_dir, model,
                             dataset_split, dataset_name)

    for gen_degree_dist in gen_degree_dists:
        divergence_scores.extend(calculate_jsd(gen_degree_dist, ref_degree_dist))

    return divergence_scores


def get_node_degree_distributions(ref_file: str, gen_files: list, compairr_output_helper_dir: str,
                                  compairr_output_dir: str, model: str, dataset_split: str,
                                  n_unique_samples: int | None) -> tuple:
    """
    Compute node degree distributions for reference and generated files.

    Args:
        ref_file (str): Path to the reference file.
        gen_files (list): List of generated sequence files.
        compairr_output_helper_dir (str): Directory for Compairr helper files.
        compairr_output_dir (str): Directory for Compairr output files.
        model (str): Model name used for analysis.
        dataset_split (str): Reference data identifier (train or test).
        n_unique_samples (int, optional): Maximum number of unique samples to consider. Defaults to None (all considered).

    Returns:
        tuple: Reference node degree distribution and list of generated node degree distributions.
    """
    gen_degree_dists = []
    for gen_file in gen_files:
        gen_connectivity = compute_connectivity_with_compairr(gen_file, compairr_output_helper_dir, compairr_output_dir,
                                                              model, n_unique_samples)
        gen_degree_dist = get_degrees_from_overlap(gen_connectivity)
        gen_degree_dists.append(gen_degree_dist)

    ref_connectivity = compute_connectivity_with_compairr(ref_file, compairr_output_helper_dir, compairr_output_dir,
                                                          dataset_split, n_unique_samples)
    ref_degree_dist = get_degrees_from_overlap(ref_connectivity)

    return ref_degree_dist, gen_degree_dists


def compute_connectivity_with_compairr(input_sequences_path: str, compairr_output_helper_dir: str,
                                       compairr_output_dir: str, dataset_type: str,
                                       n_unique_samples: int | None) -> pd.DataFrame:
    """
    Compute connectivity using Compairr for a single dataset (either train or test).

    Args:
        input_sequences_path (str): Path to the input sequences file.
        compairr_output_helper_dir (str): Directory for Compairr helper files.
        compairr_output_dir (str): Directory for Compairr output files.
        dataset_type (str): Type of dataset (train or test or generated with a specific model).
        n_unique_samples (int, optional): Maximum number of unique samples to consider. Defaults to None (all considered).

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
        deduplicate_single_dataset(input_sequences_path, unique_sequences_path, n_unique_samples)
    run_compairr_existence(compairr_output_dir, unique_sequences_path, unique_sequences_path, file_name)
    compairr_result = pd.read_csv(f"{compairr_output_dir}/{file_name}_overlap.tsv", sep='\t',
                                  names=['sequence_id', 'overlap_count'], header=0)
    return compairr_result


def get_degrees_from_overlap(compairr_result: pd.DataFrame) -> pd.Series:
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


def calculate_jsd(gen_node_degree_distribution: pd.Series, ref_node_degree_distribution: pd.Series) -> list:
    """Compute divergence between two node degree distributions.
    Args:
        gen_node_degree_distribution (pd.Series): Node degree distribution for generated sequences.
        ref_node_degree_distribution (pd.Series): Node degree distribution for reference sequences (test or train).
    Returns:
        list: List containing Jensen-Shannon divergence score.
        """
    merged_df = pd.merge(gen_node_degree_distribution, ref_node_degree_distribution, how='outer',
                         suffixes=('_gen', '_ref'),
                         left_index=True, right_index=True).fillna(0)

    p = merged_df['count_gen'] / merged_df['count_gen'].sum()
    q = merged_df['count_ref'] / merged_df['count_ref'].sum()

    jsd = jensenshannon(p, q, base=2)

    return [jsd]


def plot_connectivity_scores(mean_scores: dict, std_scores: dict, output_dir: str, dataset_split: str,
                             distribution_type: str, file_name: str) -> None:
    """Plot the mean and standard deviation of the divergence scores.
    Args:
        mean_scores (dict): Dictionary with mean divergence scores for each model.
        std_scores (dict): Dictionary with standard deviation of divergence scores for each model.
        output_dir (str): Directory to save the plot.
        dataset_split (str): Reference data identifier (train or test).
        distribution_type (str): Type of distribution being analyzed (e.g., connectivity).
        file_name (str): Name of the output file for the plot.
    Returns:
        None
    """
    plot_avg_scores(mean_scores, std_scores, output_dir, dataset_split,
                    file_name, distribution_type, scoring_method="JSD")
