import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from gen_airr_bm.core.analysis_config import AnalysisConfig
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

    divergence_scores_by_reference = defaultdict(lambda: defaultdict(list))

    for reference in analysis_config.reference_data:
        divergence_scores_by_dataset = defaultdict(lambda: defaultdict(list))

        for model_name in analysis_config.model_names:
            comparison_files = get_sequence_files(analysis_config, model_name, reference)

            for ref_file, gen_files in comparison_files.items():
                dataset_name, divergence_scores = process_dataset(ref_file, gen_files, compairr_helper_dir,
                                                                  compairr_output_dir, model_name, reference,
                                                                  analysis_config.analysis_output_dir)

                for model, scores in divergence_scores.items():
                    divergence_scores_by_reference[reference][model].extend(scores)
                    divergence_scores_by_dataset[dataset_name][model].extend(scores)

        for dataset_name, divergence_scores in divergence_scores_by_dataset.items():
            summarize_and_plot_dataset_connectivity(dataset_name, divergence_scores,
                                                    analysis_config.analysis_output_dir, reference)

    mean_reference_divergence_score = get_reference_divergence_score(analysis_config, compairr_helper_dir,
                                                                     compairr_output_dir, "train", "test")
    summarize_and_plot_all(divergence_scores_by_reference,
                           analysis_config.analysis_output_dir,
                           analysis_config.reference_data,
                           mean_reference_divergence_score)


def process_dataset(ref_file: str, gen_files: list[str], helper_dir: str, output_dir: str, model: str, reference: str,
                    analysis_output_dir: str) -> tuple[str, dict[str, list[float]]]:
    """Compute degree distributions, plot them, and return divergence scores for a dataset.
    Args:
        ref_file (str): Path to the reference file.
        gen_files (list[str]): List of generated sequence files.
        helper_dir (str): Directory for Compairr helper files.
        output_dir (str): Directory for Compairr output files.
        model (str): Model name used for analysis.
        reference (str): Reference data identifier (train or test).
        analysis_output_dir (str): Directory to save output plots.
    Returns:
        tuple: Dataset name and dictionary with divergence scores for each model.
    """
    dataset_name = os.path.splitext(os.path.basename(ref_file))[0]

    ref_degree_dist, gen_degree_dists = get_node_degree_distributions(ref_file, gen_files, helper_dir, output_dir,
                                                                      model, reference)

    plot_degree_distribution(ref_degree_dist, gen_degree_dists, analysis_output_dir, model, reference, dataset_name)

    divergence_scores = defaultdict(list)
    for gen_degree_dist in gen_degree_dists:
        jsd_score = calculate_jsd(gen_degree_dist, ref_degree_dist)
        divergence_scores[model].append(jsd_score)

    return dataset_name, divergence_scores


def get_node_degree_distributions(ref_file: str, gen_files: list, compairr_output_helper_dir: str,
                                  compairr_output_dir: str, model: str, dataset_split: str) -> tuple:
    """
    Compute node degree distributions for reference and generated files.

    Args:
        ref_file (str): Path to the reference file.
        gen_files (list): List of generated sequence files.
        compairr_output_helper_dir (str): Directory for Compairr helper files.
        compairr_output_dir (str): Directory for Compairr output files.
        model (str): Model name used for analysis.
        dataset_split (str): Reference data identifier (train or test).

    Returns:
        tuple: Reference node degree distribution and list of generated node degree distributions.
    """
    gen_degree_dists = []
    for gen_file in gen_files:
        gen_connectivity = compute_connectivity_with_compairr(gen_file, compairr_output_helper_dir, compairr_output_dir,
                                                              model)
        gen_degree_dist = get_degrees_from_overlap(gen_connectivity)
        gen_degree_dists.append(gen_degree_dist)

    ref_connectivity = compute_connectivity_with_compairr(ref_file, compairr_output_helper_dir, compairr_output_dir,
                                                          dataset_split)
    ref_degree_dist = get_degrees_from_overlap(ref_connectivity)

    return ref_degree_dist, gen_degree_dists


def get_reference_divergence_score(analysis_config: AnalysisConfig, compairr_output_helper_dir: str,
                                   compairr_output_dir: str, train_ref: str, test_ref: str) -> float:
    """ Get mean divergence score (JSD) for the reference data.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        compairr_output_helper_dir (str): Directory for Compairr helper files.
        compairr_output_dir (str): Directory for Compairr output files.
        train_ref (str): First reference data identifier.
        test_ref (str): Second reference data identifier.
    Returns:
        float: Mean divergence score for the reference data.
    """
    ref_scores = []
    reference_comparison_files = get_reference_files(analysis_config)
    for train_file, test_file in reference_comparison_files:
        dataset_name, divergence_scores = process_dataset(train_file, test_file, compairr_output_helper_dir,
                                                          compairr_output_dir, test_ref, train_ref,
                                                          analysis_config.analysis_output_dir)
        ref_scores.extend(divergence_scores[test_ref][0])
    mean_ref_divergence_score = np.mean(ref_scores)

    return mean_ref_divergence_score


def compute_connectivity_with_compairr(input_sequences_path: str, compairr_output_helper_dir: str,
                                       compairr_output_dir: str, dataset_type: str) -> pd.DataFrame:
    """
    Compute connectivity using Compairr for a single dataset (either train or test).

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
        file_name=f"{dataset_name}_connectivity.png",
    )


def summarize_and_plot_all(divergence_scores_by_reference: dict, output_dir: str,
                           reference_datasets: list[str], mean_reference_score: float) -> None:
    """ Aggregate scores across references and plot grouped averages.
    Args:
        divergence_scores_by_reference (dict): Nested dictionary with divergence scores by reference and model.
        output_dir (str): Directory to save the plot.
        reference_datasets (list[str]): List of reference dataset identifiers.
        mean_reference_score (float): Mean divergence score for the reference data.
    Returns:
        None
    """
    mean_scores, std_scores = {}, {}
    for reference, model_scores in divergence_scores_by_reference.items():
        mean_scores[reference] = {m: float(np.mean(scores)) for m, scores in model_scores.items()}
        std_scores[reference] = {m: float(np.std(scores)) for m, scores in model_scores.items()}

    plot_grouped_avg_scores(
        mean_scores_by_ref=mean_scores,
        std_scores_by_ref=std_scores,
        output_dir=output_dir,
        reference_data=reference_datasets,
        distribution_type="connectivity",
        file_name="all_datasets_connectivity.png",
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
        if ref not in {"train", "test"}:
            raise ValueError("Network analysis only supports 'train' or 'test' as reference data.")

