import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.constants.dataset_split import DatasetSplit
from gen_airr_bm.utils.file_utils import get_sequence_files, get_reference_files, get_subset_index
from gen_airr_bm.utils.plotting_utils import (
    get_collection_specification_for_title,
    plot_avg_scores,
    plot_grouped_avg_scores,
    plot_degree_distribution_by_dataset,
    plot_repertoire_statistics,
)
from gen_airr_bm.utils.compairr_utils import run_compairr_existence, deduplicate_single_dataset
from gen_airr_bm.utils.statistics_utils import (aggregate_repertoire_scores, check_one_repertoire_per_donor,
                                               run_repertoire_statistics, summarise_scores_by_reference)

METRIC_NAME = "connectivity"
SCORE_COLUMNS = ["metric", "model", "reference", "dataset", "subset", "jsd"]


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
    """Compute and plot connectivity scores and distributions for the given analysis configuration. The divergence
    of every generated subset is averaged within its repertoire before the models are summarised over repertoires (see
    statistics_utils). Against the test sequences, the models are also compared at the repertoire level: with the
    train vs. test reference, with each other, and in the contrasts given in the config. These comparisons are
    descriptive, or include confidence intervals and tests if statistical_tests is set, which requires a donor_pattern
    and one repertoire per donor.

    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        compairr_output_dir (str): Directory to store Compairr output files.
        compairr_helper_dir (str): Directory to store helper files for Compairr.
    Returns:
        None
    """
    validate_references(analysis_config.reference_data)

    divergence_scores_all = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    connectivity_distributions_all = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    scores = []

    for reference in analysis_config.reference_data:

        for model_name in analysis_config.model_names:
            comparison_files = get_sequence_files(analysis_config, model_name, reference)
            for ref_file, gen_files in comparison_files.items():
                dataset_name, ref_degree_dist, gen_degree_dists = (
                    get_connectivity_distributions_by_dataset(ref_file, gen_files, compairr_helper_dir,
                                                              compairr_output_dir, model_name, reference,
                                                              analysis_config.allowed_mismatches,
                                                              analysis_config.indels))
                divergence_scores = calculate_divergence_scores(ref_degree_dist, gen_degree_dists, reference,
                                                                model_name)
                divergence_scores_all[reference][dataset_name][model_name].extend(divergence_scores)
                scores.extend({"metric": METRIC_NAME, "model": model_name, "reference": reference,
                               "dataset": dataset_name, "subset": get_subset_index(gen_file), "jsd": score}
                              for gen_file, score in zip(gen_files, divergence_scores))

                if model_name not in connectivity_distributions_all[dataset_name].keys():
                    connectivity_distributions_all[dataset_name][model_name][model_name].extend(gen_degree_dists)
                if reference not in connectivity_distributions_all[dataset_name].keys():
                    connectivity_distributions_all[dataset_name][model_name][reference].append(ref_degree_dist)

        for dataset_name, divergence_scores in divergence_scores_all[reference].items():
            summarize_and_plot_dataset_connectivity(dataset_name, divergence_scores,
                                                    analysis_config.analysis_output_dir, reference)

    scores_df = pd.DataFrame(scores, columns=SCORE_COLUMNS)
    scores_df.to_csv(f"{analysis_config.analysis_output_dir}/connectivity_jsd_scores.tsv", sep="\t", index=False)
    if analysis_config.statistical_tests:
        check_one_repertoire_per_donor(aggregate_repertoire_scores(scores_df, "jsd", analysis_config.donor_pattern),
                                       analysis_config.donor_pattern)

    reference_scores = get_reference_divergence_scores(analysis_config, compairr_helper_dir, compairr_output_dir)
    summarize_and_plot_all(analysis_config, scores_df, reference_scores)
    run_network_statistics(analysis_config, scores_df, reference_scores)

    plot_degree_distribution_by_dataset(analysis_config, connectivity_distributions_all,)


def get_connectivity_distributions_by_dataset(ref1_file: str, ref2_or_gen_files: list[str], helper_dir: str,
                                              output_dir: str, name: str, reference: str, allowed_mismatches: int,
                                              indels: bool) -> (
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
        allowed_mismatches (int): Number of allowed mismatches for Compairr.
        indels (bool): Whether to allow indels in Compairr.
    Returns:
        tuple: Dataset name, reference1 degree distribution, list of reference 2 or generated degree distributions.
    """
    dataset_name = os.path.splitext(os.path.basename(ref1_file))[0]

    ref1_degree_dist, ref2_or_gen_degree_dists = get_node_degree_distributions(ref1_file, ref2_or_gen_files, helper_dir,
                                                                               output_dir, name, reference,
                                                                               allowed_mismatches, indels)

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
                                  compairr_output_dir: str, name: str, dataset_split: str, allowed_mismatches: int,
                                  indels: bool) -> tuple:
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
        allowed_mismatches (int): Number of allowed mismatches for Compairr.
        indels (bool): Whether to allow indels in Compairr.
    Returns:
        tuple: Reference1 node degree distribution and list of generated node degree distributions or list of one
        reference 2 node degree distribution.
    """
    ref2_or_gen_degree_dists = []
    for file in ref2_or_gen_files:
        connectivity = compute_connectivity_with_compairr(file, compairr_output_helper_dir, compairr_output_dir, name,
                                                          allowed_mismatches, indels)
        degree_dist = get_node_degree_from_compairr_output(connectivity)
        ref2_or_gen_degree_dists.append(degree_dist)

    ref1_connectivity = compute_connectivity_with_compairr(ref1_file, compairr_output_helper_dir, compairr_output_dir,
                                                           dataset_split, allowed_mismatches, indels)
    ref1_degree_dist = get_node_degree_from_compairr_output(ref1_connectivity)

    return ref1_degree_dist, ref2_or_gen_degree_dists


def get_reference_divergence_scores(analysis_config: AnalysisConfig, compairr_output_helper_dir: str,
                                    compairr_output_dir: str) -> pd.DataFrame:
    """ Get the divergence score (JSD) between the train and test connectivity of each repertoire, as a reference for
    how similar two samples of the same repertoire are.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        compairr_output_helper_dir (str): Directory for Compairr helper files.
        compairr_output_dir (str): Directory for Compairr output files.
    Returns:
        pd.DataFrame: Columns repertoire and score, one row per repertoire.
    """
    rows = []
    for train_file, test_file in sorted(get_reference_files(analysis_config)):
        dataset_name, train_node_degree, test_node_degree = (
            get_connectivity_distributions_by_dataset(train_file, [test_file], compairr_output_helper_dir,
                                                      compairr_output_dir, DatasetSplit.TEST.value,
                                                      DatasetSplit.TRAIN.value, analysis_config.allowed_mismatches,
                                                      analysis_config.indels))
        divergence_scores = calculate_divergence_scores(train_node_degree, test_node_degree, DatasetSplit.TRAIN.value,
                                                        DatasetSplit.TEST.value)
        rows.append({"repertoire": dataset_name, "score": divergence_scores[0]})
    return pd.DataFrame(rows, columns=["repertoire", "score"])


def compute_connectivity_with_compairr(input_sequences_path: str, compairr_output_helper_dir: str,
                                       compairr_output_dir: str, dataset_type: str, allowed_mismatches: int,
                                       indels: bool) -> pd.DataFrame:
    """
    Compute connectivity using Compairr for a single dataset (either train, test, or generated).

    Args:
        input_sequences_path (str): Path to the input sequences file.
        compairr_output_helper_dir (str): Directory for Compairr helper files.
        compairr_output_dir (str): Directory for Compairr output files.
        dataset_type (str): Type of dataset (train or test or generated with a specific model).
        allowed_mismatches (int): Number of allowed mismatches for Compairr.
        indels (bool): Whether to allow indels in Compairr.

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
                           allowed_mismatches=allowed_mismatches, indels=indels)
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

    jsd = jensenshannon(p, q, base=2) ** 2

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


def summarize_and_plot_all(analysis_config: AnalysisConfig, scores_df: pd.DataFrame,
                           reference_scores: pd.DataFrame) -> None:
    """ Average the scores over the generated subsets of each repertoire, summarise them over repertoires and plot the
    grouped averages, with one dot per repertoire.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores_df (pd.DataFrame): One row per model, reference, dataset and generated subset, with the divergence in jsd.
        reference_scores (pd.DataFrame): Train vs. test scores per repertoire, with columns repertoire and score.
    Returns:
        None
    """
    mean_scores, std_scores, repertoire_scores, error_bar_caption = \
        summarise_scores_by_reference(scores_df, METRIC_NAME, analysis_config.donor_pattern)

    plot_grouped_avg_scores(
        analysis_config=analysis_config,
        mean_scores_by_ref=mean_scores,
        std_scores_by_ref=std_scores,
        distribution_type="connectivity",
        file_name="all_datasets_connectivity",
        scoring_method="JSD",
        reference_score=float(reference_scores["score"].mean()) if len(reference_scores) else None,
        error_bar_caption=error_bar_caption,
        repertoire_scores_by_ref=repertoire_scores
    )


def run_network_statistics(analysis_config: AnalysisConfig, scores_df: pd.DataFrame,
                           reference_scores: pd.DataFrame) -> None:
    """ Runs the repertoire-level comparisons against the test sequences and plots the comparisons with the train vs.
    test reference and the contrasts from the config. The train sequences are left out, since the models were fitted on
    them.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores_df (pd.DataFrame): One row per model, reference, dataset and generated subset, with the divergence in jsd.
        reference_scores (pd.DataFrame): Train vs. test scores per repertoire, with columns repertoire and score.
    Returns:
        None
    """
    if DatasetSplit.TEST.value not in analysis_config.reference_data:
        print("Skipping network statistics, since they are computed against the test sequences.")
        return

    test_scores = scores_df[scores_df["reference"] == DatasetSplit.TEST.value]
    output_prefix = f"{analysis_config.analysis_output_dir}/statistics/{METRIC_NAME}"
    repertoire_scores = aggregate_repertoire_scores(test_scores, "jsd", analysis_config.donor_pattern)
    results = run_repertoire_statistics(repertoire_scores, analysis_config.model_names, output_prefix,
                                        analysis_config.statistical_tests, reference_scores, analysis_config.contrasts,
                                        analysis_config.donor_pattern)
    plot_repertoire_statistics(results, output_prefix, "connectivity",
                               get_collection_specification_for_title(analysis_config.receptor_type,
                                                                      analysis_config.collection))


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
