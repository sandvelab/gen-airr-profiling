import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_sequence_files
from gen_airr_bm.utils.plotting_utils import plot_avg_scores, plot_degree_distribution, plot_diversity_bar_chart
from gen_airr_bm.utils.compairr_utils import run_compairr_existence, run_compairr_cluster, deduplicate_single_dataset


def run_network_analysis(analysis_config: AnalysisConfig):
    """Run network analysis to compute connectivity plots.

    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.

    Returns:
        None"""
    print("Running network analysis")

    output_dir = analysis_config.analysis_output_dir
    compairr_output_helper_dir = f"{output_dir}/compairr_helper_files"
    compairr_output_dir = f"{output_dir}/compairr_output"

    for directory in [output_dir, compairr_output_helper_dir, compairr_output_dir]:
        os.makedirs(directory, exist_ok=True)

    compute_and_plot_connectivity_scores(analysis_config, compairr_output_dir, compairr_output_helper_dir)


def compute_and_plot_connectivity_scores(analysis_config: AnalysisConfig, compairr_output_dir: str,
                                         compairr_output_helper_dir: str):
    reference_data = analysis_config.reference_data
    mean_scores = defaultdict(lambda: defaultdict(list))
    std_scores = defaultdict(lambda: defaultdict(list))
    for model in analysis_config.model_names:
        comparison_files_dir = get_sequence_files(analysis_config, model, reference_data)

        for ref_file, gen_files in comparison_files_dir.items():
            divergence_scores = []
            dataset_name = os.path.splitext(os.path.basename(ref_file))[0]

            ref_degree_dist, gen_degree_dists = run_compairr_and_get_degree_distributions(ref_file, gen_files,
                                                                                          compairr_output_helper_dir,
                                                                                          compairr_output_dir,
                                                                                          model, reference_data)

            plot_degree_distribution(ref_degree_dist, gen_degree_dists, analysis_config.analysis_output_dir, model,
                                     reference_data, dataset_name)

            for gen_degree_dist in gen_degree_dists:
                divergence_scores.extend(compute_divergence(gen_degree_dist, ref_degree_dist))

            mean_scores[dataset_name][model] = np.mean(divergence_scores)
            std_scores[dataset_name][model] = np.std(divergence_scores)

    for dataset in mean_scores:
        plot_connectivity_scores(mean_scores[dataset], std_scores[dataset], analysis_config.analysis_output_dir,
                                 reference_data, "connectivity", f"{dataset}_connectivity.png")


def run_compairr_and_get_degree_distributions(ref_file, gen_files, compairr_output_helper_dir, compairr_output_dir,
                                              model, ref_name):
    gen_degree_dists = []
    for gen_file in gen_files:
        gen_connectivity = compute_compairr_connectivity(gen_file, compairr_output_helper_dir, compairr_output_dir,
                                                         model)
        gen_degree_dist = get_node_degree_distribution(gen_connectivity)
        gen_degree_dists.append(gen_degree_dist)

    ref_connectivity = compute_compairr_connectivity(ref_file, compairr_output_helper_dir, compairr_output_dir,
                                                     ref_name)
    ref_degree_dist = get_node_degree_distribution(ref_connectivity)

    return ref_degree_dist, gen_degree_dists


def compute_diversity(data_file, compairr_output_dir, file_name):
    run_compairr_cluster(compairr_output_dir, data_file, file_name)
    compairr_cluster_result = pd.read_csv(f"{compairr_output_dir}/{file_name}.tsv", sep='\t')
    cluster_counts = compairr_cluster_result["#cluster_no"].value_counts()
    probabilities = cluster_counts / cluster_counts.sum()
    shannon_entropy = entropy(probabilities)

    return shannon_entropy


def compute_compairr_connectivity(input_sequences_path, compairr_output_helper_dir, compairr_output_dir, dataset_type):
    file_name = f"{os.path.splitext(os.path.basename(input_sequences_path))[0]}_{dataset_type}"
    unique_sequences_path = f"{compairr_output_helper_dir}/{file_name}_unique.tsv"
    if os.path.exists(unique_sequences_path):
        print(f"Unique sequences already exist for {file_name}. Skipping execution.")
    else:
        deduplicate_single_dataset(input_sequences_path, unique_sequences_path)
    run_compairr_existence(compairr_output_dir, unique_sequences_path, unique_sequences_path, file_name)
    compairr_result = pd.read_csv(f"{compairr_output_dir}/{file_name}_overlap.tsv", sep='\t',
                                  names=['sequence_id', 'overlap_count'], header=0)
    return compairr_result


def get_node_degree_distribution(compairr_result):
    compairr_result['overlap_count'] -= 1
    node_degree_distribution = compairr_result['overlap_count'].value_counts()
    return node_degree_distribution


def compute_divergence(gen_node_degree_distribution, ref_node_degree_distribution):
    """Compute divergence between two node degree distributions."""
    merged_df = pd.merge(gen_node_degree_distribution, ref_node_degree_distribution, how='outer',
                         suffixes=('_gen', '_ref'),
                         left_index=True, right_index=True).fillna(0)

    p = merged_df['count_gen'] / merged_df['count_gen'].sum()
    q = merged_df['count_ref'] / merged_df['count_ref'].sum()

    jsd = jensenshannon(p, q, base=2)

    return [jsd]


def plot_connectivity_scores(mean_scores: dict, std_scores: dict, output_dir: str, reference_data: str,
                             distribution_type: str, file_name: str) -> None:
    """Plot the mean and standard deviation of the divergence scores."""
    plot_avg_scores(mean_scores, std_scores, output_dir, reference_data,
                    file_name, distribution_type, scoring_method="JSD")
