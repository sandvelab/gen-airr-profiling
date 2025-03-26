import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.plotting_utils import plot_jsd_scores, plot_degree_distribution, plot_diversity_bar_chart
from gen_airr_bm.utils.compairr_utils import process_and_save_sequences, run_compairr_existence, run_compairr_cluster


def run_network_analysis(analysis_config: AnalysisConfig):
    print("Running network analysis")

    output_dir = analysis_config.analysis_output_dir
    compairr_output_helper_dir = f"{output_dir}/compairr_helper_files"
    compairr_output_dir = f"{output_dir}/compairr_output"

    for directory in [output_dir, compairr_output_helper_dir, compairr_output_dir]:
        os.makedirs(directory, exist_ok=True)

    # calculate diversity scores
    mean_diversity = defaultdict(dict)
    std_diversity = defaultdict(dict)

    for model in analysis_config.model_names:
        diversity_scores = []
        gen_dir = f"{analysis_config.root_output_dir}/generated_compairr_sequences/{model}"
        gen_files = [os.path.join(gen_dir, file) for file in set(os.listdir(gen_dir))]
        for file in gen_files:
            dataset_name = os.path.splitext(os.path.basename(file))[0]
            diversity_scores.append(
                compute_diversity(file, compairr_output_dir, f"{dataset_name}_{model}_clustering"))
        mean_diversity[model] = np.mean(diversity_scores)
        std_diversity[model] = np.std(diversity_scores)

    for label in ["train", "test"]:
        diversity_scores = []
        dir = f"{analysis_config.root_output_dir}/{label}_compairr_sequences"
        files = [os.path.join(dir, file) for file in set(os.listdir(dir))]
        for file in files:
            dataset_name = os.path.splitext(os.path.basename(file))[0]
            diversity_scores.append(
                compute_diversity(file, compairr_output_dir, f"{dataset_name}_{label}_clustering"))
        mean_diversity[label] = np.mean(diversity_scores)
        std_diversity[label] = np.std(diversity_scores)

    plot_diversity_bar_chart(mean_diversity, std_diversity, f"{output_dir}/diversity.png")

    # calculate connectivity scores
    mean_scores_train, std_scores_train = {}, {}
    mean_scores_test, std_scores_test = {}, {}
    for model in analysis_config.model_names:
        comparison_sets = get_sequence_file_sets(analysis_config, model)

        divergence_scores_train, divergence_scores_test = [], []

        for gen_file, train_file, test_file in comparison_sets:
            file_label_pairs = list(zip((gen_file, train_file, test_file), (model, "train", "test")))
            dataset_name = os.path.splitext(os.path.basename(gen_file))[0]

            # compute and collect JS divergence
            divergence_score_train, divergence_score_test = compute_and_plot_degree_distribution(file_label_pairs,
                                                                                                  compairr_output_helper_dir,
                                                                                                  compairr_output_dir,
                                                                                                  output_dir,
                                                                                                  model,
                                                                                                  dataset_name)
            divergence_scores_train.extend(divergence_score_train)
            divergence_scores_test.extend(divergence_score_test)

        mean_scores_train[model] = np.mean(divergence_scores_train)
        std_scores_train[model] = np.std(divergence_scores_train)

        mean_scores_test[model] = np.mean(divergence_scores_test)
        std_scores_test[model] = np.std(divergence_scores_test)

    # plot the mean and standard deviation of the divergence scores
    plot_scores(mean_scores_train, std_scores_train, analysis_config.analysis_output_dir, "train", "connectivity")
    plot_scores(mean_scores_test, std_scores_test, analysis_config.analysis_output_dir, "test", "connectivity")


# TODO: code from run_network_analysis was quickly dumped here, needs to be refactored
def compute_and_plot_degree_distribution(file_label_pairs, compairr_output_helper_dir, compairr_output_dir, output_dir, model, dataset_name):
    gen_compairr_result, train_compairr_result, test_compairr_result = [
        compute_compairr_results(file, compairr_output_helper_dir, compairr_output_dir, label)
        for file, label in file_label_pairs
    ]

    gen_degree_dist, train_degree_dist, test_degree_dist = map(get_node_degree_distribution,
                                                               [gen_compairr_result,
                                                                train_compairr_result,
                                                                test_compairr_result])

    plot_degree_distribution(gen_degree_dist, train_degree_dist, output_dir, model,
                             "train", dataset_name)
    plot_degree_distribution(gen_degree_dist, test_degree_dist, output_dir, model,
                             "test", dataset_name)

    return compute_divergence(gen_degree_dist, train_degree_dist), compute_divergence(gen_degree_dist, test_degree_dist)


def compute_diversity(data_file, compairr_output_dir, file_name):
    run_compairr_cluster(compairr_output_dir, data_file, file_name)
    compairr_cluster_result = pd.read_csv(f"{compairr_output_dir}/{file_name}.tsv", sep='\t')
    cluster_counts = compairr_cluster_result["#cluster_no"].value_counts()
    probabilities = cluster_counts / cluster_counts.sum()
    shannon_entropy = entropy(probabilities)

    return shannon_entropy


def compute_compairr_results(input_sequences_path, compairr_output_helper_dir, compairr_output_dir, dataset_type):
    file_name = f"{os.path.splitext(os.path.basename(input_sequences_path))[0]}_{dataset_type}"
    unique_sequences_path = f"{compairr_output_helper_dir}/{file_name}_unique.tsv"
    concat_sequences_path = f"{compairr_output_helper_dir}/{file_name}_concat.tsv"
    process_and_save_sequences(input_sequences_path, input_sequences_path, unique_sequences_path, concat_sequences_path)
    run_compairr_existence(compairr_output_dir, unique_sequences_path, concat_sequences_path, file_name)
    compairr_result = pd.read_csv(f"{compairr_output_dir}/{file_name}_overlap.tsv", sep='\t')
    return compairr_result


def get_node_degree_distribution(compairr_result):
    compairr_result['dataset_1'] -= 1
    node_degree_distribution = compairr_result['dataset_1'].value_counts()
    return node_degree_distribution


def get_sequence_file_sets(analysis_config: AnalysisConfig, model: str):
    comparison_sets = []

    gen_dir = f"{analysis_config.root_output_dir}/generated_compairr_sequences/{model}"
    train_dir = f"{analysis_config.root_output_dir}/train_compairr_sequences"
    test_dir = f"{analysis_config.root_output_dir}/test_compairr_sequences"

    gen_files = set(os.listdir(gen_dir))

    comparison_sets.extend([
        (os.path.join(gen_dir, file), os.path.join(train_dir, file), os.path.join(test_dir, file))
        for file in gen_files
    ])

    return comparison_sets


def compute_divergence(gen_node_degree_distribution, ref_node_degree_distribution):
    """Compute divergence between two node degree distributions."""
    merged_df = pd.merge(gen_node_degree_distribution, ref_node_degree_distribution, how='outer',
                         suffixes=('_gen', '_ref'),
                         left_index=True, right_index=True).fillna(0)

    p = merged_df['count_gen'] / merged_df['count_gen'].sum()
    q = merged_df['count_ref'] / merged_df['count_ref'].sum()

    jsd = jensenshannon(p, q, base=2)

    return [jsd]


def plot_scores(mean_scores: dict, std_scores: dict, output_dir: str, reference_data: str,
                distribution_type: str) -> None:
    """Plot the mean and standard deviation of the divergence scores."""
    file_name = f"{distribution_type}.png"
    plot_jsd_scores(mean_scores, std_scores, output_dir, reference_data,
                    file_name, distribution_type)
