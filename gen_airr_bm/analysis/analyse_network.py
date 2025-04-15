import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from gen_airr_bm.constants.dataset_split import DatasetSplit
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.plotting_utils import plot_jsd_scores, plot_degree_distribution, plot_diversity_bar_chart
from gen_airr_bm.utils.compairr_utils import deduplicate_and_merge_two_datasets, run_compairr_existence, \
    run_compairr_cluster, deduplicate_single_dataset


def run_network_analysis(analysis_config: AnalysisConfig):
    print("Running network analysis")

    output_dir = analysis_config.analysis_output_dir
    compairr_output_helper_dir = f"{output_dir}/compairr_helper_files"
    compairr_output_dir = f"{output_dir}/compairr_output"

    for directory in [output_dir, compairr_output_helper_dir, compairr_output_dir]:
        os.makedirs(directory, exist_ok=True)

    compute_and_plot_diversity_scores(analysis_config, compairr_output_dir)

    # calculate connectivity scores
    compute_and_plot_connectivity_scores(analysis_config, compairr_output_dir, compairr_output_helper_dir)


def compute_and_plot_diversity_scores(analysis_config: AnalysisConfig, compairr_output_dir: str):
    def collect_diversity_scores(dir_path, label):
        scores = []
        for file in set(os.listdir(dir_path)):
            path = os.path.join(dir_path, file)
            name = os.path.splitext(file)[0]
            scores.append(compute_diversity(path, compairr_output_dir, f"{name}_{label}_clustering"))
        return scores

    mean_diversity, std_diversity = {}, {}

    for model in analysis_config.model_names:
        dir_path = f"{analysis_config.root_output_dir}/generated_compairr_sequences/{model}"
        scores = collect_diversity_scores(dir_path, model)
        mean_diversity[model], std_diversity[model] = np.mean(scores), np.std(scores)

    for label in ["train", "test"]:
        dir_path = f"{analysis_config.root_output_dir}/{label}_compairr_sequences"
        scores = collect_diversity_scores(dir_path, label)
        mean_diversity[label], std_diversity[label] = np.mean(scores), np.std(scores)

    plot_diversity_bar_chart(mean_diversity, std_diversity, f"{analysis_config.analysis_output_dir}/diversity.png")


def compute_and_plot_connectivity_scores(analysis_config: AnalysisConfig, compairr_output_dir: str,
                                         compairr_output_helper_dir: str):
    dataset_splits = DatasetSplit.train_and_test()
    mean_scores = {split: {} for split in dataset_splits}
    std_scores = {split: {} for split in dataset_splits}
    for model in analysis_config.model_names:
        comparison_sets = get_sequence_file_sets(analysis_config, model)
        divergence_scores = {split: [] for split in dataset_splits}

        for gen_file, train_file, test_file in comparison_sets:
            dataset_name = os.path.splitext(os.path.basename(gen_file))[0]
            files = {
                DatasetSplit.TRAIN: train_file,
                DatasetSplit.TEST: test_file,
                model: gen_file
            }

            compairr_connectivity_results = {
                label: compute_compairr_connectivity(f, compairr_output_helper_dir, compairr_output_dir, label)
                for label, f in files.items()
            }

            degree_distributions = {label: get_node_degree_distribution(result)
                                    for label, result in compairr_connectivity_results.items()}

            for split in dataset_splits:
                plot_degree_distribution(degree_distributions[model], degree_distributions[split],
                                         analysis_config.analysis_output_dir, model, split.value, dataset_name)
                divergence_scores[split].extend(compute_divergence(degree_distributions[model],
                                                                   degree_distributions[split]))

        for split in dataset_splits:
            mean_scores[split][model] = np.mean(divergence_scores[split])
            std_scores[split][model] = np.std(divergence_scores[split])

    for split in dataset_splits:
        plot_connectivity_scores(mean_scores[split], std_scores[split], analysis_config.analysis_output_dir,
                                 split.value, "connectivity")


def compute_diversity(data_file, compairr_output_dir, file_name):
    run_compairr_cluster(compairr_output_dir, data_file, file_name)
    compairr_cluster_result = pd.read_csv(f"{compairr_output_dir}/{file_name}.tsv", sep='\t')
    cluster_counts = compairr_cluster_result["#cluster_no"].value_counts()
    probabilities = cluster_counts / cluster_counts.sum()
    shannon_entropy = entropy(probabilities)

    return shannon_entropy


def compute_compairr_connectivity(input_sequences_path, compairr_output_helper_dir, compairr_output_dir, dataset_type):
    dataset_type = dataset_type.value if isinstance(dataset_type, DatasetSplit) else dataset_type
    file_name = f"{os.path.splitext(os.path.basename(input_sequences_path))[0]}_{dataset_type}"
    unique_sequences_path = f"{compairr_output_helper_dir}/{file_name}_unique.tsv"
    deduplicate_single_dataset(input_sequences_path, unique_sequences_path)
    run_compairr_existence(compairr_output_dir, unique_sequences_path, unique_sequences_path, file_name)
    compairr_result = pd.read_csv(f"{compairr_output_dir}/{file_name}_overlap.tsv", sep='\t',
                                  names=['sequence_id', 'overlap_count'], header=0)
    return compairr_result


def get_node_degree_distribution(compairr_result):
    compairr_result['overlap_count'] -= 1
    node_degree_distribution = compairr_result['overlap_count'].value_counts()
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


def plot_connectivity_scores(mean_scores: dict, std_scores: dict, output_dir: str, reference_data: str,
                             distribution_type: str) -> None:
    """Plot the mean and standard deviation of the divergence scores."""
    file_name = f"{distribution_type}.png"
    plot_jsd_scores(mean_scores, std_scores, output_dir, reference_data,
                    file_name, distribution_type)
