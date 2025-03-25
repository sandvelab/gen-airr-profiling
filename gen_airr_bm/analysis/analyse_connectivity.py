import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
import plotly.graph_objects as go

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.plotting_utils import plot_jsd_scores
from gen_airr_bm.utils.compairr_utils import process_and_save_sequences, run_compairr


def run_connectivity_analysis(analysis_config: AnalysisConfig):
    print("Running connectivity analysis")

    output_dir = analysis_config.analysis_output_dir
    compairr_output_helper_dir = f"{output_dir}/compairr_helper_files"
    compairr_output_dir = f"{output_dir}/compairr_output"

    for directory in [output_dir, compairr_output_helper_dir, compairr_output_dir]:
        os.makedirs(directory, exist_ok=True)

    mean_divergence_scores_dict, std_divergence_scores_dict = {}, {}

    for model in analysis_config.model_names:
        comparison_pairs = get_sequence_file_pairs(analysis_config, model)

        divergence_scores = []

        for gen_file, ref_file in comparison_pairs:

            gen_compairr_result = compute_compairr_results(gen_file, compairr_output_helper_dir, compairr_output_dir, model)
            ref_compairr_result = compute_compairr_results(ref_file, compairr_output_helper_dir, compairr_output_dir,
                                                           analysis_config.reference_data)

            gen_node_degree_distribution, ref_node_degree_distribution = map(get_node_degree_distribution,
                                                                             [gen_compairr_result, ref_compairr_result])

            # plot histogram of each node degree distribution
            dataset_name = os.path.splitext(os.path.basename(gen_file))[0]
            plot_histograms(gen_node_degree_distribution, ref_node_degree_distribution, output_dir, model,
                            analysis_config.reference_data, dataset_name)

            # compute and collect JS divergence
            score = compute_divergence(gen_node_degree_distribution, ref_node_degree_distribution)
            divergence_scores.extend(score)

        mean_divergence_scores_dict[model] = np.mean(divergence_scores)
        std_divergence_scores_dict[model] = np.std(divergence_scores)

    # plot the mean and standard deviation of the divergence scores
    plot_scores(mean_divergence_scores_dict, std_divergence_scores_dict, analysis_config, "connectivity")


def compute_compairr_results(input_sequences_path, compairr_output_helper_dir, compairr_output_dir, dataset_type):
    file_name = f"{os.path.splitext(os.path.basename(input_sequences_path))[0]}_{dataset_type}"
    unique_sequences_path = f"{compairr_output_helper_dir}/{file_name}_unique.tsv"
    concat_sequences_path = f"{compairr_output_helper_dir}/{file_name}_concat.tsv"
    process_and_save_sequences(input_sequences_path, input_sequences_path, unique_sequences_path, concat_sequences_path)
    run_compairr(compairr_output_dir, unique_sequences_path, concat_sequences_path, file_name)
    compairr_result = pd.read_csv(f"{compairr_output_dir}/{file_name}_overlap.tsv", sep='\t')
    return compairr_result


def get_node_degree_distribution(compairr_result):
    compairr_result['dataset_1'] -= 1
    node_degree_distribution = compairr_result['dataset_1'].value_counts()
    return node_degree_distribution


def get_sequence_file_pairs(analysis_config: AnalysisConfig, model: str):
    comparison_pairs = []

    gen_dir = f"{analysis_config.root_output_dir}/generated_compairr_sequences/{model}"
    ref_dir = f"{analysis_config.root_output_dir}/{analysis_config.reference_data}_compairr_sequences"

    gen_files = set(os.listdir(gen_dir))

    comparison_pairs.extend([
        (os.path.join(gen_dir, file), os.path.join(ref_dir, file))
        for file in gen_files
    ])

    return comparison_pairs


def compute_divergence(gen_node_degree_distribution, ref_node_degree_distribution):
    """Compute divergence between two node degree distributions."""
    merged_df = pd.merge(gen_node_degree_distribution, ref_node_degree_distribution, how='outer',
                         suffixes=('_gen', '_ref'),
                         left_index=True, right_index=True).fillna(0)

    p = merged_df['count_gen'] / merged_df['count_gen'].sum()
    q = merged_df['count_ref'] / merged_df['count_ref'].sum()

    jsd = jensenshannon(p, q, base=2)

    return [jsd]


def plot_scores(mean_scores: dict, std_scores: dict,
                analysis_config: AnalysisConfig, distribution_type: str) -> None:
    """Plot the mean and standard deviation of the divergence scores."""
    file_name = f"{distribution_type}.png"
    plot_jsd_scores(mean_scores, std_scores,
                    analysis_config.analysis_output_dir, analysis_config.reference_data,
                    file_name, distribution_type)


def plot_histograms(gen_node_degree_distribution, ref_node_degree_distribution, output_dir, model_name, reference_data,
                    dataset_name):
    """Plot histograms of the two node degree distributions in one plot."""
    fig_dir = os.path.join(output_dir, reference_data)
    os.makedirs(fig_dir, exist_ok=True)

    merged_df = pd.merge(gen_node_degree_distribution, ref_node_degree_distribution, how='outer',
                         suffixes=('_gen', '_ref'),
                         left_index=True, right_index=True).fillna(0)

    merged_df["freq_gen"] = merged_df["count_gen"] / merged_df["count_gen"].sum()
    merged_df["freq_ref"] = merged_df["count_ref"] / merged_df["count_ref"].sum()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=merged_df.index,
        y=merged_df["freq_gen"],
        name=model_name,
        marker=dict(color='skyblue')
    ))

    fig.add_trace(go.Bar(
        x=merged_df.index,
        y=merged_df["freq_ref"],
        name=reference_data,
        marker=dict(color='orange')
    ))

    fig.update_layout(
        title="Comparison of Connectivity Distributions",
        xaxis_title="Number of neighbors",
        yaxis_title="Frequency",
        barmode="group"
    )

    png_path = f"{fig_dir}/histogram_{dataset_name}_{model_name}_{reference_data}.png"
    fig.write_image(png_path)

    print(f"Plot saved as PNG at: {png_path}")





