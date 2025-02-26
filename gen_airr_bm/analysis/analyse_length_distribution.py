import os
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy.spatial.distance import jensenshannon

from gen_airr_bm.core.analysis_config import AnalysisConfig


def run_length_distribution_analysis(analysis_config: AnalysisConfig):
    print(f"Analyzing length distribution for {analysis_config}")

    mean_divergence_scores_dict = {}
    std_divergence_scores_dict = {}

    for model in analysis_config.model_names:
        length_comparison_pairs = []

        generated_sequences_dir = f"{analysis_config.root_output_dir}/generated_sequences/{model}"
        reference_sequences_dir = f"{analysis_config.root_output_dir}/{analysis_config.reference_data}_sequences"

        # Generated sequences and train sequences files have same names
        generated_sequences_files = set(os.listdir(generated_sequences_dir))

        length_comparison_pairs.extend([
            (os.path.join(generated_sequences_dir, file), os.path.join(reference_sequences_dir, file))
            for file in generated_sequences_files
        ])

        divergence_scores = []
        for generated_file, training_file in length_comparison_pairs:
            generated_data_df = pd.read_csv(generated_file, sep='\t', usecols=["junction_aa"])
            training_data_df = pd.read_csv(training_file, sep='\t', usecols=["junction_aa"])

            generated_lengths = generated_data_df["junction_aa"].apply(len).tolist()
            training_lengths = training_data_df["junction_aa"].apply(len).tolist()

            generated_length_dist = Counter(generated_lengths)
            training_length_dist = Counter(training_lengths)

            divergence_scores.append(compute_jsd(generated_length_dist, training_length_dist))

        mean_divergence_scores_dict[model] = np.mean(divergence_scores)
        std_divergence_scores_dict[model] = np.std(divergence_scores)

    plot_jsd_scores(mean_divergence_scores_dict, std_divergence_scores_dict, analysis_config.analysis_output_dir,
                    analysis_config.reference_data, analysis_config.analysis)


def compute_jsd(dist1, dist2):
    """Compute Jensen-Shannon Divergence"""
    all_lengths = set(dist1.keys()).union(set(dist2.keys()))
    p = [dist1.get(k, 0) for k in all_lengths]
    q = [dist2.get(k, 0) for k in all_lengths]

    return jensenshannon(p, q, base=2)


def plot_jsd_scores(mean_divergence_scores_dict, std_divergence_scores_dict, output_dir, reference_data, analysis_type):
    file_name = f"{analysis_type}.png"
    fig_dir = os.path.join(output_dir, reference_data)
    os.makedirs(fig_dir, exist_ok=True)

    models, scores = zip(*sorted(mean_divergence_scores_dict.items(), key=lambda x: x[1]))
    errors = [std_divergence_scores_dict[model] for model in models]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=models,
        y=scores,
        error_y=dict(type='data', array=errors, visible=True),
        marker=dict(color='skyblue'),
    ))

    fig.update_layout(
        title=f"JSD Scores Comparing Length Distributions Across Models and {reference_data.capitalize()} Data",
        xaxis_title="Models",
        yaxis_title="Mean JSD for Length Distributions",
        xaxis_tickangle=-45,
        template="plotly_white"
    )

    png_path = os.path.join(fig_dir, file_name)
    fig.write_image(png_path)

    print(f"Plot saved as PNG at: {png_path}")
