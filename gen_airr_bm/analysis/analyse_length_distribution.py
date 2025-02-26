import os
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon

from gen_airr_bm.core.analysis_config import AnalysisConfig


def run_length_distribution_analysis(analysis_config: AnalysisConfig):
    print(f"Analyzing length distribution for {analysis_config}")

    mean_divergence_scores_dict = {}
    std_divergence_scores_dict = {}

    for model in analysis_config.model_names:
        length_comparison_pairs = []

        generated_sequences_dir = f"{analysis_config.root_output_dir}/generated_sequences/{model}"
        reference_sequences_dir = f"{analysis_config.root_output_dir}/{analysis_config.reference_data}_sequences/{model}"

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
                    analysis_config.reference_data)


def compute_jsd(dist1, dist2):
    """Compute Jensen-Shannon Divergence"""
    all_lengths = set(dist1.keys()).union(set(dist2.keys()))
    p = [dist1.get(k, 0) for k in all_lengths]
    q = [dist2.get(k, 0) for k in all_lengths]

    return jensenshannon(p, q, base=2)


def plot_jsd_scores(mean_divergence_scores_dict, std_divergence_scores_dict, output_dir, refernce_data):
    os.makedirs(output_dir, exist_ok=True)
    models, scores = zip(*sorted(mean_divergence_scores_dict.items(), key=lambda x: x[1]))
    errors = [std_divergence_scores_dict[model] for model in models]

    plt.figure(figsize=(10, 5))
    plt.bar(
        models, scores,
        yerr=errors,
        capsize=8,
        color='skyblue',
        alpha=0.7,
        error_kw={'capsize': 5, 'capthick': 1, 'elinewidth': 1}
    )
    plt.xlabel("Models")
    plt.ylabel("Mean JSD for Length Distributions")
    plt.title(f"JSD Scores Comparing Length Distributions Across Models and {refernce_data.capitalize()} Data")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(f"{output_dir}/length_distribution.png")
