import os
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon

from gen_airr_bm.core.analysis_config import AnalysisConfig


def run_length_distribution_analysis(analysis_config: AnalysisConfig):
    print(f"Analyzing length distribution for {analysis_config}")

    divergence_scores_dict = {}

    for model in analysis_config.model_names:
        length_comparison_pairs = []

        generated_sequences_dir = f"{analysis_config.root_output_dir}/generated_sequences/{model}"
        training_sequences_dir = f"{analysis_config.root_output_dir}/train_sequences/{model}"

        # Generated sequences and train sequences files have same names
        generated_sequences_files = set(os.listdir(generated_sequences_dir))

        length_comparison_pairs.extend([
            (os.path.join(generated_sequences_dir, file), os.path.join(training_sequences_dir, file))
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

        divergence_scores_dict[model] = np.mean(divergence_scores)

    plot_jsd_scores(divergence_scores_dict, analysis_config.analysis_output_dir)


def compute_jsd(dist1, dist2):
    """Compute Jensen-Shannon Divergence"""
    all_lengths = set(dist1.keys()).union(set(dist2.keys()))
    p = [dist1.get(k, 0) for k in all_lengths]
    q = [dist2.get(k, 0) for k in all_lengths]

    return jensenshannon(p, q, base=2)


def plot_jsd_scores(divergence_scores_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    models, scores = zip(*sorted(divergence_scores_dict.items(), key=lambda x: x[1]))

    plt.figure(figsize=(10, 5))
    plt.bar(models, scores, color='skyblue')
    plt.xlabel("Models")
    plt.ylabel("Mean JSD for Length Distributions")
    plt.title("JSD Scores Comparing Length Distributions Across Models")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(f"{output_dir}/length_distribution.png")
