import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.plotting_utils import plot_jsd_scores


#TODO: This function is too similar to the one in analyse_length_distribution.py, consider refactoring
def run_kmer_distribution_analysis(analysis_config: AnalysisConfig):
    print(f"Analyzing kmer distribution for {analysis_config}")

    mean_divergence_scores_dict = {}
    std_divergence_scores_dict = {}

    for model in analysis_config.model_names:
        kmer_comparison_pairs = []

        generated_sequences_dir = f"{analysis_config.root_output_dir}/generated_sequences/{model}"
        reference_sequences_dir = f"{analysis_config.root_output_dir}/{analysis_config.reference_data}_sequences"

        # Generated sequences and train sequences files have same names
        generated_sequences_files = set(os.listdir(generated_sequences_dir))

        kmer_comparison_pairs.extend([
            (os.path.join(generated_sequences_dir, file), os.path.join(reference_sequences_dir, file))
            for file in generated_sequences_files
        ])

        divergence_scores = []
        for generated_file, training_file in kmer_comparison_pairs:
            generated_data_df = pd.read_csv(generated_file, sep='\t', usecols=["junction_aa"])
            training_data_df = pd.read_csv(training_file, sep='\t', usecols=["junction_aa"])

            # At the moment we look only into 3 mers
            generated_kmers_dist = compute_kmer_distribution(generated_data_df["junction_aa"].tolist(), 3)
            reference_kmers_dist = compute_kmer_distribution(training_data_df["junction_aa"].tolist(), 3)

            divergence_scores.append(compute_jsd_kmers(generated_kmers_dist, reference_kmers_dist))

        mean_divergence_scores_dict[model] = np.mean(divergence_scores)
        std_divergence_scores_dict[model] = np.std(divergence_scores)

    plot_jsd_scores(mean_divergence_scores_dict, std_divergence_scores_dict, analysis_config.analysis_output_dir,
                        analysis_config.reference_data, analysis_config.analysis, "kmer")

def compute_kmer_distribution(sequences, k):
    """Computes normalized k-mer frequency distribution."""
    kmer_counts = Counter()
    total_kmers = 0

    for seq in sequences:
        kmers = extract_kmers(seq, k)
        kmer_counts.update(kmers)
        total_kmers += len(kmers)

    # Convert counts to probabilities
    return {kmer: count / total_kmers for kmer, count in kmer_counts.items()} if total_kmers > 0 else {}


def extract_kmers(sequence, k):
    """Extracts all k-mers from a given sequence."""
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def compute_jsd_kmers(dist1, dist2, smooth=1e-10):
    """Computes Jensen-Shannon Divergence between two k-mer distributions."""
    all_kmers = set(dist1.keys()).union(set(dist2.keys()))

    p = np.array([dist1.get(k, smooth) for k in all_kmers])
    q = np.array([dist2.get(k, smooth) for k in all_kmers])

    return jensenshannon(p, q, base=2)
