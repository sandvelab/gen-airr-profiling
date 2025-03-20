import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.plotting_utils import plot_jsd_scores


#TODO: We should look into it since it was fast coded with ChatGPT
def run_aa_distribution_analysis(analysis_config: AnalysisConfig):
    print(f"Analyzing AA distribution for {analysis_config}")

    mean_divergence_scores_dict = defaultdict(dict)
    std_divergence_scores_dict = defaultdict(dict)

    for model in analysis_config.model_names:
        aa_comparison_pairs = []

        generated_sequences_dir = f"{analysis_config.root_output_dir}/generated_sequences/{model}"
        reference_sequences_dir = f"{analysis_config.root_output_dir}/{analysis_config.reference_data}_sequences"

        # Generated sequences and train sequences files have same names
        generated_sequences_files = set(os.listdir(generated_sequences_dir))

        aa_comparison_pairs.extend([
            (os.path.join(generated_sequences_dir, file), os.path.join(reference_sequences_dir, file))
            for file in generated_sequences_files
        ])

        divergence_scores = defaultdict(list)
        for generated_file, reference_file in aa_comparison_pairs:
            generated_data_df = pd.read_csv(generated_file, sep='\t', usecols=["junction_aa"])
            reference_data_df = pd.read_csv(reference_file, sep='\t', usecols=["junction_aa"])

            generated_aas = generated_data_df["junction_aa"].tolist()
            reference_aas = reference_data_df["junction_aa"].tolist()

            for length in range(10, 21):
                generated_aas_length = [aa for aa in generated_aas if len(aa) == length]
                reference_aas_length = [aa for aa in reference_aas if len(aa) == length]

                generated_aa_dist = compute_positional_aa_dist(generated_aas_length)
                reference_aa_dist = compute_positional_aa_dist(reference_aas_length)

                divergence_scores[length].append(compute_jsd(generated_aa_dist, reference_aa_dist))

        for length, scores in divergence_scores.items():
            mean_divergence_scores_dict[length][model] = np.mean(scores)
            std_divergence_scores_dict[length][model] = np.std(scores)

    for length in range(10, 21):
        file_name = f"{analysis_config.analysis}_{length}.png"
        plot_jsd_scores(mean_divergence_scores_dict[length], std_divergence_scores_dict[length],
                        analysis_config.analysis_output_dir, analysis_config.reference_data, file_name,
                        f"aminoacid {length}")


def compute_positional_aa_dist(sequences):
    """Computes normalized amino acid frequency distribution at each position."""
    if not sequences:
        return {pos: {aa: 0.0 for aa in 'ACDEFGHIKLMNPQRSTVWY'} for pos in range(10, 21)}

    seq_length = len(sequences[0])
    aa_distribution = {pos: defaultdict(float) for pos in range(seq_length)}

    for pos in range(seq_length):
        aa_counts = defaultdict(int)
        total_count = len(sequences)

        for seq in sequences:
            aa_counts[seq[pos]] += 1

        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            aa_distribution[pos][aa] = aa_counts[aa] / total_count

    return aa_distribution


def compute_jsd(dist1, dist2, smooth=1e-10):
    """Computes Jensen-Shannon Divergence between two amino acid distributions across all positions (flattened)."""
    flattened_dist1 = defaultdict(float)
    flattened_dist2 = defaultdict(float)

    for pos in dist1:
        for aa, freq in dist1[pos].items():
            flattened_dist1[aa] += freq

    for pos in dist2:
        for aa, freq in dist2[pos].items():
            flattened_dist2[aa] += freq

    total1 = sum(flattened_dist1.values())
    total2 = sum(flattened_dist2.values())

    if total1 == 0 and total2 == 0:
        return 0.0

    p = {aa: (flattened_dist1[aa] / total1 if total1 > 0 else smooth) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    q = {aa: (flattened_dist2[aa] / total2 if total2 > 0 else smooth) for aa in 'ACDEFGHIKLMNPQRSTVWY'}

    p_vector = np.array([p[aa] for aa in 'ACDEFGHIKLMNPQRSTVWY'])
    q_vector = np.array([q[aa] for aa in 'ACDEFGHIKLMNPQRSTVWY'])

    return jensenshannon(p_vector, q_vector, base=2)
