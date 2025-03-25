from collections import defaultdict

import numpy as np
from scipy.spatial.distance import jensenshannon

from gen_airr_bm.analysis.distribution.base_distribution_strategy import BaseDistributionStrategy
from gen_airr_bm.utils.plotting_utils import plot_jsd_scores


class AADistributionStrategy(BaseDistributionStrategy):
    def compute_divergence(self, gen_seqs, ref_seqs):
        scores = defaultdict(list)
        for length in range(10, 21):
            gen_filtered = [s for s in gen_seqs if len(s) == length]
            ref_filtered = [s for s in ref_seqs if len(s) == length]
            gen_dist = compute_positional_aa_dist(gen_filtered)
            ref_dist = compute_positional_aa_dist(ref_filtered)
            scores[length].append(compute_jsd_aa(gen_dist, ref_dist))
        return scores

    def init_mean_std_scores(self):
        return defaultdict(dict), defaultdict(dict)

    def init_divergence_scores(self):
        return defaultdict(list)

    def update_divergence_scores(self, divergence_scores, new_scores):
        for length, value in new_scores.items():
            divergence_scores[length].extend(value)

    def update_mean_std_scores(self, divergence_scores, model_name, mean_scores, std_scores):
        for length, scores in divergence_scores.items():
            mean_scores[length][model_name] = np.mean(scores)
            std_scores[length][model_name] = np.std(scores)

    def plot_scores(self, mean_scores, std_scores, analysis_config, distribution_type):
        for length in range(10, 21):
            file_name = f"{distribution_type}_{length}.png"
            plot_jsd_scores(mean_scores[length], std_scores[length],
                            analysis_config.analysis_output_dir, analysis_config.reference_data, file_name,
                            f"{distribution_type} {length}")


def compute_positional_aa_dist(sequences):
    if not sequences:
        return {pos: {aa: 0.0 for aa in 'ACDEFGHIKLMNPQRSTVWY'} for pos in range(10, 21)}

    length = len(sequences[0])
    dist = {pos: defaultdict(float) for pos in range(length)}

    for pos in range(length):
        counts = defaultdict(int)
        for seq in sequences:
            counts[seq[pos]] += 1
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            dist[pos][aa] = counts[aa] / len(sequences)

    return dist


def compute_jsd_aa(dist1, dist2, smooth=1e-10):
    flat1 = defaultdict(float)
    flat2 = defaultdict(float)

    for pos in dist1:
        for aa, val in dist1[pos].items():
            flat1[aa] += val
    for pos in dist2:
        for aa, val in dist2[pos].items():
            flat2[aa] += val

    total1 = sum(flat1.values())
    total2 = sum(flat2.values())

    if total1 == 0 and total2 == 0:
        return 0.0

    p = [flat1[aa] / total1 if total1 > 0 else smooth for aa in 'ACDEFGHIKLMNPQRSTVWY']
    q = [flat2[aa] / total2 if total2 > 0 else smooth for aa in 'ACDEFGHIKLMNPQRSTVWY']

    return jensenshannon(np.array(p), np.array(q), base=2)
