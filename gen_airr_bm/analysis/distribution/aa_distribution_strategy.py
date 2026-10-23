from collections import defaultdict

import numpy as np
from scipy.spatial.distance import jensenshannon

from gen_airr_bm.analysis.distribution.base_distribution_strategy import BaseDistributionStrategy


class AADistributionStrategy(BaseDistributionStrategy):
    metric_group_name = "sequence lengths"

    def compute_divergence(self, seqs1, seqs2):
        scores = defaultdict(list)
        for length in range(10, 21):
            filtered1 = [s for s in seqs1 if len(s) == length]
            filtered2 = [s for s in seqs2 if len(s) == length]
            dist1 = compute_positional_aa_dist(filtered1)
            dist2 = compute_positional_aa_dist(filtered2)
            scores[length].append(compute_jsd_aa(dist1, dist2))
        return scores

    def get_metric_scores(self, divergence):
        """ Returns one metric per sequence length, since the positional distributions are compared per length. """
        return {length: scores[0] for length, scores in divergence.items()}

    def get_metric_label(self, distribution_type, metric):
        return f"{distribution_type} (Seq. Len.: {metric})"

    def get_file_stem(self, distribution_type, metric):
        return f"{distribution_type.replace(' ', '')}_{metric}"


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

    return jensenshannon(np.array(p), np.array(q), base=2) ** 2
