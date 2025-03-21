from collections import defaultdict

import numpy as np
from scipy.spatial.distance import jensenshannon

from gen_airr_bm.analysis.distribution.base_strategy import DistributionStrategy


class AADistributionStrategy(DistributionStrategy):
    def compute_divergence(self, gen_seqs, ref_seqs):
        scores = defaultdict(list)
        for length in range(10, 21):
            gen_filtered = [s for s in gen_seqs if len(s) == length]
            ref_filtered = [s for s in ref_seqs if len(s) == length]
            gen_dist = compute_positional_aa_dist(gen_filtered)
            ref_dist = compute_positional_aa_dist(ref_filtered)
            scores[length].append(compute_jsd_aa(gen_dist, ref_dist))
        return scores


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
