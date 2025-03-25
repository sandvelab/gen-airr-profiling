from collections import Counter

import numpy as np
from scipy.spatial.distance import jensenshannon

from gen_airr_bm.analysis.distribution.base_distribution_strategy import BaseDistributionStrategy


class KmerDistributionStrategy(BaseDistributionStrategy):
    def __init__(self, k):
        self.k = k

    def compute_divergence(self, gen_seqs, ref_seqs):
        gen_dist = compute_kmer_distribution(gen_seqs, self.k)
        ref_dist = compute_kmer_distribution(ref_seqs, self.k)
        return [compute_jsd_kmers(gen_dist, ref_dist)]


def compute_kmer_distribution(sequences, k):
    counter = Counter()
    total = 0
    for seq in sequences:
        kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
        counter.update(kmers)
        total += len(kmers)
    return {kmer: count / total for kmer, count in counter.items()} if total > 0 else {}


def compute_jsd_kmers(dist1, dist2, smooth=1e-10):
    keys = set(dist1) | set(dist2)
    p = np.array([dist1.get(k, smooth) for k in keys])
    q = np.array([dist2.get(k, smooth) for k in keys])
    return jensenshannon(p, q, base=2)
