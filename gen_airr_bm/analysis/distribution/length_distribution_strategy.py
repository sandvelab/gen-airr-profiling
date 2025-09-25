from collections import Counter

from scipy.spatial.distance import jensenshannon

from gen_airr_bm.analysis.distribution.base_distribution_strategy import BaseDistributionStrategy


class LengthDistributionStrategy(BaseDistributionStrategy):
    def compute_divergence(self, seqs1, seqs2):
        lengths1 = [len(seq) for seq in seqs1]
        lengths2 = [len(seq) for seq in seqs2]
        dist1 = Counter(lengths1)
        dist2 = Counter(lengths2)
        return [compute_jsd_length(dist1, dist2)]


def compute_jsd_length(dist1, dist2):
    all_lengths = set(dist1.keys()).union(set(dist2.keys()))
    p = [dist1.get(k, 0) for k in all_lengths]
    q = [dist2.get(k, 0) for k in all_lengths]
    return jensenshannon(p, q, base=2)
