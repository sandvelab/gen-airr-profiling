from collections import Counter

from scipy.spatial.distance import jensenshannon

from gen_airr_bm.analysis.distribution.base_strategy import DistributionStrategy


class LengthDistributionStrategy(DistributionStrategy):
    def compute_divergence(self, gen_seqs, ref_seqs):
        gen_lengths = [len(seq) for seq in gen_seqs]
        ref_lengths = [len(seq) for seq in ref_seqs]
        gen_dist = Counter(gen_lengths)
        ref_dist = Counter(ref_lengths)
        return [compute_jsd_length(gen_dist, ref_dist)]


def compute_jsd_length(dist1, dist2):
    all_lengths = set(dist1.keys()).union(set(dist2.keys()))
    p = [dist1.get(k, 0) for k in all_lengths]
    q = [dist2.get(k, 0) for k in all_lengths]
    return jensenshannon(p, q, base=2)
