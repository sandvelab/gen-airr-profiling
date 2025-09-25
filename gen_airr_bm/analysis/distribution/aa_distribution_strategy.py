from collections import defaultdict

import numpy as np
from scipy.spatial.distance import jensenshannon

from gen_airr_bm.analysis.distribution.base_distribution_strategy import BaseDistributionStrategy
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_reference_files
from gen_airr_bm.utils.plotting_utils import plot_grouped_avg_scores


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
            if length not in divergence_scores:
                divergence_scores[length] = []
            divergence_scores[length].extend(value)

    def update_mean_std_scores(self, divergence_scores, model_name, mean_scores, std_scores):
        for length, scores in divergence_scores.items():
            if length not in mean_scores:
                mean_scores[length] = {}
                std_scores[length] = {}
            mean_scores[length][model_name] = np.mean(scores)
            std_scores[length][model_name] = np.std(scores)

    def get_mean_reference_score(self, analysis_config: AnalysisConfig) -> list[float] | None:
        if "train" in analysis_config.reference_data and "test" in analysis_config.reference_data:
            ref_scores = []
            reference_comparison_files = get_reference_files(analysis_config)
            for train_file, test_file in reference_comparison_files:
                train_seqs = self.get_sequences_from_file(train_file)
                test_seqs = self.get_sequences_from_file(test_file)
                ref_scores.append(self.compute_divergence(test_seqs, train_seqs))

            mean_scores_by_length = {}
            for scores in ref_scores:
                for length, vals in scores.items():
                    if length not in mean_scores_by_length:
                        mean_scores_by_length[length] = []
                    mean_scores_by_length[length].extend(vals)
            all_means = [np.mean(vals) for vals in mean_scores_by_length.values() if vals]
            return all_means
        else:
            return None

    def plot_scores_by_reference(self, mean_scores_by_ref, std_scores_by_ref,
                                 analysis_config, distribution_type, mean_reference_scores):
        for length in range(10, 21):
            mean_by_ref = {ref: mean_scores_by_ref[ref].get(length, {}) for ref in mean_scores_by_ref}
            std_by_ref = {ref: std_scores_by_ref[ref].get(length, {}) for ref in std_scores_by_ref}
            file_name = f"{distribution_type}_{length}_grouped"
            mean_reference_score = mean_reference_scores[length-10] if mean_reference_scores else None
            plot_grouped_avg_scores(mean_by_ref, std_by_ref,
                                    analysis_config.analysis_output_dir, analysis_config.reference_data,
                                    file_name, f"{distribution_type} {length}", "JSD",
                                    mean_reference_score)


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
