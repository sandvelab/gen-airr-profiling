from abc import ABC, abstractmethod

import numpy as np

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.plotting_utils import plot_jsd_scores


class DistributionStrategy(ABC):
    @abstractmethod
    def compute_divergence(self, gen_seqs, ref_seqs):
        pass

    def init_mean_std_scores(self):
        return {}, {}

    def init_divergence_scores(self):
        return []

    def update_divergence_scores(self, divergence_scores, new_scores):
        divergence_scores.extend(new_scores)

    def update_mean_std_scores(self, divergence_scores, model_name, mean_scores, std_scores):
        mean_scores[model_name] = np.mean(divergence_scores)
        std_scores[model_name] = np.std(divergence_scores)

    def plot_scores(self, mean_scores, std_scores, analysis_config: AnalysisConfig, distribution_type):
        file_name = f"{distribution_type}.png"
        plot_jsd_scores(mean_scores, std_scores,
                        analysis_config.analysis_output_dir, analysis_config.reference_data,
                        file_name, distribution_type)
