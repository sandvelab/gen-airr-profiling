from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.plotting_utils import plot_avg_scores


class BaseDistributionStrategy(ABC):
    @abstractmethod
    def compute_divergence(self, gen_seqs: list[str], ref_seqs: list[str]) -> Any:
        pass

    def init_mean_std_scores(self) -> tuple[dict, dict]:
        return {}, {}

    def init_divergence_scores(self) -> list:
        return []

    def update_divergence_scores(self, divergence_scores: list[float], new_scores: list[float]) -> None:
        divergence_scores.extend(new_scores)

    def update_mean_std_scores(self, divergence_scores: list[float], model_name: str,
                               mean_scores: dict, std_scores: dict) -> None:
        mean_scores[model_name] = np.mean(divergence_scores)
        std_scores[model_name] = np.std(divergence_scores)

    def plot_scores(self, mean_scores: dict, std_scores: dict,
                    analysis_config: AnalysisConfig, distribution_type: str) -> None:
        file_name = f"{distribution_type}.png"
        plot_avg_scores(mean_scores, std_scores,
                        analysis_config.analysis_output_dir, analysis_config.reference_data,
                        file_name, distribution_type, "JSD")
