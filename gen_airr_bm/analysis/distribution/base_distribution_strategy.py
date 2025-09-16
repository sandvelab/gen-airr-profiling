from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_reference_files
from gen_airr_bm.utils.plotting_utils import plot_avg_scores, plot_grouped_avg_scores


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

    def get_mean_reference_score(self, analysis_config: AnalysisConfig) -> float | None:
        if "train" in analysis_config.reference_data and "test" in analysis_config.reference_data:
            ref_scores = []
            reference_comparison_files = get_reference_files(analysis_config)
            for train_file, test_file in reference_comparison_files:
                train_seqs = self.get_sequences_from_file(train_file)
                test_seqs = self.get_sequences_from_file(test_file[0])
                ref_scores.extend(self.compute_divergence(test_seqs, train_seqs))
            return np.mean(ref_scores)
        else:
            return None

    def get_sequences_from_file(self, file_path) -> list:
        """ Reads sequences from a file and returns them as a list. """
        data_df = pd.read_csv(file_path, sep='\t', usecols=["junction_aa"])
        return data_df["junction_aa"].tolist()

    def plot_scores(self, mean_scores: dict, std_scores: dict,
                    analysis_config: AnalysisConfig, distribution_type: str) -> None:
        file_name = f"{distribution_type}.png"
        plot_avg_scores(mean_scores, std_scores,
                        analysis_config.analysis_output_dir, analysis_config.reference_data,
                        file_name, distribution_type, "JSD")

    def plot_scores_by_reference(self, mean_scores_by_ref: dict, std_scores_by_ref: dict,
                                 analysis_config: AnalysisConfig, distribution_type: str,
                                 mean_reference_score: float | None) -> None:
        """
        mean_scores_by_ref: {ref_label: {model: mean_score}}
        std_scores_by_ref: {ref_label: {model: std_score}}
        """
        file_name = f"{distribution_type}_grouped.png"
        plot_grouped_avg_scores(mean_scores_by_ref, std_scores_by_ref,
                                analysis_config.analysis_output_dir, analysis_config.reference_data,
                                file_name, distribution_type, "JSD", mean_reference_score)
