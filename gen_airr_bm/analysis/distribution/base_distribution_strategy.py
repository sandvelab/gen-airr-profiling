from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.plotting_utils import plot_avg_scores


class BaseDistributionStrategy(ABC):
    # What the metrics of a distribution are, for plot footnotes, e.g. "sequence lengths"
    metric_group_name = "metrics"

    @abstractmethod
    def compute_divergence(self, gen_seqs: list[str], ref_seqs: list[str]) -> Any:
        pass

    def plot_distributions_per_dataset(self, analysis_config: AnalysisConfig, dataset_label: str, gen_seqs: list,
                                       ref_seqs: list, gen_label, ref_label) -> None:
        pass

    def get_metric_scores(self, divergence: Any) -> dict:
        """ Returns the output of compute_divergence as {metric: score}. Most distributions are compared as a whole,
        so they have the single metric "all". """
        return {"all": divergence[0]}

    def get_metric_label(self, distribution_type: str, metric) -> str:
        """ Returns the label of a metric as used in plot titles. """
        return distribution_type

    def get_file_stem(self, distribution_type: str, metric) -> str:
        """ Returns the start of the output file names of a metric. """
        return distribution_type.replace(" ", "")

    def get_sequences_from_file(self, file_path) -> list:
        """ Reads sequences from a file and returns them as a list. """
        data_df = pd.read_csv(file_path, sep='\t', usecols=["junction_aa"])
        return data_df["junction_aa"].tolist()

    #TODO: this can be removed. It is not used anywhere.
    def plot_scores(self, mean_scores: dict, std_scores: dict,
                    analysis_config: AnalysisConfig, distribution_type: str) -> None:
        file_name = f"{distribution_type}"
        plot_avg_scores(mean_scores, std_scores,
                        analysis_config.analysis_output_dir, analysis_config.reference_data,
                        file_name, distribution_type, "JSD")
