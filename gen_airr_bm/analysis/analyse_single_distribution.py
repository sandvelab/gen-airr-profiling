import os
from collections import defaultdict

import numpy as np
import pandas as pd

from gen_airr_bm.analysis.distribution.factory import get_distribution_strategy
from gen_airr_bm.constants.distribution_type import DistributionType
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.plotting_utils import plot_jsd_scores


def run_single_distribution_analyses(analysis_config: AnalysisConfig):
    for distribution_type in DistributionType:
        run_single_distribution_analysis(analysis_config, distribution_type)


def run_single_distribution_analysis(analysis_config: AnalysisConfig, distribution_type: DistributionType):
    print(f"Analyzing {distribution_type.name.lower()} distribution for {analysis_config}")
    strategy = get_distribution_strategy(distribution_type, 3)

    mean_divergence_scores_dict = defaultdict(dict) if distribution_type == DistributionType.AA else {}
    std_divergence_scores_dict = defaultdict(dict) if distribution_type == DistributionType.AA else {}

    for model in analysis_config.model_names:
        comparison_pairs = []

        gen_dir = f"{analysis_config.root_output_dir}/generated_sequences/{model}"
        ref_dir = f"{analysis_config.root_output_dir}/{analysis_config.reference_data}_sequences"

        gen_files = set(os.listdir(gen_dir))

        comparison_pairs.extend([
            (os.path.join(gen_dir, file), os.path.join(ref_dir, file))
            for file in gen_files
        ])

        divergence_scores = defaultdict(list) if distribution_type == DistributionType.AA else []

        for gen_file, ref_file in comparison_pairs:
            gen_data_df = pd.read_csv(gen_file, sep='\t', usecols=["junction_aa"])
            ref_data_df = pd.read_csv(ref_file, sep='\t', usecols=["junction_aa"])

            gen_seqs = gen_data_df["junction_aa"].tolist()
            ref_seqs = ref_data_df["junction_aa"].tolist()

            scores = strategy.compute_divergence(gen_seqs, ref_seqs)
            if distribution_type == DistributionType.AA:
                for length, value in scores.items():
                    divergence_scores[length].extend(value)
            else:
                divergence_scores.extend(scores)

        if distribution_type == DistributionType.AA:
            for length, scores in divergence_scores.items():
                mean_divergence_scores_dict[length][model] = np.mean(scores)
                std_divergence_scores_dict[length][model] = np.std(scores)
        else:
            mean_divergence_scores_dict[model] = np.mean(divergence_scores)
            std_divergence_scores_dict[model] = np.std(divergence_scores)

    if distribution_type == DistributionType.AA:
        for length in range(10, 21):
            file_name = f"{distribution_type.name.lower()}_{length}.png"
            plot_jsd_scores(mean_divergence_scores_dict[length], std_divergence_scores_dict[length],
                            analysis_config.analysis_output_dir, analysis_config.reference_data, file_name,
                            f"aminoacid {length}")
    else:
        file_name = f"{distribution_type.name.lower()}.png"
        plot_jsd_scores(mean_divergence_scores_dict, std_divergence_scores_dict,
                        analysis_config.analysis_output_dir, analysis_config.reference_data, file_name, "kmer")
