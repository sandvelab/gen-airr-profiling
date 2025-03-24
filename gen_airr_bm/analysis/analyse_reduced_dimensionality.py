import os

import pandas as pd

from gen_airr_bm.analysis.distribution.distribution_factory import get_distribution_strategy
from gen_airr_bm.constants.distribution_type import DistributionType
from gen_airr_bm.core.analysis_config import AnalysisConfig


def run_reduced_dimensionality_analyses(analysis_config: AnalysisConfig):
    for distribution_type in DistributionType:
        run_reduced_dimensionality_analysis(analysis_config, distribution_type)


def run_reduced_dimensionality_analysis(analysis_config: AnalysisConfig, distribution_type: DistributionType):
    print(f"Analyzing {distribution_type.value.lower()} distribution for {analysis_config}")
    strategy = get_distribution_strategy(distribution_type)

    mean_divergence_scores_dict, std_divergence_scores_dict = strategy.init_mean_std_scores()

    for model in analysis_config.model_names:
        comparison_pairs = get_sequence_file_pairs(analysis_config, model)

        divergence_scores = strategy.init_divergence_scores()

        for gen_file, ref_file in comparison_pairs:
            ref_seqs, gen_seqs = map(get_sequences_from_file, [ref_file, gen_file])

            scores = strategy.compute_divergence(gen_seqs, ref_seqs)
            strategy.update_divergence_scores(divergence_scores, scores)

        strategy.update_mean_std_scores(divergence_scores, model,
                                        mean_divergence_scores_dict, std_divergence_scores_dict)

    strategy.plot_scores(mean_divergence_scores_dict, std_divergence_scores_dict,
                         analysis_config, distribution_type.value.lower())


def get_sequence_file_pairs(analysis_config: AnalysisConfig, model: str):
    comparison_pairs = []

    gen_dir = f"{analysis_config.root_output_dir}/generated_sequences/{model}"
    ref_dir = f"{analysis_config.root_output_dir}/{analysis_config.reference_data}_sequences"

    gen_files = set(os.listdir(gen_dir))

    comparison_pairs.extend([
        (os.path.join(gen_dir, file), os.path.join(ref_dir, file))
        for file in gen_files
    ])

    return comparison_pairs


def get_sequences_from_file(file_path):
    data_df = pd.read_csv(file_path, sep='\t', usecols=["junction_aa"])
    return data_df["junction_aa"].tolist()
