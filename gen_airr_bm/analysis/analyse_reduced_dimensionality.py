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
        divergence_scores_by_ref = process_model(analysis_config, model, strategy)
        update_mean_std_scores_by_reference(divergence_scores_by_ref,
                                            mean_divergence_scores_dict,
                                            std_divergence_scores_dict,
                                            strategy,
                                            model)

    strategy.plot_scores_by_reference(mean_divergence_scores_dict, std_divergence_scores_dict,
                                      analysis_config, distribution_type.value.lower())


def process_model(analysis_config: AnalysisConfig, model: str, strategy):
    """
    Processes a single model: computes divergence scores for each reference.
    Returns a dict mapping ref_label to divergence scores.
    """
    comparison_pairs = get_sequence_file_pairs(analysis_config, model)
    divergence_scores_by_ref = {}

    for gen_file, ref_file, ref_label in comparison_pairs:
        scores = process_reference(gen_file, ref_file, ref_label, strategy, divergence_scores_by_ref)
        strategy.update_divergence_scores(divergence_scores_by_ref[ref_label], scores)

    return divergence_scores_by_ref


def process_reference(gen_file, ref_file, ref_label, strategy, divergence_scores_by_ref):
    """
    Processes a single reference: computes divergence scores and updates the dict.
    """
    gen_seqs = get_sequences_from_file(gen_file)
    ref_seqs = get_sequences_from_file(ref_file)
    scores = strategy.compute_divergence(gen_seqs, ref_seqs)
    if ref_label not in divergence_scores_by_ref:
        divergence_scores_by_ref[ref_label] = strategy.init_divergence_scores()
    strategy.update_divergence_scores(divergence_scores_by_ref[ref_label], scores)
    return scores


def update_mean_std_scores_by_reference(divergence_scores_by_ref, mean_dict, std_dict, strategy, model):
    """
    Updates mean and std dictionaries for each reference label.
    """
    for ref_label, scores in divergence_scores_by_ref.items():
        if ref_label not in mean_dict:
            mean_dict[ref_label], std_dict[ref_label] = strategy.init_mean_std_scores()
        strategy.update_mean_std_scores(scores, model, mean_dict[ref_label], std_dict[ref_label])


def get_sequence_file_pairs(analysis_config: AnalysisConfig, model: str):
    """
    Returns a list of (gen_file, ref_file, ref_label) tuples.
    ref_label is the name of the reference (e.g. 'train', 'test').
    """
    comparison_pairs = []
    gen_dir = f"{analysis_config.root_output_dir}/generated_sequences/{model}"
    gen_files = set(os.listdir(gen_dir))

    if isinstance(analysis_config.reference_data, str):
        analysis_config.reference_data = [analysis_config.reference_data]

    for reference in analysis_config.reference_data:
        ref_dir = f"{analysis_config.root_output_dir}/{reference}_sequences"
        for file in gen_files:
            gen_file_path = os.path.join(gen_dir, file)
            ref_file_path = os.path.join(ref_dir, file)
            if os.path.exists(ref_file_path):
                comparison_pairs.append((gen_file_path, ref_file_path, reference))
    return comparison_pairs


def get_sequences_from_file(file_path):
    data_df = pd.read_csv(file_path, sep='\t', usecols=["junction_aa"])
    return data_df["junction_aa"].tolist()
