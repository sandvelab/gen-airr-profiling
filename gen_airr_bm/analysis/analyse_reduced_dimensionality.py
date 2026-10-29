import os
from collections import defaultdict

import pandas as pd

from gen_airr_bm.analysis.distribution.base_distribution_strategy import BaseDistributionStrategy
from gen_airr_bm.analysis.distribution.distribution_factory import get_distribution_strategy
from gen_airr_bm.constants.dataset_split import DatasetSplit
from gen_airr_bm.constants.distribution_type import DistributionType
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_generated_sequences_dir_name, get_reference_files
from gen_airr_bm.utils.plotting_utils import (get_collection_specification_for_title, plot_grouped_avg_scores,
                                             plot_repertoire_statistics)
from gen_airr_bm.utils.statistics_utils import (aggregate_repertoire_scores, check_one_repertoire_per_donor,
                                               correct_across_metrics, run_repertoire_statistics,
                                               summarise_scores_by_reference, write_repertoire_statistics)


def run_reduced_dimensionality_analyses(analysis_config: AnalysisConfig) -> None:
    """ Runs reduced dimensionality analyses for all specified distribution types.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        None
    """
    for distribution_type in DistributionType:
        run_reduced_dimensionality_analysis(analysis_config, distribution_type)


def run_reduced_dimensionality_analysis(analysis_config: AnalysisConfig, distribution_type: DistributionType) -> None:
    """ Runs a single reduced dimensionality analysis for a specified distribution type. The divergence of every
    generated subset is averaged within its repertoire before the models are summarised over repertoires (see
    statistics_utils). Against the test sequences, the models are also compared at the repertoire level: with the
    train vs. test reference, with each other, and in the contrasts given in the config. These comparisons are
    descriptive, or include confidence intervals and tests if statistical_tests is set, which requires a donor_pattern
    and one repertoire per donor. Distributions with several metrics (the amino acid distribution per sequence length)
    are presented together, so their p-values are corrected jointly across the metrics.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        distribution_type (DistributionType): The type of distribution to analyze (e.g., 'length', 'kmer').
    Returns:
        None
    """
    print(f"Analyzing {distribution_type.value.lower()} distribution for {analysis_config}")
    os.makedirs(analysis_config.analysis_output_dir, exist_ok=True)
    if isinstance(analysis_config.reference_data, str):
        analysis_config.reference_data = [analysis_config.reference_data]

    strategy = get_distribution_strategy(distribution_type)
    distribution_name = distribution_type.value.lower()
    sequences_dict = defaultdict(lambda: defaultdict(list))

    scores_df = compute_divergence_scores(analysis_config, strategy, sequences_dict)
    scores_df.to_csv(f"{analysis_config.analysis_output_dir}/{distribution_name.replace(' ', '')}_jsd_scores.tsv",
                     sep="\t", index=False)
    if analysis_config.statistical_tests:
        check_one_repertoire_per_donor(aggregate_repertoire_scores(scores_df, "jsd", analysis_config.donor_pattern),
                                       analysis_config.donor_pattern)

    reference_scores_df = compute_train_test_reference_scores(analysis_config, strategy)
    metrics = scores_df["metric"].unique()
    statistics_by_metric = {}
    for metric in metrics:
        file_stem = strategy.get_file_stem(distribution_name, metric)
        metric_label = strategy.get_metric_label(distribution_name, metric)
        mean_scores_by_ref, std_scores_by_ref, repertoire_scores_by_ref, error_bar_caption = \
            summarise_scores_by_reference(scores_df, metric, analysis_config.donor_pattern)
        reference_scores = (reference_scores_df.loc[reference_scores_df["metric"] == metric, ["repertoire", "score"]]
                            if reference_scores_df is not None else None)
        reference_score = float(reference_scores["score"].mean()) if reference_scores is not None else None
        plot_grouped_avg_scores(analysis_config, mean_scores_by_ref, std_scores_by_ref, f"{file_stem}_grouped",
                                metric_label, "JSD", reference_score, error_bar_caption, repertoire_scores_by_ref)
        statistics = compute_reduced_dimensionality_statistics(analysis_config, scores_df, metric, file_stem,
                                                               reference_scores)
        if statistics is not None:
            statistics_by_metric[metric] = statistics

    correction_scope = "the comparisons in this plot only"
    if len(metrics) > 1:
        correct_across_metrics(statistics_by_metric)
        correction_scope = (f"the comparisons in this plot and the corresponding plots of the other "
                            f"{len(metrics) - 1} {strategy.metric_group_name}")
    for metric, statistics in statistics_by_metric.items():
        write_and_plot_reduced_dimensionality_statistics(analysis_config, statistics,
                                                         strategy.get_file_stem(distribution_name, metric),
                                                         strategy.get_metric_label(distribution_name, metric),
                                                         correction_scope)

    for dataset_label in sequences_dict.keys():
        for ref in analysis_config.reference_data:
            for model in analysis_config.model_names:
                strategy.plot_distributions_per_dataset(analysis_config,
                                                        dataset_label,
                                                        sequences_dict[dataset_label][model],
                                                        sequences_dict[dataset_label][ref],
                                                        model,
                                                        ref)


def compute_divergence_scores(analysis_config: AnalysisConfig, strategy: BaseDistributionStrategy,
                              sequences_dict: dict) -> pd.DataFrame:
    """ Computes the divergence between every generated subset and every reference dataset.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        strategy (BaseDistributionStrategy): The distribution strategy to use for divergence computation.
        sequences_dict (dict): Dictionary to store sequences by dataset and model or reference, for the per-dataset
            distribution plots.
    Returns:
        pd.DataFrame: One row per metric, model, reference, dataset and generated subset, with the divergence in jsd.
    """
    scores = []
    for model in analysis_config.model_names:
        for gen_file, ref_file, ref_label in get_sequence_file_pairs(analysis_config, model):
            divergence = process_reference(gen_file, ref_file, model, ref_label, strategy, sequences_dict)
            subset = int(os.path.splitext(os.path.basename(gen_file))[0].rsplit("_", 1)[1])
            dataset = os.path.splitext(os.path.basename(ref_file))[0]
            for metric, score in strategy.get_metric_scores(divergence).items():
                scores.append({"metric": metric, "model": model, "reference": ref_label, "dataset": dataset,
                               "subset": subset, "jsd": score})
    return pd.DataFrame(scores, columns=["metric", "model", "reference", "dataset", "subset", "jsd"])


def process_reference(gen_file: str, ref_file: str, gen_label: str, ref_label: str, strategy: BaseDistributionStrategy,
                      sequences_dict: dict):
    """ Processes a single reference: computes divergence scores between generated and reference sequences.
    Args:
        gen_file (str): Path to the generated sequences file.
        ref_file (str): Path to the reference sequences file.
        gen_label (str): Label for the generated data (e.g., model name).
        ref_label (str): Label for the reference (e.g., 'train', 'test').
        strategy (BaseDistributionStrategy): The distribution strategy to use for divergence computation.
        sequences_dict (dict): Dictionary to store sequences by model and reference.
    Returns:
        The divergence as returned by strategy.compute_divergence.
    """
    gen_seqs = strategy.get_sequences_from_file(gen_file)
    ref_seqs = strategy.get_sequences_from_file(ref_file)
    dataset_label = "_".join(os.path.basename(gen_file).split(".")[0].split("_")[:-1])
    # Generated sequences are the same for each reference, so only add once
    if gen_seqs not in sequences_dict[dataset_label][gen_label]:
        sequences_dict[dataset_label][gen_label].append(gen_seqs)
    if ref_label not in sequences_dict[dataset_label]:
        sequences_dict[dataset_label][ref_label] = [ref_seqs]

    return strategy.compute_divergence(gen_seqs, ref_seqs)


def compute_train_test_reference_scores(analysis_config: AnalysisConfig,
                                        strategy: BaseDistributionStrategy) -> pd.DataFrame | None:
    """ Computes the divergence between the test and train sequences of each repertoire, as a reference for how
    similar two samples of the same repertoire are.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        strategy (BaseDistributionStrategy): The distribution strategy to use for divergence computation.
    Returns:
        pd.DataFrame | None: Columns metric, repertoire and score, or None if train and test are not both used.
    """
    if not (DatasetSplit.TRAIN.value in analysis_config.reference_data
            and DatasetSplit.TEST.value in analysis_config.reference_data):
        return None

    rows = []
    for train_file, test_file in sorted(get_reference_files(analysis_config)):
        divergence = strategy.compute_divergence(strategy.get_sequences_from_file(test_file),
                                                 strategy.get_sequences_from_file(train_file))
        repertoire = os.path.splitext(os.path.basename(train_file))[0]
        rows.extend({"metric": metric, "repertoire": repertoire, "score": score}
                    for metric, score in strategy.get_metric_scores(divergence).items())
    return pd.DataFrame(rows, columns=["metric", "repertoire", "score"])


def compute_reduced_dimensionality_statistics(analysis_config: AnalysisConfig, scores_df: pd.DataFrame, metric,
                                              file_stem: str, reference_scores: pd.DataFrame | None) -> dict | None:
    """ Runs the repertoire-level comparisons of a metric against the test sequences, without writing them, so that
    they can first be corrected across metrics. The train sequences are left out, since the models were fitted on them.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores_df (pd.DataFrame): Scores as returned by compute_divergence_scores.
        metric: The metric to test.
        file_stem (str): Start of the output file names of the metric.
        reference_scores (pd.DataFrame | None): Train vs. test scores per repertoire.
    Returns:
        dict | None: The results as returned by run_repertoire_statistics, or None if test is not a reference.
    """
    if DatasetSplit.TEST.value not in analysis_config.reference_data:
        print("Skipping reduced dimensionality statistics, since they are computed against the test sequences.")
        return None

    test_scores = scores_df[(scores_df["metric"] == metric) & (scores_df["reference"] == DatasetSplit.TEST.value)]
    repertoire_scores = aggregate_repertoire_scores(test_scores, "jsd", analysis_config.donor_pattern)
    return run_repertoire_statistics(repertoire_scores, analysis_config.model_names,
                                     get_statistics_output_prefix(analysis_config, file_stem),
                                     analysis_config.statistical_tests, reference_scores, analysis_config.contrasts,
                                     analysis_config.donor_pattern, write=False)


def write_and_plot_reduced_dimensionality_statistics(analysis_config: AnalysisConfig, statistics: dict,
                                                     file_stem: str, metric_label: str,
                                                     correction_scope: str) -> None:
    """ Writes the repertoire-level comparisons of a metric and plots the comparisons with the train vs. test
    reference and the contrasts from the config.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        statistics (dict): The results as returned by compute_reduced_dimensionality_statistics.
        file_stem (str): Start of the output file names of the metric.
        metric_label (str): Label of the metric as used in plot titles.
        correction_scope (str): The comparisons the p-values were corrected across, for the plot footnote.
    Returns:
        None
    """
    output_prefix = get_statistics_output_prefix(analysis_config, file_stem)
    write_repertoire_statistics(statistics, output_prefix)

    plot_repertoire_statistics(statistics, output_prefix, metric_label,
                               get_collection_specification_for_title(analysis_config.receptor_type,
                                                                      analysis_config.collection),
                               correction_scope)


def get_statistics_output_prefix(analysis_config: AnalysisConfig, file_stem: str) -> str:
    """ Returns the path prefix of the statistics output files of a metric. """
    return f"{analysis_config.analysis_output_dir}/statistics/{file_stem}"


# TODO: Refactor to use get_sequence_files from file_utils.py
def get_sequence_file_pairs(analysis_config: AnalysisConfig, model: str) -> list:
    """ Gets pairs of generated and reference sequence files for comparison.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        model (str): The name of the generative model to analyze.
    Returns:
        list: A list of tuples, each containing (generated_file_path, reference_file_path, reference_label).
    """
    comparison_pairs = []
    sequences_dir_name = get_generated_sequences_dir_name(analysis_config)
    gen_dir = f"{analysis_config.root_output_dir}/{sequences_dir_name}/{model}"
    gen_files = sorted(os.listdir(gen_dir))
    gen_files = [file for file in gen_files if
                 int(os.path.splitext(file)[0].split('_')[-1]) < analysis_config.n_subsets]

    if isinstance(analysis_config.reference_data, str):
        analysis_config.reference_data = [analysis_config.reference_data]

    for reference in analysis_config.reference_data:
        ref_dir = f"{analysis_config.root_output_dir}/{reference}_sequences"
        for file in gen_files:
            file_name, file_ext = os.path.splitext(file)
            file_base = file_name.rsplit("_", 1)[0]
            ref_file = file_base + file_ext

            gen_file_path = os.path.join(gen_dir, file)
            ref_file_path = os.path.join(ref_dir, ref_file)
            if os.path.exists(ref_file_path):
                comparison_pairs.append((gen_file_path, ref_file_path, reference))
    return comparison_pairs
