import os
import numpy as np
import pandas as pd

from collections import defaultdict
from dataclasses import dataclass, field
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_sequence_files, get_subset_index
from gen_airr_bm.utils.compairr_utils import run_compairr_existence, run_sequence_deduplication
from gen_airr_bm.utils.plotting_utils import plot_grouped_bar_precision_recall, plot_avg_innovation_scores, \
    get_collection_specification_for_title, plot_repertoire_statistics, title_case
from gen_airr_bm.utils.statistics_utils import aggregate_repertoire_scores, check_one_repertoire_per_donor, \
    run_repertoire_statistics, summarise_scores_by_reference

METRICS = ["precision", "recall"]
SCORE_COLUMNS = ["metric", "model", "reference", "dataset", "subset", "score"]


@dataclass
class PrecisionRecallScores:
    """ Class to store precision and recall scores for different models and datasets. """
    mean_precision: dict = field(default_factory=lambda: defaultdict(dict))
    std_precision: dict = field(default_factory=lambda: defaultdict(dict))
    mean_recall: dict = field(default_factory=lambda: defaultdict(dict))
    std_recall: dict = field(default_factory=lambda: defaultdict(dict))
    precision_all: dict = field(default_factory=lambda: defaultdict(dict))
    recall_all: dict = field(default_factory=lambda: defaultdict(dict))
    scores: list = field(default_factory=list)


def run_precision_recall_analysis(analysis_config: AnalysisConfig) -> None:
    """ Runs precision recall analysis on the generated and reference sequences.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        None
    """
    print("Running precision recall analysis")

    output_dir = analysis_config.analysis_output_dir
    compairr_output_dir = f"{output_dir}/compairr_output"

    for directory in [output_dir, compairr_output_dir]:
        os.makedirs(directory, exist_ok=True)

    compute_and_plot_precision_recall_scores(analysis_config, compairr_output_dir)


def compute_and_plot_precision_recall_scores(analysis_config: AnalysisConfig, compairr_output_dir: str) -> None:
    """ Compute precision and recall scores and plot them. The scores of every generated subset are averaged within its
    repertoire before the models are summarised over repertoires (see statistics_utils). The models are also compared
    at the repertoire level: with each other and in the contrasts given in the config. These comparisons are
    descriptive, or include confidence intervals and tests if statistical_tests is set, which requires a donor_pattern
    and one repertoire per donor.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        compairr_output_dir (str): Directory to store CompAIRR output files.
    Returns:
        None
    """
    train_reference = 'train' if 'train' in analysis_config.reference_data else None
    test_reference = 'test' if 'test' in analysis_config.reference_data else None
    if not test_reference:
        raise ValueError("Could not run precision recall analysis without test data. 'test' required in reference_data "
                         "list.")
    else:
        print(f"Continuing running precision recall analysis using {test_reference} as reference data...")

    scores = PrecisionRecallScores()

    for model in analysis_config.model_names:
        collect_model_scores(analysis_config, model, test_reference, compairr_output_dir, scores)

    scores_df = pd.DataFrame(scores.scores, columns=SCORE_COLUMNS)
    scores_df.to_csv(f"{analysis_config.analysis_output_dir}/precision_recall_scores.tsv", sep="\t", index=False)
    if analysis_config.statistical_tests:
        check_one_repertoire_per_donor(aggregate_repertoire_scores(scores_df, "score", analysis_config.donor_pattern),
                                       analysis_config.donor_pattern)

    plot_precision_recall_scores(analysis_config, scores, test_reference, scores_df)
    run_precision_recall_statistics(analysis_config, scores_df)


def collect_model_scores(analysis_config: AnalysisConfig, model: str, test_reference: str, compairr_output_dir: str,
                         scores: PrecisionRecallScores) -> None:
    """ Collect precision and recall scores for a given model.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        model (str): Name of the model to analyze.
        test_reference (str): Reference dataset for testing.
        compairr_output_dir (str): Directory to store CompAIRR output files.
        scores (PrecisionRecallScores): Storage for precision and recall scores.
    Returns:
        None
    """
    comparison_files_dir = get_sequence_files(analysis_config, model, test_reference)

    for ref_file, gen_files in comparison_files_dir.items():
        dataset_name = os.path.splitext(os.path.basename(ref_file))[0]

        precision_scores, recall_scores = get_precision_recall_scores(analysis_config, ref_file, gen_files,
                                                                      compairr_output_dir, model)

        mean_p, std_p = np.mean(precision_scores), np.std(precision_scores)
        mean_r, std_r = np.mean(recall_scores), np.std(recall_scores)

        scores.mean_precision[dataset_name][model] = mean_p
        scores.std_precision[dataset_name][model] = std_p
        scores.mean_recall[dataset_name][model] = mean_r
        scores.std_recall[dataset_name][model] = std_r

        scores.precision_all[dataset_name][model] = precision_scores
        scores.recall_all[dataset_name][model] = recall_scores

        for gen_file, precision, recall in zip(gen_files, precision_scores, recall_scores):
            for metric, score in [("precision", precision), ("recall", recall)]:
                scores.scores.append({"metric": metric, "model": model, "reference": test_reference,
                                      "dataset": dataset_name, "subset": get_subset_index(gen_file), "score": score})


def add_upper_reference(analysis_config: AnalysisConfig, train_reference: str, test_reference: str,
                        scores: PrecisionRecallScores, compairr_output_dir: str) -> None:
    """ Add upper reference precision/recall scores between train and test data.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        train_reference (str): Reference dataset train to compare with test data.
        test_reference (str): Reference dataset test to compare with train data.
        scores (PrecisionRecallScores): Storage for precision and recall scores.
        compairr_output_dir (str): Directory to store CompAIRR output files.
    Returns:
        None
    """
    test_dir = f"{analysis_config.root_output_dir}/{test_reference}_compairr_sequences"
    train_dir = f"{analysis_config.root_output_dir}/{train_reference}_compairr_sequences"

    for dataset in scores.mean_precision.keys():
        train_file = f"{train_dir}/{dataset}.tsv"
        test_file = f"{test_dir}/{dataset}.tsv"

        ref_precision, ref_recall = get_precision_recall_reference(analysis_config, train_file, test_file,
                                                                   compairr_output_dir)

        scores.precision_all[dataset]["upper_reference"] = [ref_precision]
        scores.recall_all[dataset]["upper_reference"] = [ref_recall]


def get_precision_recall_scores(analysis_config: AnalysisConfig, ref_file: str, gen_files: list,
                                compairr_output_dir: str, model: str) -> tuple:
    """ Get precision and recall scores for the generated files compared to the reference file.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        ref_file (str): Path to the reference file.
        gen_files (list): List of paths to generated files.
        compairr_output_dir (str): Directory to store CompAIRR output files.
        model (str): Name of the model used for generation.
    Returns:
        tuple: Lists of precision and recall scores for the generated files.
    """
    precision_scores, recall_scores = [], []
    for gen_file in gen_files:
        precision = compute_compairr_overlap_ratio(analysis_config, gen_file, ref_file, compairr_output_dir,
                                                   model, "precision")
        recall = compute_compairr_overlap_ratio(analysis_config, ref_file, gen_file, compairr_output_dir,
                                                model, "recall")

        precision_scores.append(precision)
        recall_scores.append(recall)

    return precision_scores, recall_scores


def get_precision_recall_reference(analysis_config: AnalysisConfig, train_file: str, test_file: str,
                                   compairr_output_dir: str) -> tuple:
    """ Get precision and recall scores for the upper reference between train and test files.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        train_file (str): Path to the train file (train used as replacement of generated (model) file for upper
        reference.
        test_file (str): Path to the test file.
        compairr_output_dir (str): Directory to store CompAIRR output files.
    Returns:
        tuple: Precision and recall scores for the upper reference.
    """
    precision = compute_compairr_overlap_ratio(analysis_config, train_file, test_file, compairr_output_dir,
                                               'upper_reference', "precision")
    recall = compute_compairr_overlap_ratio(analysis_config, test_file, train_file, compairr_output_dir,
                                            'upper_reference', "recall")
    return precision, recall


def compute_compairr_overlap_ratio(analysis_config: AnalysisConfig, search_for_file: str, search_in_file: str,
                                   compairr_output_dir: str, name: str, metric: str) -> float:
    """ Compute the overlap ratio between two sequence sets using CompAIRR for precision or recall.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        search_for_file (str): Path to the file of sequences for which to search for existence in another sequence set.
        search_in_file (str): Path to the file to search for existence in.
        compairr_output_dir (str): Directory to store CompAIRR output files.
        name (str): Name of the model used for generation, or "upper_reference" for the upper reference.
        metric (str): Metric type, either "precision" or "recall".
    Returns:
        float: Ratio of non-zero overlap counts to total counts.
    """
    if metric == "precision":
        file_name = f"{os.path.splitext(os.path.basename(search_for_file))[0]}_{name}_{metric}"
    else:
        file_name = f"{os.path.splitext(os.path.basename(search_in_file))[0]}_{name}_{metric}"

    if analysis_config.deduplicate:
        search_for_file, search_in_file = run_sequence_deduplication(analysis_config, search_for_file, search_in_file)

    run_compairr_existence(compairr_output_dir, search_for_file, search_in_file, file_name,
                           allowed_mismatches=analysis_config.allowed_mismatches, indels=analysis_config.indels)
    compairr_result = pd.read_csv(f"{compairr_output_dir}/{file_name}_overlap.tsv", sep='\t',
                                  names=['sequence_id', 'overlap_count'], header=0)
    n_nonzero_rows = compairr_result[(compairr_result['overlap_count'] != 0)].shape[0]
    ratio = n_nonzero_rows / len(compairr_result)

    return ratio


#TODO: Refactor recall plotting hack
def plot_precision_recall_scores(analysis_config: AnalysisConfig, scores: PrecisionRecallScores,
                                 test_reference: str, scores_df: pd.DataFrame) -> None:
    """ Plot precision and recall scores for each dataset and model, and the recall of each model over repertoires.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores (PrecisionRecallScores): Storage class for precision and recall scores.
        test_reference (str): Reference dataset for testing.
        scores_df (pd.DataFrame): One row per metric, model, dataset and generated subset, with the score in score.
    Returns:
        None
    """
    for dataset in scores.mean_precision:
        plot_avg_innovation_scores(analysis_config, scores.mean_precision[dataset], scores.std_precision[dataset],
                        analysis_config.analysis_output_dir, "precision",
                        f"{dataset}_precision", "precision",
                        scoring_method="precision")

        plot_avg_innovation_scores(analysis_config, scores.mean_recall[dataset], scores.std_recall[dataset],
                        analysis_config.analysis_output_dir, "recall",
                        f"{dataset}_recall", "recall",
                        scoring_method="recall")

    plot_grouped_bar_precision_recall(scores.precision_all, scores.recall_all,
                                      analysis_config.analysis_output_dir, test_reference,
                                      analysis_config.receptor_type, analysis_config.allowed_mismatches)

    mean_recall, std_recall, repertoire_recall, error_bar_caption = summarise_scores_by_reference(
        scores_df, "recall", analysis_config.donor_pattern, score_column="score")
    if test_reference not in mean_recall:
        return

    plot_avg_innovation_scores(analysis_config, mean_recall[test_reference], std_recall[test_reference],
                               analysis_config.analysis_output_dir, "recall",
                               f"recall_{analysis_config.receptor_type}", "recall",
                               scoring_method="recall", error_bar_caption=error_bar_caption,
                               repertoire_scores=repertoire_recall[test_reference])


def run_precision_recall_statistics(analysis_config: AnalysisConfig, scores_df: pd.DataFrame) -> None:
    """ Runs the repertoire-level comparisons of the precision and recall against the test sequences and plots the
    contrasts from the config. There is no train vs. test reference, since the upper reference is no longer computed.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores_df (pd.DataFrame): One row per metric, model, dataset and generated subset, with the score in score.
    Returns:
        None
    """
    collection_specification = get_collection_specification_for_title(analysis_config.receptor_type,
                                                                     analysis_config.collection)
    for metric in METRICS:
        output_prefix = f"{analysis_config.analysis_output_dir}/statistics/{metric}"
        repertoire_scores = aggregate_repertoire_scores(scores_df[scores_df["metric"] == metric], "score",
                                                        analysis_config.donor_pattern)
        results = run_repertoire_statistics(repertoire_scores, analysis_config.model_names, output_prefix,
                                            analysis_config.statistical_tests, None, analysis_config.contrasts,
                                            analysis_config.donor_pattern)
        plot_repertoire_statistics(results, output_prefix, metric, collection_specification,
                                   score_description=f"{title_case(metric)} on Test",
                                   x_label=f"Difference in {title_case(metric)}")
