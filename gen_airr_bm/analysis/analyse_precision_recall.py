import os
import numpy as np
import pandas as pd

from collections import defaultdict
from dataclasses import dataclass, field
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_sequence_files
from gen_airr_bm.utils.compairr_utils import run_compairr_existence
from gen_airr_bm.utils.plotting_utils import plot_avg_scores, plot_grouped_bar_precision_recall


@dataclass
class PrecisionRecallScores:
    """ Class to store precision and recall scores for different models and datasets. """
    mean_precision: dict = field(default_factory=lambda: defaultdict(dict))
    std_precision: dict = field(default_factory=lambda: defaultdict(dict))
    mean_recall: dict = field(default_factory=lambda: defaultdict(dict))
    std_recall: dict = field(default_factory=lambda: defaultdict(dict))
    precision_all: dict = field(default_factory=lambda: defaultdict(dict))
    recall_all: dict = field(default_factory=lambda: defaultdict(dict))


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
    """ Compute precision and recall scores and plot them.
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

    if train_reference:
        print(f"Adding upper reference scores using {train_reference} data.")
        add_upper_reference(analysis_config, train_reference, test_reference, scores, compairr_output_dir)

    plot_precision_recall_scores(analysis_config, scores, test_reference)


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

        precision_scores, recall_scores = get_precision_recall_scores(ref_file, gen_files, compairr_output_dir,
                                                                      model)

        mean_p, std_p = np.mean(precision_scores), np.std(precision_scores)
        mean_r, std_r = np.mean(recall_scores), np.std(recall_scores)

        scores.mean_precision[dataset_name][model] = mean_p
        scores.std_precision[dataset_name][model] = std_p
        scores.mean_recall[dataset_name][model] = mean_r
        scores.std_recall[dataset_name][model] = std_r

        scores.precision_all[dataset_name][model] = precision_scores
        scores.recall_all[dataset_name][model] = recall_scores


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

        ref_precision, ref_recall = get_precision_recall_reference(train_file, test_file, compairr_output_dir)

        scores.precision_all[dataset]["upper_reference"] = [ref_precision]
        scores.recall_all[dataset]["upper_reference"] = [ref_recall]


def get_precision_recall_scores(ref_file: str, gen_files: list, compairr_output_dir: str, model: str) -> tuple:
    """ Get precision and recall scores for the generated files compared to the reference file.
    Args:
        ref_file (str): Path to the reference file.
        gen_files (list): List of paths to generated files.
        compairr_output_dir (str): Directory to store CompAIRR output files.
        model (str): Name of the model used for generation.
    Returns:
        tuple: Lists of precision and recall scores for the generated files.
    """
    precision_scores, recall_scores = [], []
    for gen_file in gen_files:
        precision = compute_compairr_overlap_ratio(gen_file, ref_file, compairr_output_dir,
                                                   model, "precision")
        recall = compute_compairr_overlap_ratio(ref_file, gen_file, compairr_output_dir,
                                                model, "recall")

        precision_scores.append(precision)
        recall_scores.append(recall)

    return precision_scores, recall_scores


def get_precision_recall_reference(train_file, test_file, compairr_output_dir) -> tuple:
    """ Get precision and recall scores for the upper reference between train and test files.
    Args:
        train_file (str): Path to the train file (train used as replacement of generated (model) file for upper
        reference.
        test_file (str): Path to the test file.
        compairr_output_dir (str): Directory to store CompAIRR output files.
    Returns:
        tuple: Precision and recall scores for the upper reference.
    """
    precision = compute_compairr_overlap_ratio(train_file, test_file, compairr_output_dir,
                                               'upper_reference', "precision")
    recall = compute_compairr_overlap_ratio(test_file, train_file, compairr_output_dir,
                                            'upper_reference', "recall")
    return precision, recall


def compute_compairr_overlap_ratio(search_for_file: str, search_in_file: str, compairr_output_dir: str, model_name: str,
                                   metric: str) -> float:
    """ Compute the overlap ratio between two sequence sets using CompAIRR for precision or recall.
    Args:
        search_for_file (str): Path to the file of sequences for which to search for existence in another sequence set.
        search_in_file (str): Path to the file to search for existence in.
        compairr_output_dir (str): Directory to store CompAIRR output files.
        model_name (str): Name of the model used for generation, or "upper_reference" for the upper reference.
        metric (str): Metric type, either "precision" or "recall".
    Returns:
        float: Ratio of non-zero overlap counts to total counts.
    """
    if metric == "precision":
        file_name = f"{os.path.splitext(os.path.basename(search_for_file))[0]}_{model_name}_{metric}"
    else:
        file_name = f"{os.path.splitext(os.path.basename(search_in_file))[0]}_{model_name}_{metric}"

    run_compairr_existence(compairr_output_dir, search_for_file, search_in_file, file_name)
    compairr_result = pd.read_csv(f"{compairr_output_dir}/{file_name}_overlap.tsv", sep='\t',
                                  names=['sequence_id', 'overlap_count'], header=0)
    n_nonzero_rows = compairr_result[(compairr_result['overlap_count'] != 0)].shape[0]
    ratio = n_nonzero_rows / len(compairr_result)

    return ratio


def plot_precision_recall_scores(analysis_config: AnalysisConfig, scores: PrecisionRecallScores,
                                 test_reference: str) -> None:
    """ Plot precision and recall scores for each dataset and model.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores (PrecisionRecallScores): Storage class for precision and recall scores.
        test_reference (str): Reference dataset for testing.
    Returns:
        None
    """
    for dataset in scores.mean_precision:
        plot_avg_scores(scores.mean_precision[dataset], scores.std_precision[dataset],
                        analysis_config.analysis_output_dir, "precision",
                        f"{dataset}_precision.png", "precision",
                        scoring_method="precision")

        plot_avg_scores(scores.mean_recall[dataset], scores.std_recall[dataset],
                        analysis_config.analysis_output_dir, "recall",
                        f"{dataset}_recall.png", "recall",
                        scoring_method="recall")

    plot_grouped_bar_precision_recall(scores.precision_all, scores.recall_all,
                                      analysis_config.analysis_output_dir, test_reference)
