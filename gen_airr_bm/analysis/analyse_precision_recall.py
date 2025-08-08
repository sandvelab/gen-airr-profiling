import os
from collections import defaultdict

import numpy as np
import pandas as pd

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_sequence_files
from gen_airr_bm.utils.compairr_utils import run_compairr_existence
from gen_airr_bm.utils.plotting_utils import plot_avg_scores, plot_grouped_bar_precision_recall


def run_precision_recall_analysis(analysis_config: AnalysisConfig):
    print("Running precision recall analysis")

    output_dir = analysis_config.analysis_output_dir
    compairr_output_dir = f"{output_dir}/compairr_output"

    for directory in [output_dir, compairr_output_dir]:
        os.makedirs(directory, exist_ok=True)

    compute_precision_recall_scores(analysis_config, compairr_output_dir)


def compute_precision_recall_scores(analysis_config: AnalysisConfig, compairr_output_dir: str):
    reference_data = analysis_config.reference_data
    if not isinstance(reference_data, str):
        raise ValueError("Reference data should be a single string for running precision recall analysis.")

    mean_precision_scores = defaultdict(lambda: defaultdict(list))
    std_precision_scores = defaultdict(lambda: defaultdict(list))
    mean_recall_scores = defaultdict(lambda: defaultdict(list))
    std_recall_scores = defaultdict(lambda: defaultdict(list))

    precision_scores_all = defaultdict(lambda: defaultdict(list))
    recall_scores_all = defaultdict(lambda: defaultdict(list))
    for model in analysis_config.model_names:
        comparison_files_dir = get_sequence_files(analysis_config, model, reference_data)

        for ref_file, gen_files in comparison_files_dir.items():
            dataset_name = os.path.splitext(os.path.basename(ref_file))[0]

            precision_metrics, recall_metrics = get_precision_recall_metrics(ref_file, gen_files, compairr_output_dir, model)

            mean_precision_scores[dataset_name][model] = np.mean(precision_metrics)
            std_precision_scores[dataset_name][model] = np.std(precision_metrics)
            mean_recall_scores[dataset_name][model] = np.mean(recall_metrics)
            std_recall_scores[dataset_name][model] = np.std(recall_metrics)

            precision_scores_all[dataset_name][model] = precision_metrics
            recall_scores_all[dataset_name][model] = recall_metrics

    # Add upper reference of precision recall between train and test
    test_dir = f"{analysis_config.root_output_dir}/{reference_data}_compairr_sequences"
    train_dir = f"{analysis_config.root_output_dir}/train_compairr_sequences"
    for dataset in mean_precision_scores.keys():
        train_file = f"{train_dir}/{dataset}.tsv"
        test_file = f"{test_dir}/{dataset}.tsv"
        ref_precision, ref_recall = get_precision_recall_reference(train_file, test_file, compairr_output_dir,
                                                                   "upper_reference")
        precision_scores_all[dataset]["upper_reference"] = [ref_precision]
        recall_scores_all[dataset]["upper_reference"] = [ref_recall]

    for dataset in mean_precision_scores:
        plot_avg_scores(mean_precision_scores[dataset], std_precision_scores[dataset], analysis_config.analysis_output_dir,
                        "precision", f"{dataset}_precision.png", "precision",
                        scoring_method="precision")

        plot_avg_scores(mean_recall_scores[dataset], std_recall_scores[dataset], analysis_config.analysis_output_dir,
                        "recall", f"{dataset}_recall.png", "recall",
                        scoring_method="recall")

    # barchart of precision and recall scores
    plot_grouped_bar_precision_recall(precision_scores_all, recall_scores_all,
                                      analysis_config.analysis_output_dir, analysis_config.reference_data)


def get_precision_recall_metrics(ref_file, gen_files, compairr_output_dir, model):
    precision_metrics, recall_metrics = [], []
    for gen_file in gen_files:
        precision = compute_compairr_overlap_ratio(gen_file, ref_file, compairr_output_dir,
                                                         model, "precision")
        recall = compute_compairr_overlap_ratio(ref_file, gen_file, compairr_output_dir,
                                                  model, "recall")

        precision_metrics.append(precision)
        recall_metrics.append(recall)

    return precision_metrics, recall_metrics


def get_precision_recall_reference(train_file, test_file, compairr_output_dir, model):
    precision = compute_compairr_overlap_ratio(train_file, test_file, compairr_output_dir,
                                               model, "precision")
    recall = compute_compairr_overlap_ratio(test_file, train_file, compairr_output_dir,
                                            model, "recall")
    return precision, recall


def compute_compairr_overlap_ratio(search_for_file, search_in_file, compairr_output_dir, model_name, metric):
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
