import os
import re

import numpy as np
import pandas as pd
import plotly.express as px

from collections import defaultdict
from dataclasses import dataclass, field
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_sequence_files_for_no_train_overlap
from gen_airr_bm.utils.compairr_utils import run_compairr_existence, run_sequence_deduplication
from gen_airr_bm.utils.plotting_utils import plot_avg_innovation_scores


@dataclass
class InnovationScores:
    """ Class to store innovation scores for different models and datasets. """
    mean_innovation_recall: dict = field(default_factory=lambda: defaultdict(dict))
    std_innovation_recall: dict = field(default_factory=lambda: defaultdict(dict))
    innovation_recall_all: dict = field(default_factory=lambda: defaultdict(dict))
    mean_innovation_precision: dict = field(default_factory=lambda: defaultdict(dict))
    std_innovation_precision: dict = field(default_factory=lambda: defaultdict(dict))
    innovation_precision_all: dict = field(default_factory=lambda: defaultdict(dict))
    innovation_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["dataset", "model", "precision_innovation", "recall_innovation", "n_gen_novel", "n_test_only"]
    ))


def run_innovation_umi_analysis(analysis_config: AnalysisConfig) -> None:
    """ Runs innovation analysis on the generated and reference sequences.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        None
    """
    print("Running innovation umi analysis")

    output_dir = analysis_config.analysis_output_dir
    compairr_output_dir = f"{output_dir}/compairr_output"

    for directory in [output_dir, compairr_output_dir]:
        os.makedirs(directory, exist_ok=True)

    compute_and_plot_innovation_scores(analysis_config, compairr_output_dir)


def compute_and_plot_innovation_scores(analysis_config: AnalysisConfig, compairr_output_dir: str) -> None:
    """ Compute innovation scores and plot them.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        compairr_output_dir (str): Directory to store CompAIRR output files.
    Returns:
        None
    """
    scores = InnovationScores()
    preprocess_test_for_innovation_recall(analysis_config)
    preprocess_gen_for_innovation_precision(analysis_config)

    for model in analysis_config.model_names:
        collect_model_scores(analysis_config, model, "test_only", compairr_output_dir, scores)

    plot_innovation_scores(analysis_config, scores)
    scores.innovation_df.to_csv(f"{analysis_config.analysis_output_dir}/innovation_scores.csv", index=False)


def collect_model_scores(analysis_config: AnalysisConfig, model: str, test_reference: str, compairr_output_dir: str,
                         scores: InnovationScores) -> None:
    """ Collect nnovation scores for a given model.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        model (str): Name of the model to analyze.
        test_reference (str): Reference dataset for testing.
        compairr_output_dir (str): Directory to store CompAIRR output files.
        scores (InnovationScores): Storage for innovation scores.
    Returns:
        None
    """
    comparison_files_dir = get_sequence_files_for_no_train_overlap(analysis_config, model, test_reference)

    for ref_file, gen_files in comparison_files_dir.items():
        dataset_name = os.path.splitext(os.path.basename(ref_file))[0]

        innovation_recall_scores, innovation_precision_scores = get_innovation_scores(analysis_config, ref_file,
                                                                                      gen_files, compairr_output_dir,
                                                                                      model, scores)

        mean_innovation_recall_ratio, std_innovation_recall_ratio = np.mean(innovation_recall_scores), np.std(
            innovation_recall_scores)
        mean_innovation_precision_ratio, std_innovation_precision_ratio = np.mean(innovation_precision_scores), np.std(
            innovation_precision_scores)

        scores.mean_innovation_recall[dataset_name][model] = mean_innovation_recall_ratio
        scores.std_innovation_recall[dataset_name][model] = std_innovation_recall_ratio

        scores.mean_innovation_precision[dataset_name][model] = mean_innovation_precision_ratio
        scores.std_innovation_precision[dataset_name][model] = std_innovation_precision_ratio

        scores.innovation_recall_all[dataset_name][model] = innovation_recall_scores
        scores.innovation_precision_all[dataset_name][model] = innovation_precision_scores


def preprocess_test_for_innovation_recall(analysis_config: AnalysisConfig) -> str:
    test_dir = f"{analysis_config.root_output_dir}/test_compairr_sequences"
    train_dir = f"{analysis_config.root_output_dir}/train_compairr_sequences"

    helper_dir = f"{analysis_config.root_output_dir}/test_only_compairr_sequences"
    os.makedirs(helper_dir, exist_ok=True)

    for file_name in os.listdir(test_dir):
        test_df = pd.read_csv(f"{test_dir}/{file_name}", sep='\t')
        train_df = pd.read_csv(f"{train_dir}/{file_name}", sep='\t')

        test_unique_df = test_df.drop_duplicates(subset=["junction_aa"])
        train_unique_df = train_df.drop_duplicates(subset=["junction_aa"])

        test_only_df = test_unique_df[~test_unique_df["junction_aa"].isin(train_unique_df["junction_aa"])]
        test_only_df.to_csv(f"{helper_dir}/{file_name}", sep='\t', index=False)


def preprocess_gen_for_innovation_precision(analysis_config: AnalysisConfig) -> str:
    gen_dir = f"{analysis_config.root_output_dir}/generated_compairr_sequences_split"
    train_dir = f"{analysis_config.root_output_dir}/train_compairr_sequences"

    helper_dir_gen = f"{analysis_config.root_output_dir}/generated_only_compairr_sequences_split"
    os.makedirs(helper_dir_gen, exist_ok=True)

    for model in os.listdir(gen_dir):
        os.makedirs(f"{helper_dir_gen}/{model}", exist_ok=True)
        for file_name in os.listdir(f"{gen_dir}/{model}"):
            gen_df = pd.read_csv(f"{gen_dir}/{model}/{file_name}", sep='\t')
            train_file_name = re.sub(r'_\d+\.tsv$', '.tsv', file_name)
            train_df = pd.read_csv(f"{train_dir}/{train_file_name}", sep='\t')

            gen_unique_df = gen_df.drop_duplicates(subset=["junction_aa"])
            train_unique_df = train_df.drop_duplicates(subset=["junction_aa"])

            gen_only_df = gen_unique_df[~gen_unique_df["junction_aa"].isin(train_unique_df["junction_aa"])]
            gen_only_df.to_csv(f"{helper_dir_gen}/{model}/{file_name}", sep='\t', index=False)


def get_innovation_scores(analysis_config: AnalysisConfig, ref_file: str, gen_files: list,
                          compairr_output_dir: str, model: str, scores: InnovationScores) -> tuple[list, list]:
    """ Get innovation scores for the generated files compared to the reference file.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        ref_file (str): Path to the reference file.
        gen_files (list): List of paths to generated files.
        compairr_output_dir (str): Directory to store CompAIRR output files.
        model (str): Name of the model used for generation.
    Returns:
        tuple: Lists of innovation scores for the generated files.
    """
    innovation_recall_scores = []
    innovation_precision_scores = []
    for gen_file in gen_files:
        innovation, innovation_normalized = compute_compairr_overlap_ratio(analysis_config, ref_file, gen_file,
                                                                           compairr_output_dir,
                                                                           model, "innovation", scores)

        innovation_recall_scores.append(innovation)
        innovation_precision_scores.append(innovation_normalized)

    return innovation_recall_scores, innovation_precision_scores


def compute_compairr_overlap_ratio(analysis_config: AnalysisConfig, search_for_file: str, search_in_file: str,
                                   compairr_output_dir: str, name: str, metric: str, scores: InnovationScores) -> tuple[float, float]:
    """ Compute the overlap ratio between two sequence sets using CompAIRR for innovation.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        search_for_file (str): Path to the file of sequences for which to search for existence in another sequence set.
        search_in_file (str): Path to the file to search for existence in.
        compairr_output_dir (str): Directory to store CompAIRR output files.
        name (str): Name of the model used for generation, or "upper_reference" for the upper reference.
        metric (str): Metric type, either "innovation".
    Returns:
        float: Ratio of non-zero overlap counts to total counts.
    """
    dataset_name = os.path.splitext(os.path.basename(search_in_file))[0]
    file_name = f"{os.path.splitext(os.path.basename(search_in_file))[0]}_{name}_{metric}"

    if analysis_config.deduplicate:
        search_for_file, search_in_file = run_sequence_deduplication(analysis_config, search_for_file, search_in_file)

    run_compairr_existence(compairr_output_dir, search_for_file, search_in_file, file_name,
                           allowed_mismatches=analysis_config.allowed_mismatches, indels=analysis_config.indels)
    compairr_result = pd.read_csv(f"{compairr_output_dir}/{file_name}_overlap.tsv", sep='\t',
                                  names=['sequence_id', 'overlap_count'], header=0)
    n_nonzero_rows = compairr_result[(compairr_result['overlap_count'] != 0)].shape[0]
    innovation_recall_ratio = n_nonzero_rows / len(compairr_result)
    gen_only_df = pd.read_csv(search_in_file, sep='\t')
    innovation_precision_ratio = n_nonzero_rows / len(gen_only_df)

    scores.innovation_df.loc[len(scores.innovation_df)] = [
        dataset_name, name, innovation_precision_ratio, innovation_recall_ratio, len(gen_only_df), len(compairr_result)
    ]

    return innovation_recall_ratio, innovation_precision_ratio


# TODO: Refactor innovation plotting hack
def plot_innovation_scores(analysis_config: AnalysisConfig, scores: InnovationScores) -> None:
    """ Plot innovation scores for each dataset and model.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores (InnovationScores): Storage class for innovation scores.
    Returns:
        None
    """
    for dataset in scores.mean_innovation_recall:
        plot_avg_innovation_scores(analysis_config, scores.mean_innovation_recall[dataset],
                                   scores.std_innovation_recall[dataset],
                                   analysis_config.analysis_output_dir, "innovation",
                                   f"{dataset}_innovation", "innovation",
                                   scoring_method="innovation")

        plot_avg_innovation_scores(analysis_config, scores.mean_innovation_precision[dataset],
                                   scores.std_innovation_precision[dataset],
                                   analysis_config.analysis_output_dir, "innovation",
                                   f"{dataset}_innovation_normalized", "innovation",
                                   scoring_method="innovation")

    mean_innovation_recall, std_innovation_recall = collapse_mean_std_across_datasets(scores.mean_innovation_recall,
                                                                                      scores.std_innovation_recall)
    mean_innovation_precision, std_innovation_precision = collapse_mean_std_across_datasets(
        scores.mean_innovation_precision, scores.std_innovation_precision)

    plot_avg_innovation_scores(analysis_config, mean_innovation_recall, std_innovation_recall,
                               analysis_config.analysis_output_dir, "innovation",
                               f"innovation_{analysis_config.receptor_type.replace(' ', '_')}", "innovation",
                               scoring_method="innovation")

    plot_avg_innovation_scores(analysis_config, mean_innovation_precision, std_innovation_precision,
                               analysis_config.analysis_output_dir, "innovation",
                               f"innovation_{analysis_config.receptor_type.replace(' ', '_')}_normalized", "innovation",
                               scoring_method="innovation")

    plot_innovation_scores_by_n_gen_novel(analysis_config, scores)
    plot_innovation_scores_by_n_gen_novel_pseudo_log(analysis_config, scores)
    plot_innovation_precision_recall(analysis_config, scores)


def plot_innovation_precision_recall(analysis_config: AnalysisConfig, scores: InnovationScores) -> None:
    """ Plot innovation precision vs recall for each dataset and model.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores (InnovationScores): Storage class for innovation scores.
    Returns:
        None
    """
    fig = px.scatter(
        scores.innovation_df,
        x="recall_innovation",
        y="precision_innovation",
        color="model",
        hover_data=["dataset"],
        opacity=0.6
    )

    fig.update_layout(
        title={'text': f"Innovation precision vs recall for {analysis_config.receptor_type}",
               'font': {'size': 22}},
        template="plotly_white",
        colorway=px.colors.qualitative.Safe,
        xaxis_title={'text': "Innovation recall", 'font': {'size': 18}},
        yaxis_title={'text': "Innovation precision", 'font': {'size': 18}},
    )

    output_path = (
        f"{analysis_config.analysis_output_dir}/"
        "innovation_precision_vs_recall.png"
    )

    fig.write_image(output_path, width=900, height=600, scale=2)

    mean_scores_df = (
        scores.innovation_df
        .groupby("model", as_index=False)[
            ["precision_innovation", "recall_innovation"]
        ]
        .mean()
    )

    fig_mean = px.scatter(
        mean_scores_df,
        x="recall_innovation",
        y="precision_innovation",
        color="model",
        opacity=0.6
    )

    fig_mean.update_layout(
        title={'text': f"Mean innovation precision vs recall for {analysis_config.receptor_type}",
               'font': {'size': 22}},
        template="plotly_white",
        colorway=px.colors.qualitative.Safe,
        xaxis_title={'text': "Mean innovation precision", 'font': {'size': 18}},
        yaxis_title={'text': "Mean innovation recall", 'font': {'size': 18}}
    )

    output_path_mean = (
        f"{analysis_config.analysis_output_dir}/"
        "mean_innovation_precision_vs_recall.png"
    )

    fig_mean.write_image(output_path_mean, width=900, height=600, scale=2)


def plot_innovation_scores_by_n_gen_novel(analysis_config: AnalysisConfig, scores: InnovationScores) -> None:
    """ Plot innovation scores by number of generated novel sequences for each dataset and model.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores (InnovationScores): Storage class for innovation scores.
    Returns:
        None
    """
    fig = px.scatter(
        scores.innovation_df,
        x="n_gen_novel",
        y="precision_innovation",
        color="model",
        hover_data=["dataset"],
        opacity=0.6
    )

    fig.update_layout(
        title={'text': f"Innovation precision by number of generated novel sequences for "
                       f"{analysis_config.receptor_type}", 'font': {'size': 22}},
        template="plotly_white",
        colorway=px.colors.qualitative.Safe,
        xaxis_title={'text': "Unique generated sequences not in train", 'font': {'size': 18}},
        yaxis_title={'text': "Innovation precision", 'font': {'size': 18}}
    )

    output_path = (
        f"{analysis_config.analysis_output_dir}/"
        "innovation_precision_by_n_gen_novel.png"
    )

    fig.write_image(output_path, width=900, height=600, scale=2)


def symlog_transform(x, linthresh=1e-3, base=10.0):
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    s = np.sign(x)
    out = np.empty_like(x)

    # linear region
    m = ax <= linthresh
    out[m] = s[m] * (ax[m] / linthresh)

    # log region
    out[~m] = s[~m] * (1.0 + np.log(ax[~m] / linthresh) / np.log(base))
    return out


def plot_innovation_scores_by_n_gen_novel_pseudo_log(analysis_config: AnalysisConfig, scores: InnovationScores) -> None:
    """ Plot innovation scores by number of generated novel sequences for each dataset and model, with pseudo-log scale on precision.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores (InnovationScores): Storage class for innovation scores.
    Returns:
        None
    """
    df = scores.innovation_df.copy()
    threshold = 1e-6
    df["precision_innovation_pseudolog"] = symlog_transform(df["precision_innovation"], linthresh=threshold, base=10)

    fig_pseudo = px.scatter(
        df,
        x="n_gen_novel",
        y="precision_innovation_pseudolog",
        color="model",
        hover_data=["dataset"],
        opacity=0.6
    )

    fig_pseudo.update_layout(
        title={'text': f"Innovation precision (pseudo-log) by number of generated novel sequences for "
                       f"{analysis_config.receptor_type}",
               'font': {'size': 22}},
        template="plotly_white",
        colorway=px.colors.qualitative.Safe,
        xaxis_title={'text': "Unique generated sequences not in train", 'font': {'size': 18}},
        yaxis_title={'text': "Innovation precision (pseudo-log)", 'font': {'size': 18}}
    )

    threshold = 1e-6
    max_val = int(np.ceil(df["precision_innovation_pseudolog"].max()))
    tickvals = np.arange(0, max_val + 1)
    ticktext = ["0"] + [f"{threshold * 10 ** (i - 1):.0e}" for i in tickvals[1:]]

    fig_pseudo.update_yaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext
    )

    fig_pseudo.write_image(
        f"{analysis_config.analysis_output_dir}/innovation_precision_by_n_gen_novel_pseudolog.png",
        width=900,
        height=600,
        scale=2
    )


def collapse_mean_std_across_datasets(mean_dict, std_dict):
    """
    mean_dict: {dataset: {model: mean_value}}
    std_dict:  {dataset: {model: std_value}}

    Returns:
        final_mean: {model: float}
        final_std:  {model: float}
    """

    # Collect values across datasets
    mean_values = defaultdict(list)
    std_values = defaultdict(list)

    for dataset in mean_dict:
        for model in mean_dict[dataset]:
            mean_values[model].append(mean_dict[dataset][model])
            std_values[model].append(std_dict[dataset][model])

    # Compute final aggregated mean + std
    final_mean = {model: float(np.mean(vals)) for model, vals in mean_values.items()}
    final_std = {model: float(np.mean(vals)) for model, vals in std_values.items()}

    return final_mean, final_std
