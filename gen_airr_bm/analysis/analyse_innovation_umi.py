import os
import re

import numpy as np
import pandas as pd
import plotly.express as px

from collections import defaultdict
from dataclasses import dataclass, field
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_sequence_files, get_subset_index
from gen_airr_bm.utils.compairr_utils import run_compairr_existence, run_sequence_deduplication
from gen_airr_bm.utils.plotting_utils import plot_avg_innovation_scores, wrap_title, \
    get_collection_specification_for_title, plot_repertoire_statistics
from gen_airr_bm.utils.statistics_utils import aggregate_repertoire_scores, check_one_repertoire_per_donor, \
    run_repertoire_statistics, summarise_scores_by_reference

TEST_ONLY_REFERENCE = "test_only"
# The metrics with the suffix of their overview plot and their description in the statistics plot titles
METRICS = {"innovation_sensitivity": ("", "Innovation Sensitivity"),
           "innovation_precision": ("_normalized", "Innovation Precision")}
SCORE_COLUMNS = ["metric", "model", "reference", "dataset", "subset", "score"]


@dataclass
class InnovationScores:
    """ Class to store innovation scores for different models and datasets. """
    mean_innovation_sensitivity: dict = field(default_factory=lambda: defaultdict(dict))
    std_innovation_sensitivity: dict = field(default_factory=lambda: defaultdict(dict))
    innovation_sensitivity_all: dict = field(default_factory=lambda: defaultdict(dict))
    mean_innovation_precision: dict = field(default_factory=lambda: defaultdict(dict))
    std_innovation_precision: dict = field(default_factory=lambda: defaultdict(dict))
    innovation_precision_all: dict = field(default_factory=lambda: defaultdict(dict))
    innovation_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["dataset", "model", "precision_innovation", "sensitivity_innovation", "n_gen_novel", "n_test_only"]
    ))
    scores: list = field(default_factory=list)


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
    """ Compute innovation scores and plot them. The scores of every generated subset are averaged within its repertoire
    before the models are summarised over repertoires (see statistics_utils). The models are also compared at the
    repertoire level: with each other and in the contrasts given in the config. These comparisons are descriptive, or
    include confidence intervals and tests if statistical_tests is set, which requires a donor_pattern and one
    repertoire per donor.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        compairr_output_dir (str): Directory to store CompAIRR output files.
    Returns:
        None
    """
    scores = InnovationScores()
    preprocess_test_for_innovation_sensitivity(analysis_config)
    preprocess_gen_for_innovation_precision(analysis_config)

    for model in analysis_config.model_names:
        collect_model_scores(analysis_config, model, TEST_ONLY_REFERENCE, compairr_output_dir, scores)

    scores_df = pd.DataFrame(scores.scores, columns=SCORE_COLUMNS)
    scores_df.to_csv(f"{analysis_config.analysis_output_dir}/innovation_repertoire_scores.tsv", sep="\t", index=False)
    if analysis_config.statistical_tests:
        check_one_repertoire_per_donor(aggregate_repertoire_scores(scores_df, "score", analysis_config.donor_pattern),
                                       analysis_config.donor_pattern)

    plot_innovation_scores(analysis_config, scores, scores_df)
    scores.innovation_df.to_csv(f"{analysis_config.analysis_output_dir}/innovation_scores.csv", index=False)
    run_innovation_statistics(analysis_config, scores_df)


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
    comparison_files_dir = get_sequence_files(analysis_config, model, test_reference)

    for ref_file, gen_files in sorted(comparison_files_dir.items()):
        dataset_name = os.path.splitext(os.path.basename(ref_file))[0]

        innovation_sensitivity_scores, innovation_precision_scores = get_innovation_scores(analysis_config, ref_file,
                                                                                      gen_files, compairr_output_dir,
                                                                                      model, scores)

        mean_innovation_sensitivity_ratio, std_innovation_sensitivity_ratio = np.mean(innovation_sensitivity_scores), np.std(
            innovation_sensitivity_scores)
        mean_innovation_precision_ratio, std_innovation_precision_ratio = np.mean(innovation_precision_scores), np.std(
            innovation_precision_scores)

        scores.mean_innovation_sensitivity[dataset_name][model] = mean_innovation_sensitivity_ratio
        scores.std_innovation_sensitivity[dataset_name][model] = std_innovation_sensitivity_ratio

        scores.mean_innovation_precision[dataset_name][model] = mean_innovation_precision_ratio
        scores.std_innovation_precision[dataset_name][model] = std_innovation_precision_ratio

        scores.innovation_sensitivity_all[dataset_name][model] = innovation_sensitivity_scores
        scores.innovation_precision_all[dataset_name][model] = innovation_precision_scores

        for gen_file, sensitivity, precision in zip(gen_files, innovation_sensitivity_scores,
                                                    innovation_precision_scores):
            for metric, score in [("innovation_sensitivity", sensitivity), ("innovation_precision", precision)]:
                scores.scores.append({"metric": metric, "model": model, "reference": test_reference,
                                      "dataset": dataset_name, "subset": get_subset_index(gen_file), "score": score})


def preprocess_test_for_innovation_sensitivity(analysis_config: AnalysisConfig) -> None:
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


def preprocess_gen_for_innovation_precision(analysis_config: AnalysisConfig) -> None:
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
    innovation_sensitivity_scores = []
    innovation_precision_scores = []
    for gen_file in gen_files:
        innovation, innovation_normalized = compute_compairr_overlap_ratio(analysis_config, ref_file, gen_file,
                                                                           compairr_output_dir,
                                                                           model, "innovation", scores)

        innovation_sensitivity_scores.append(innovation)
        innovation_precision_scores.append(innovation_normalized)

    return innovation_sensitivity_scores, innovation_precision_scores


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
        scores (InnovationScores): Storage for innovation scores to update with the computed scores.
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
    innovation_sensitivity_ratio = n_nonzero_rows / len(compairr_result)
    gen_only_df = pd.read_csv(search_in_file, sep='\t')
    innovation_precision_ratio = n_nonzero_rows / len(gen_only_df)

    scores.innovation_df.loc[len(scores.innovation_df)] = [
        dataset_name, name, innovation_precision_ratio, innovation_sensitivity_ratio, len(gen_only_df), len(compairr_result)
    ]

    return innovation_sensitivity_ratio, innovation_precision_ratio


# TODO: Refactor innovation plotting hack
def plot_innovation_scores(analysis_config: AnalysisConfig, scores: InnovationScores, scores_df: pd.DataFrame) -> None:
    """ Plot innovation scores for each dataset and model, and the innovation of each model over repertoires.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores (InnovationScores): Storage class for innovation scores.
        scores_df (pd.DataFrame): One row per metric, model, dataset and generated subset, with the score in score.
    Returns:
        None
    """
    for dataset in sorted(scores.mean_innovation_sensitivity):
        plot_avg_innovation_scores(analysis_config, scores.mean_innovation_sensitivity[dataset],
                                   scores.std_innovation_sensitivity[dataset],
                                   analysis_config.analysis_output_dir, "innovation",
                                   f"{dataset}_innovation", "innovation",
                                   scoring_method="innovation")

        plot_avg_innovation_scores(analysis_config, scores.mean_innovation_precision[dataset],
                                   scores.std_innovation_precision[dataset],
                                   analysis_config.analysis_output_dir, "innovation",
                                   f"{dataset}_innovation_normalized", "innovation",
                                   scoring_method="innovation")

    for metric, (file_suffix, _) in METRICS.items():
        mean_scores, std_scores, repertoire_scores, error_bar_caption = summarise_scores_by_reference(
            scores_df, metric, analysis_config.donor_pattern, score_column="score")
        if TEST_ONLY_REFERENCE not in mean_scores:
            continue
        plot_avg_innovation_scores(analysis_config, mean_scores[TEST_ONLY_REFERENCE],
                                   std_scores[TEST_ONLY_REFERENCE], analysis_config.analysis_output_dir, "innovation",
                                   f"innovation_{analysis_config.receptor_type.replace(' ', '_')}{file_suffix}",
                                   "innovation", scoring_method="innovation", error_bar_caption=error_bar_caption,
                                   repertoire_scores=repertoire_scores[TEST_ONLY_REFERENCE])

    plot_innovation_scores_by_n_gen_novel(analysis_config, scores)
    plot_innovation_scores_by_n_gen_novel_pseudo_log(analysis_config, scores)
    plot_innovation_precision_sensitivity(analysis_config, scores)


def plot_innovation_precision_sensitivity(analysis_config: AnalysisConfig, scores: InnovationScores) -> None:
    """ Plot innovation precision vs sensitivity for each dataset and model.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores (InnovationScores): Storage class for innovation scores.
    Returns:
        None
    """
    if "UMI" in analysis_config.receptor_type:
        pseudolog = True
    else:
        pseudolog = False

    df = scores.innovation_df.copy()
    df = df.sort_values(["dataset", "model"]).reset_index(drop=True)

    threshold = 1e-5
    df["precision_innovation_pseudolog"] = symlog_transform(df["precision_innovation"], linthresh=threshold, base=10)

    colors = px.colors.qualitative.Dark24
    model_names_sorted = sorted(df["model"].unique())
    color_map = {model: colors[i % len(colors)] for i, model in enumerate(model_names_sorted)}

    fig = px.scatter(
        df,
        x="sensitivity_innovation",
        y="precision_innovation_pseudolog" if pseudolog else "precision_innovation",
        color="model",
        hover_data=["dataset"],
        opacity=0.6,
        color_discrete_map=color_map
    )

    fig.update_traces(marker=dict(size=11))

    if pseudolog:
        threshold = 1e-5
        max_val = int(np.ceil(df["precision_innovation_pseudolog"].max()))
        tickvals = np.arange(0, max_val + 1)
        ticktext = ["0"] + [f"{threshold * 10 ** (i - 1):.0e}" for i in tickvals[1:]]

        fig.update_yaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext
        )

    collection_specification = get_collection_specification_for_title(analysis_config.receptor_type,
                                                                     analysis_config.collection)
    y_axis_text = "Innovation Precision (pseudo-log)" if pseudolog else "Innovation Precision"
    fig.update_layout(
        title={'text': wrap_title(f"Innovation Precision vs Sensitivity for {collection_specification} Repertoires", width=50),
               'font': {'size': 20},
               'y': 0.95,
               'yanchor': 'top'
               },
        margin=dict(t=100),
        template="plotly_white",
        colorway=px.colors.qualitative.Dark24,
        xaxis_title={'text': "Innovation Sensitivity", 'font': {'size': 20}},
        yaxis_title={'text': y_axis_text, 'font': {'size': 20}},
        xaxis=dict(tickfont=dict(size=18)),
        yaxis=dict(tickfont=dict(size=18)),
        legend=dict(font=dict(size=18))
    )

    output_path = (
        f"{analysis_config.analysis_output_dir}/"
        "innovation_precision_vs_sensitivity.png"
    )

    fig.write_image(output_path, scale=3)

    mean_scores_df = (
        scores.innovation_df
        .groupby("model", as_index=False)[
            ["precision_innovation", "sensitivity_innovation"]
        ]
        .mean()
        .sort_values("model")
        .reset_index(drop=True)
    )

    fig_mean = px.scatter(
        mean_scores_df,
        x="sensitivity_innovation",
        y="precision_innovation",
        color="model",
        opacity=0.6
    )

    fig_mean.update_layout(
        title={'text': f"Mean Innovation Precision vs Sensitivity for {collection_specification} Repertoires",
               'font': {'size': 22}},
        template="plotly_white",
        xaxis_title={'text': "Mean Innovation Precision", 'font': {'size': 20}},
        yaxis_title={'text': "Mean Innovation Sensitivity", 'font': {'size': 20}}
    )

    output_path_mean = (
        f"{analysis_config.analysis_output_dir}/"
        "mean_innovation_precision_vs_sensitivity.png"
    )

    fig_mean.write_image(output_path_mean, scale=3)


def plot_innovation_scores_by_n_gen_novel(analysis_config: AnalysisConfig, scores: InnovationScores) -> None:
    """ Plot innovation scores by number of generated novel sequences for each dataset and model.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores (InnovationScores): Storage class for innovation scores.
    Returns:
        None
    """
    df = scores.innovation_df.sort_values(["dataset", "model"]).reset_index(drop=True)

    fig = px.scatter(
        df,
        x="n_gen_novel",
        y="precision_innovation",
        color="model",
        hover_data=["dataset"],
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Dark24
    )

    fig.update_traces(marker=dict(size=11))

    collection_specification = get_collection_specification_for_title(analysis_config.receptor_type,
                                                                     analysis_config.collection)
    fig.update_layout(
        title={'text': wrap_title(f"Innovation Precision by Number of Generated Novel Sequences for "
                       f"{collection_specification} Repertoires", width=50), 'font': {'size': 20},
               'y': 0.95,
               'yanchor': 'top'
               },
        margin=dict(t=100),
        template="plotly_white",
        xaxis_title={'text': "Unique Generated Sequences Not in Train", 'font': {'size': 20}},
        yaxis_title={'text': "Innovation Precision", 'font': {'size': 20}},
        xaxis=dict(tickfont=dict(size=18)),
        yaxis=dict(tickfont=dict(size=18)),
        legend=dict(font=dict(size=18))
    )

    output_path = (
        f"{analysis_config.analysis_output_dir}/"
        "innovation_precision_by_n_gen_novel.png"
    )

    fig.write_image(output_path, scale=3)


def symlog_transform(x, linthresh=1/450000, base=10.0):
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
    df = df.sort_values(["dataset", "model"]).reset_index(drop=True)

    threshold = 1e-5
    df["precision_innovation_pseudolog"] = symlog_transform(df["precision_innovation"], linthresh=threshold, base=10)

    fig_pseudo = px.scatter(
        df,
        x="n_gen_novel",
        y="precision_innovation_pseudolog",
        color="model",
        hover_data=["dataset"],
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Dark24
    )

    fig_pseudo.update_traces(marker=dict(size=12))

    collection_specification = get_collection_specification_for_title(analysis_config.receptor_type,
                                                                     analysis_config.collection)
    fig_pseudo.update_layout(
        title={'text': wrap_title(f"Innovation Precision (Pseudo-log) by Number of Generated Novel Sequences for "
                       f"{collection_specification} Repertoires", width=50),
               'font': {'size': 20},
               'y': 0.95,
               'yanchor': 'top'
               },
        margin=dict(t=100),
        template="plotly_white",
        xaxis_title={'text': "Unique generated sequences not in train", 'font': {'size': 20}},
        yaxis_title={'text': "Innovation precision (pseudo-log)", 'font': {'size': 20}},
        xaxis=dict(tickfont=dict(size=18)),
        yaxis=dict(tickfont=dict(size=18)),
        legend=dict(font=dict(size=18))
    )

    threshold = 1e-5
    max_val = int(np.ceil(df["precision_innovation_pseudolog"].max()))
    tickvals = np.arange(0, max_val + 1)
    ticktext = ["0"] + [f"{threshold * 10 ** (i - 1):.0e}" for i in tickvals[1:]]

    fig_pseudo.update_yaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext
    )

    fig_pseudo.write_image(
        f"{analysis_config.analysis_output_dir}/innovation_precision_by_n_gen_novel_pseudolog.png", scale=3)


def run_innovation_statistics(analysis_config: AnalysisConfig, scores_df: pd.DataFrame) -> None:
    """ Runs the repertoire-level comparisons of the innovation sensitivity and precision and plots the contrasts from
    the config. There is no train vs. test reference, since the test-only sequences are by definition not in train.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores_df (pd.DataFrame): One row per metric, model, dataset and generated subset, with the score in score.
    Returns:
        None
    """
    collection_specification = get_collection_specification_for_title(analysis_config.receptor_type,
                                                                     analysis_config.collection)
    for metric, (_, description) in METRICS.items():
        output_prefix = f"{analysis_config.analysis_output_dir}/statistics/{metric}"
        repertoire_scores = aggregate_repertoire_scores(scores_df[scores_df["metric"] == metric], "score",
                                                        analysis_config.donor_pattern)
        results = run_repertoire_statistics(repertoire_scores, analysis_config.model_names, output_prefix,
                                            analysis_config.statistical_tests, None, analysis_config.contrasts,
                                            analysis_config.donor_pattern)
        plot_repertoire_statistics(results, output_prefix, metric, collection_specification,
                                   score_description=description, x_label=f"Difference in {description}")
