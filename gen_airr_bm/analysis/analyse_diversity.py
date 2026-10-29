import math
import os
from collections import defaultdict, Counter
from typing import Callable

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_generated_sequences_dir_name, split_subset_name
from gen_airr_bm.utils.plotting_utils import get_collection_specification_for_title, wrap_title, \
    plot_repertoire_statistics
from gen_airr_bm.utils.statistics_utils import aggregate_repertoire_scores, check_one_repertoire_per_donor, \
    describe_error_bars, run_repertoire_statistics, summarise_repertoire_scores

SCORE_COLUMNS = ["model", "dataset", "subset", "score"]
# With more repertoires, a colour per repertoire is unreadable, so the plots show one uncoloured dot per repertoire
MAX_COLOURED_REPERTOIRES = 19


def run_diversity_analysis(analysis_config: AnalysisConfig) -> None:
    """ Run diversity analysis on the generated sequences.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        None
    """
    print("Running diversity analysis...")

    os.makedirs(analysis_config.analysis_output_dir, exist_ok=True)
    reference_dirs = {}

    for data_split in analysis_config.reference_data:
        reference_dirs[data_split] = f"{analysis_config.root_output_dir}/{data_split}_compairr_sequences"

    diversity_metrics = {"Shannon Entropy": shannon_entropy, "Gini Simpson Index": gini_simpson_index,
                         "Pielou Evenness": pielou_evenness, "Gini Coefficient": gini_coefficient}

    for metric_name, diversity_function in diversity_metrics.items():
        output_path = (f"{analysis_config.analysis_output_dir}/diversity_scores_"
                       f"{metric_name.lower().replace(' ', '_')}")
        compute_and_plot_diversity_scores(analysis_config, reference_dirs, output_path,
                                          diversity_function, metric_name)


def compute_and_plot_diversity_scores(analysis_config: AnalysisConfig, reference_dirs: dict, output_path: str,
                                      diversity_function: Callable, metric_name: str) -> None:
    """ Compute diversity scores and plot them. The diversity of every generated subset is averaged within its
    repertoire before the models are summarised over repertoires (see statistics_utils). The models are also compared
    at the repertoire level: with each other and in the contrasts given in the config. These comparisons are
    descriptive, or include confidence intervals and tests if statistical_tests is set, which requires a donor_pattern
    and one repertoire per donor.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        reference_dirs (dict): Dictionary mapping reference dataset names to their directories.
        output_path (str): Path to save the output plot.
        diversity_function (Callable): Diversity function to compute the scores.
        metric_name (str): Name of the metric for labeling the plot.
    Returns:
        None
    """
    print(f"Computing {metric_name} diversity scores...")
    reference_diversities = {}
    for reference_name, reference_dir in reference_dirs.items():
        reference_diversities[reference_name] = compute_diversity(reference_dir, diversity_function)

    sequences_dir_name = get_generated_sequences_dir_name(analysis_config)
    models_diversities = compute_diversities_for_models(analysis_config.model_names,
                                                        f"{analysis_config.root_output_dir}/"
                                                        f"{sequences_dir_name}",
                                                        diversity_function)
    scores = []
    for model in analysis_config.model_names:
        for subset_name, score in models_diversities[model].items():
            repertoire, subset = split_subset_name(subset_name)
            scores.append({"model": model, "dataset": repertoire, "subset": subset, "score": score})
    scores_df = pd.DataFrame(scores, columns=SCORE_COLUMNS)
    if analysis_config.statistical_tests:
        check_one_repertoire_per_donor(aggregate_repertoire_scores(scores_df, "score", analysis_config.donor_pattern),
                                       analysis_config.donor_pattern)

    plot_diversity_scatter_plotly(reference_diversities, scores_df, output_path, metric_name,
                                  analysis_config.receptor_type, analysis_config.collection,
                                  analysis_config.donor_pattern)
    run_diversity_statistics(analysis_config, scores_df, metric_name)


def compute_diversities_for_models(models: list, gen_dir, diversity_function: Callable) -> dict:
    """ Compute diversity scores for each model in the specified list.
    Args:
        models (list): List of model names.
        gen_dir (str): Directory containing generated sequences for each model.
        diversity_function (Callable): Function to compute the diversity score.
    Returns:
        dict: A dictionary with model names as keys and their diversity scores as values.
    """
    models_diversities = defaultdict(dict)
    for model in models:
        gen_dir_model = f"{gen_dir}/{model}"
        models_diversities[model] = compute_diversity(gen_dir_model, diversity_function)

    return models_diversities


def compute_diversity(directory_path: str, diversity_function: Callable) -> dict:
    """ Compute diversity scores for all datasets in the specified directory.
    Args:
        directory_path (str): Path to the directory containing datasets.
        diversity_function (Callable): Function to compute the diversity score.
    Returns:
        dict: A dictionary with dataset names as keys and their diversity scores as values.
    """
    diversity_measures = {}

    for dataset in [os.path.join(directory_path, file) for file in os.listdir(directory_path)]:
        dataset_without_ext = os.path.splitext(os.path.basename(dataset))[0]
        sequences_df = pd.read_csv(dataset, sep="\t", usecols=["junction_aa"])
        sequences = sequences_df["junction_aa"].tolist()

        diversity_measures[dataset_without_ext] = diversity_function(sequences)

    return diversity_measures


def shannon_entropy(sequences: list) -> float:
    """ Calculate the Shannon entropy of a list of sequences.
    Args:
        sequences (list): List of sequences (strings).
    Returns:
        float: The Shannon entropy of the sequences.
    """
    counts = Counter(sequences)
    total = len(sequences)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return entropy


def gini_simpson_index(sequences: list) -> float:
    """ Calculate the Gini-Simpson index of a list of sequences.
    Args:
        sequences (list): List of sequences (strings).
    Returns:
        float: The Gini-Simpson index of the sequences.
    """
    counts = Counter(sequences)
    total = sum(counts.values())
    if total == 0:
        return 0
    proportions = [count / total for count in counts.values()]
    return 1 - sum(p ** 2 for p in proportions)


def pielou_evenness(sequences: list) -> float:
    """ Calculate the Pielou evenness of a list of sequences.
    Args:
        sequences (list): List of sequences (strings).
    Returns:
        float: The Pielou evenness of the sequences.
    """
    counts = Counter(sequences)
    total = sum(counts.values())
    S = len(counts)
    if total == 0 or S <= 1:
        return 0
    proportions = [count / total for count in counts.values()]
    shannon_entropy = -sum(p * math.log2(p) for p in proportions if p > 0)
    max_entropy = math.log2(S)
    return shannon_entropy / max_entropy


def gini_coefficient(sequences: list) -> float:
    """ Calculate the Gini coefficient of a list of sequences.
    Args:
        sequences (list): List of sequences (strings).
    Returns:
        float: The Gini coefficient of the sequences.
    """
    counts = list(Counter(sequences).values())
    n = len(counts)
    if n == 0:
        return 0
    counts.sort()
    total = sum(counts)
    if total == 0:
        return 0
    cumulative_diff = 0
    for i, xi in enumerate(counts):
        for xj in counts:
            cumulative_diff += abs(xi - xj)
    mean = total / n
    return cumulative_diff / (2 * n ** 2 * mean)


def plot_diversity_scatter_plotly(reference_diversities: dict, scores_df: pd.DataFrame, output_path: str,
                                  metric_name: str, receptor_type: str, collection: str = None,
                                  donor_pattern: str = None) -> None:
    """ Plot diversity scores as one dot per generated subset, coloured by repertoire, with a marker for the mean of
    each source over repertoires and the SD across repertoires as error bars. The subsets are first averaged within
    each repertoire for the mean and SD. With more than MAX_COLOURED_REPERTOIRES repertoires, the dots are instead the
    repertoire means, uncoloured and without a legend. For UMI data, the train and test repertoires are shown as
    sources, with one dot per repertoire; otherwise, the diversity of the train sequences, which are all unique, is a
    reference line.
    Args:
        reference_diversities (dict): {reference: {repertoire: diversity score}}.
        scores_df (pd.DataFrame): One row per model, dataset and generated subset, with the diversity in score.
        output_path (str): Path to save the output plot and plotting data, without extension.
        metric_name (str): Name of the metric for labeling the plot.
        receptor_type (str): Type of receptor being analyzed, e.g. "TCR" or "BCR UMI".
        collection (str): Optional collection, as used in plot titles.
        donor_pattern (str | None): Regular expression that extracts the donor from the repertoire name.
    Returns:
        None
    """
    metric_column = metric_name.lower()
    unique_set_ref_value = None
    df = scores_df.rename(columns={"model": "source", "score": metric_column})
    if "UMI" in receptor_type.upper():
        assert not (reference_diversities.keys() & set(df["source"])), "Overlapping keys!"
        reference_df = pd.DataFrame([{"source": name, "dataset": dataset, "subset": np.nan, metric_column: value}
                                     for name, div_dict in reference_diversities.items()
                                     for dataset, value in div_dict.items()])
        df = pd.concat([reference_df, df], ignore_index=True)
    else:
        unique_set_ref_value = list(reference_diversities['train'].values())[0]
    df = df[["dataset", "subset", metric_column, "source"]]
    # The repertoire names end with the experiment index added in training (e.g. 6174_pLN_CD8_3), which is left out
    df["dataset"] = df["dataset"].str.replace(r"_\d+$", "", regex=True)
    df.to_csv(output_path + ".tsv", sep="\t", index=False)

    sources = list(dict.fromkeys(df["source"]))
    repertoire_scores = aggregate_repertoire_scores(df, metric_column, donor_pattern, source_column="source",
                                                    repertoire_column="dataset")
    summary = summarise_repertoire_scores(repertoire_scores, statistical_tests=False)
    mean_per_source = summary.set_index("source").reindex(sources)

    collection_specification = get_collection_specification_for_title(receptor_type, collection)
    title = f"{metric_name} for Generated {collection_specification} Repertoires"
    colour_repertoires = df["dataset"].nunique() <= MAX_COLOURED_REPERTOIRES
    labels = {"source": "Source", metric_column: f"{metric_name} Score"}
    if colour_repertoires:
        fig = px.strip(
            df,
            x="source",
            y=metric_column,
            color="dataset",
            hover_data=["dataset", "subset", metric_column],
            category_orders={"source": sources, "dataset": sorted(df["dataset"].unique())},
            stripmode="overlay",
            labels=labels,
        )
        fig.update_traces(marker=dict(size=10, opacity=0.8), jitter=0.3)
    else:
        fig = px.strip(
            repertoire_scores.rename(columns={"repertoire": "dataset", "score": metric_column}),
            x="source",
            y=metric_column,
            hover_data=["dataset", metric_column],
            category_orders={"source": sources},
            labels=labels,
        )
        fig.update_traces(marker=dict(size=7, opacity=0.5, color="grey"), jitter=0.4)

    # Mean per source, drawn as a horizontal tick over each column
    fig.add_trace(go.Scatter(
        x=mean_per_source.index,
        y=mean_per_source["mean"],
        error_y=dict(type="data", array=mean_per_source["sd"], visible=True, color="rgba(50,50,50,0.9)"),
        mode="markers",
        marker=dict(symbol="line-ew", size=26, line=dict(width=2.0, color="rgba(50,50,50,0.9)")),
        name="Mean",
        hovertemplate="Mean: %{y:.3g}<extra></extra>",
        showlegend=False,
    ))

    legend_size = 16 if len(df['dataset'].unique()) <= 10 else 12
    fig.update_layout(legend_title_text="Dataset",
                      title={'text': wrap_title(title, width=50),
                             'font': {'size': 20},
                             'y': 0.93,
                             'yanchor': 'top'
                             },
                      margin=dict(t=150),
                      xaxis_title={'text': "Data Origin",
                                           'font': {'size': 20}},
                      yaxis_title={'text': metric_name,
                                   'font': {'size': 20}},
                      template="plotly_white",
                      colorway=px.colors.qualitative.Safe,
                      xaxis=dict(tickfont=dict(size=18), tickangle=-45),
                      yaxis=dict(tickfont=dict(size=18)),
                      legend=dict(font=dict(size=legend_size)),
                      showlegend=colour_repertoires,
                      # The figure grows with the repertoire legend, so that it is not cut off
                      height=max(500, 300 + 28 * df["dataset"].nunique()) if colour_repertoires else 500
                      )

    if colour_repertoires:
        caption = describe_error_bars(summary["n_repertoires"].max(), summary["n_donors"].max(),
                                      dots="generated subsets")
        if unique_set_ref_value is None:
            caption += "<br>Train and test dots: repertoires"
    else:
        caption = describe_error_bars(summary["n_repertoires"].max(), summary["n_donors"].max(),
                                      dots="repertoire means")
    # Left-aligned, since the repertoire legend narrows the plot area
    fig.add_annotation(text=caption,
                       xref="paper", yref="paper", x=0, y=1.02, showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=14))

    if unique_set_ref_value is not None:
        fig.add_hline(
            y=unique_set_ref_value,
            line=dict(color="black", dash="dash"),
            annotation_text=f"All Unique = {unique_set_ref_value:.6g}",
            annotation_position="top right",
            annotation_font=dict(size=18, color="black")
        )

    fig.write_image(output_path + ".png", scale=3)


def run_diversity_statistics(analysis_config: AnalysisConfig, scores_df: pd.DataFrame, metric_name: str) -> None:
    """ Runs the repertoire-level comparisons of the diversity of the generated sequences and plots the contrasts from
    the config. The train and test diversities are not compared with, since the statistics only have a train vs. test
    reference, not a reference per split.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores_df (pd.DataFrame): One row per model, dataset and generated subset, with the diversity in score.
        metric_name (str): Name of the diversity metric, e.g. "Shannon Entropy".
    Returns:
        None
    """
    output_prefix = (f"{analysis_config.analysis_output_dir}/statistics/diversity_"
                     f"{metric_name.lower().replace(' ', '_')}")
    repertoire_scores = aggregate_repertoire_scores(scores_df, "score", analysis_config.donor_pattern)
    results = run_repertoire_statistics(repertoire_scores, analysis_config.model_names, output_prefix,
                                        analysis_config.statistical_tests, None, analysis_config.contrasts,
                                        analysis_config.donor_pattern)
    plot_repertoire_statistics(results, output_prefix, metric_name,
                               get_collection_specification_for_title(analysis_config.receptor_type,
                                                                      analysis_config.collection),
                               score_description=metric_name, x_label=f"Difference in {metric_name}")
