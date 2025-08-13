import math
import os
from collections import defaultdict, Counter
from typing import Callable

import numpy as np
import plotly.express as px
import pandas as pd

from gen_airr_bm.core.analysis_config import AnalysisConfig


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
                       f"{metric_name.lower().replace(' ', '_')}.png")
        compute_and_plot_diversity_scores(analysis_config, reference_dirs, output_path,
                                          diversity_function, metric_name)


def compute_and_plot_diversity_scores(analysis_config: AnalysisConfig, reference_dirs: dict, output_path: str,
                                      diversity_function: Callable, metric_name: str) -> None:
    """ Compute diversity scores and plot them.
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

    models_diversities = compute_diversities_for_models(analysis_config.model_names,
                                                        f"{analysis_config.root_output_dir}/"
                                                        f"generated_compairr_sequences_split",
                                                        diversity_function)
    models_diversities_grouped = defaultdict(dict)
    for model in analysis_config.model_names:
        datasets = models_diversities[model].keys()
        datasets_grouped = set("_".join(dataset.split("_")[:-1]) for dataset in datasets)
        for datasets_group in datasets_grouped:
            models_diversities_grouped[model][datasets_group] = np.mean([
                models_diversities[model][dataset]
                for dataset in datasets
                if dataset.startswith(datasets_group)
            ])

    plot_diversity_scatter_plotly(reference_diversities, models_diversities_grouped, output_path, metric_name)


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


def plot_diversity_scatter_plotly(reference_diversities: dict, models_diversities: dict, output_path: str,
                                  metric_name: str):
    """ Plot diversity scores using Plotly scatter plot.
    Args:
        reference_diversities (dict): Dictionary with reference dataset names and their diversity scores.
        models_diversities (dict): Dictionary with model names and their diversity scores.
        output_path (str): Path to save the output plot.
        metric_name (str): Name of the metric for labeling the plot.
    Returns:
        None
    """
    data = []

    for ref_name, div_dict in reference_diversities.items():
        for dataset, value in div_dict.items():
            data.append({"dataset": dataset, metric_name.lower(): value, "source": ref_name})

    for model_name, div_dict in models_diversities.items():
        for dataset, value in div_dict.items():
            data.append({"dataset": dataset, metric_name.lower(): value, "source": model_name})

    df = pd.DataFrame(data)

    fig = px.scatter(
        df,
        x="source",
        y=metric_name.lower(),
        color="dataset",
        hover_data=["dataset", metric_name.lower()],
        title=f"{metric_name} by Source and Dataset",
        labels={"source": "Source", metric_name.lower(): f"{metric_name} Score"},
    )

    fig.update_traces(marker=dict(size=10, opacity=0.8), selector=dict(mode='markers'))
    fig.update_layout(legend_title_text="Dataset", xaxis_title="Source", yaxis_title=metric_name)

    fig.write_image(output_path)
