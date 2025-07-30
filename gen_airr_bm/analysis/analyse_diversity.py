import math
import os
from collections import defaultdict, Counter
from typing import Callable

import numpy as np
import plotly.express as px
import pandas as pd

from gen_airr_bm.constants.dataset_split import DatasetSplit
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

    test_dir = f"{analysis_config.root_output_dir}/{DatasetSplit.TEST.value}_compairr_sequences"
    train_dir = f"{analysis_config.root_output_dir}/{DatasetSplit.TRAIN.value}_compairr_sequences"

    diversity_metrics = {"Shannon Entropy": shannon_entropy, "Gini Simpson Index": gini_simpson_index,
                         "Pielou Evenness": pielou_evenness, "Gini Coefficient": gini_coefficient}

    for metric_name, diversity_function in diversity_metrics.items():
        output_path = (f"{analysis_config.analysis_output_dir}/diversity_scores_"
                       f"{metric_name.lower().replace(' ', '_')}.png")
        compute_and_plot_diversity_scores(analysis_config, test_dir, train_dir, output_path,
                                          diversity_function, metric_name)


def compute_and_plot_diversity_scores(analysis_config: AnalysisConfig, test_dir: str, train_dir: str, output_path: str,
                                      diversity_function: Callable, metric_name: str) -> None:
    """ Compute diversity scores and plot them.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        test_dir (str): Path to the test dataset directory.
        train_dir (str): Path to the training dataset directory.
        output_path (str): Path to save the output plot.
        diversity_function (Callable): Diversity function to compute the scores.
        metric_name (str): Name of the metric for labeling the plot.
    Returns:
        None
    """
    print(f"Computing {metric_name} diversity scores...")
    test_diversity = compute_diversity(test_dir, diversity_function)
    train_diversity = compute_diversity(train_dir, diversity_function)
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

    plot_diversity_scatter_plotly(train_diversity, test_diversity, models_diversities_grouped, output_path, metric_name)


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
        gen_dir = f"{gen_dir}/{model}"
        models_diversities[model] = compute_diversity(gen_dir, diversity_function)

    return models_diversities


def compute_diversity(directory_path, diversity_function):
    diversity_measures = {}

    for dataset in [os.path.join(directory_path, file) for file in os.listdir(directory_path)]:
        dataset_without_ext = os.path.splitext(os.path.basename(dataset))[0]
        sequences_df = pd.read_csv(dataset, sep="\t", usecols=["junction_aa"])
        sequences = sequences_df["junction_aa"].tolist()

        diversity_measures[dataset_without_ext] = diversity_function(sequences)

    return diversity_measures


def shannon_entropy(sequences):
    counts = Counter(sequences)
    total = len(sequences)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return entropy


def gini_simpson_index(sequences):
    counts = Counter(sequences)
    total = sum(counts.values())
    if total == 0:
        return 0
    proportions = [count / total for count in counts.values()]
    return 1 - sum(p ** 2 for p in proportions)


def pielou_evenness(sequences):
    counts = Counter(sequences)
    total = sum(counts.values())
    S = len(counts)
    if total == 0 or S <= 1:
        return 0
    proportions = [count / total for count in counts.values()]
    shannon_entropy = -sum(p * math.log2(p) for p in proportions if p > 0)
    max_entropy = math.log2(S)
    return shannon_entropy / max_entropy


def gini_coefficient(sequences):
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


def plot_diversity_scatter_plotly(train_div, test_div, models_div, output_path, metric_name):
    data = []

    for dataset, value in train_div.items():
        data.append({"dataset": dataset, metric_name.lower(): value, "source": "train"})
    for dataset, value in test_div.items():
        data.append({"dataset": dataset, metric_name.lower(): value, "source": "test"})

    for model_name, div_dict in models_div.items():
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
