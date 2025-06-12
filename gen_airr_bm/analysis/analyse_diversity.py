import math
import os
from collections import defaultdict, Counter

import plotly.express as px
import pandas as pd

from gen_airr_bm.constants.dataset_split import DatasetSplit
from gen_airr_bm.core.analysis_config import AnalysisConfig


def run_diversity_analysis(analysis_config: AnalysisConfig):
    print("Running diversity analysis...")

    os.makedirs(analysis_config.analysis_output_dir, exist_ok=True)
    output_path = f"{analysis_config.analysis_output_dir}/diversity_scores.png"

    test_dir = f"{analysis_config.root_output_dir}/{DatasetSplit.TEST.value}_compairr_sequences"
    train_dir = f"{analysis_config.root_output_dir}/{DatasetSplit.TRAIN.value}_compairr_sequences"

    test_diversity = compute_diversity(test_dir, shannon_entropy)
    train_diversity = compute_diversity(train_dir, shannon_entropy)

    models_diversities = defaultdict(dict)
    for model in analysis_config.model_names:
        gen_dir = f"{analysis_config.root_output_dir}/generated_compairr_sequences_split/{model}"
        models_diversities[model] = compute_diversity(gen_dir, shannon_entropy)

    plot_diversity_scatter_plotly(train_diversity, test_diversity, models_diversities, output_path)


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


def plot_diversity_scatter_plotly(train_div, test_div, models_div, output_path):
    data = []

    for dataset, value in train_div.items():
        data.append({"dataset": dataset, "diversity": value, "source": "train"})
    for dataset, value in test_div.items():
        data.append({"dataset": dataset, "diversity": value, "source": "test"})

    for model_name, div_dict in models_div.items():
        for dataset, value in div_dict.items():
            data.append({"dataset": dataset, "diversity": value, "source": model_name})

    df = pd.DataFrame(data)

    fig = px.scatter(
        df,
        x="source",
        y="diversity",
        color="dataset",
        hover_data=["dataset", "diversity"],
        title="Diversity by Source and Dataset",
        labels={"source": "Source", "diversity": "Diversity Score"},
    )

    fig.update_traces(marker=dict(size=10, opacity=0.8), selector=dict(mode='markers'))
    fig.update_layout(legend_title_text="Dataset", xaxis_title="Source", yaxis_title="Diversity")

    fig.write_image(output_path)
