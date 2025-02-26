import concurrent.futures
from pathlib import Path

import numpy as np
import plotly.figure_factory as ff
import plotly.colors as pc
import os
import plotly.graph_objects as go

import pandas as pd
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.olga_utils import compute_pgen


def run_pgen_analysis(analysis_config: AnalysisConfig):
    print(f"Analyzing sequence generation probabilities for {analysis_config}")

    os.makedirs(analysis_config.analysis_output_dir, exist_ok=True)
    models_dir = f"{analysis_config.root_output_dir}/generated_sequences"
    olga_helper_data_dir = f"{analysis_config.root_output_dir}/olga_helper_data"
    olga_inputs_model = get_datasets(models_dir, olga_helper_data_dir, analysis_config.model_names,
                               analysis_config.default_model_name)

    olga_inputs_train_test = get_datasets(analysis_config.root_output_dir, olga_helper_data_dir, ["train_sequences", "test_sequences"],
                               analysis_config.default_model_name)

    olga_inputs_all = olga_inputs_model + olga_inputs_train_test

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(compute_pgen_wrapper, olga_inputs_all)

    plotting_data = get_plotting_data(olga_helper_data_dir, analysis_config.model_names + ["train_sequences", "test_sequences"])
    plot_pgen_distributions(plotting_data, analysis_config.analysis_output_dir)
    plot_legal_proportions(olga_helper_data_dir, analysis_config.model_names + ["train_sequences", "test_sequences"], analysis_config.analysis_output_dir)


def compute_pgen_wrapper(args):
    return compute_pgen(*args)


def get_datasets(dataset_dir, olga_helper_data_dir, model_names, default_model_name):

    olga_inputs = []
    for model_name in model_names:
        os.makedirs(f"{olga_helper_data_dir}/{model_name}", exist_ok=True)
        datasets = os.listdir(f"{dataset_dir}/{model_name}")

        for dataset in datasets:
            sequences_file_path = f"{dataset_dir}/{model_name}/{dataset}"
            sequences_df = pd.read_csv(sequences_file_path, sep='\t')[["junction_aa", "v_call", "j_call"]]

            sequence_file_path_olga = f"{olga_helper_data_dir}/{model_name}/{dataset.split('.')[0]}_olga_sequences.tsv"
            sequences_df.to_csv(sequence_file_path_olga, sep='\t', index=False, header=False)

            pgens_file_path = f"{olga_helper_data_dir}/{model_name}/{dataset.split('.')[0]}_pgen.tsv"
            olga_inputs.append([sequence_file_path_olga, pgens_file_path, default_model_name])

    return olga_inputs


def get_plotting_data(olga_helper_data_dir, model_names):
    plotting_data = {}
    for model_name in model_names:
        model_path = Path(olga_helper_data_dir) / model_name
        pgen_files = sorted(model_path.glob("*_pgen.tsv"))

        plotting_data[model_name] = [
            (file.stem, [val for val in pd.read_csv(file, sep='\t', header=None).iloc[:, 1] if val != 0])
            for file in pgen_files
        ]

    return plotting_data


def plot_pgen_distributions(plotting_data, analysis_dir):
    all_pgen_values = []
    labels = []
    colors = []

    available_colors = pc.qualitative.Set2
    model_colors = {model: available_colors[i % len(available_colors)] for i, model in enumerate(plotting_data)}

    for model, experiments in plotting_data.items():
        for file_name, pgen_values in experiments:
            log_pgen_values = [np.log(pgen) for pgen in pgen_values]
            all_pgen_values.append(log_pgen_values)
            labels.append(f"{model}_{file_name}")
            colors.append(model_colors[model])

    fig = ff.create_distplot(all_pgen_values, labels, show_hist=False, show_rug=True, colors=colors)
    fig.update_layout(title='Density Plot of Multiple Models and Experiments',
                      xaxis_title='Pgen values',
                      yaxis_title='Frequency')

    fig.write_html(f"{analysis_dir}/pgen_distribution.html")


def compute_legal_proportions(olga_helper_data_dir, model_names):
    model_stats = {}

    for model_name in model_names:
        model_path = Path(olga_helper_data_dir) / model_name
        pgen_files = sorted(model_path.glob("*_pgen.tsv"))

        nonzero_proportions = []
        for file in pgen_files:
            values = pd.read_csv(file, sep='\t', header=None).iloc[:, 1]
            if len(values) > 0:
                nonzero_proportion = (values != 0).sum() / len(values)
                nonzero_proportions.append(nonzero_proportion)

        if nonzero_proportions:
            mean_prop = sum(nonzero_proportions) / len(nonzero_proportions)
            std_dev = pd.Series(nonzero_proportions).std()
            model_stats[model_name] = (mean_prop, std_dev)

    return model_stats


def plot_legal_proportions(olga_helper_data_dir, model_names, analysis_dir):
    model_stats = compute_legal_proportions(olga_helper_data_dir, model_names)

    models = list(model_stats.keys())
    means = [model_stats[m][0] for m in models]
    std_devs = [model_stats[m][1] for m in models]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=models,
        y=means,
        error_y=dict(type='data', array=std_devs),
        marker=dict(color="royalblue"),
    ))

    fig.update_layout(
        title="Average Proportion of Legal Generation Probability Values per Dataset (with Std Dev)",
        xaxis_title="Dataset",
        yaxis_title="Avg. Proportion of Legal Generation Probability Values",
    )

    fig.write_html(f"{analysis_dir}/legal_sequences.html")

