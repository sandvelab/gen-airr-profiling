import concurrent.futures
from pathlib import Path

import plotly.figure_factory as ff
import os

import pandas as pd
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.olga_utils import compute_pgen


def run_pgen_analysis(analysis_config: AnalysisConfig):

    os.makedirs(analysis_config.analysis_output_dir, exist_ok=True)

    models_dir = f"{analysis_config.root_output_dir}/generated_sequences"
    olga_inputs_model = get_datasets(models_dir, analysis_config.model_names,
                               analysis_config.default_model_name)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda args: compute_pgen(*args), olga_inputs_model)

    plotting_data = get_plotting_data(models_dir, analysis_config.model_names)
    plot_pgen_distributions(plotting_data, analysis_config.analysis_output_dir)

    # train_dir = f"{analysis_config.root_output_dir}/train_sequences"
    # olga_inputs_train = get_datasets(train_dir, analysis_config.model_names,
    #                              analysis_config.default_model_name)
    #
    # test_dir = f"{analysis_config.root_output_dir}/test_sequences"
    # olga_inputs_test = get_datasets(test_dir, analysis_config.model_names,
    #                             analysis_config.default_model_name)


def get_datasets(dataset_dir, model_names, default_model_name):

    olga_inputs = []
    for model_name in model_names:
        os.makedirs(f"{dataset_dir}/olga_helper_data/{model_name}", exist_ok=True)
        datasets = os.listdir(f"{dataset_dir}/{model_name}")

        for dataset in datasets:
            sequences_file_path = f"{dataset_dir}/{model_name}/{dataset}"
            sequences_df = pd.read_csv(sequences_file_path, sep='\t')[["junction", "junction_aa", "v_call", "j_call"]]

            sequence_file_path_olga = f"{dataset_dir}/olga_helper_data/{model_name}/{dataset.split('.')[0]}_olga_sequences.tsv"
            sequences_df.to_csv(sequence_file_path_olga, sep='\t', index=False, header=False)

            pgens_file_path = f"{dataset_dir}/olga_helper_data/{model_name}/{dataset.split('.')[0]}_pgen.tsv"
            olga_inputs.append([sequence_file_path_olga, pgens_file_path, default_model_name])

    return olga_inputs


def get_plotting_data(dataset_dir, model_names):
    plotting_data = {}
    for model_name in model_names:
        model_path = Path(dataset_dir) / "olga_helper_data" / model_name
        pgen_files = sorted(model_path.glob("*_pgen.tsv"))

        plotting_data[model_name] = [
            (file.stem, pd.read_csv(file, sep='\t', header=None).iloc[:, 1].tolist())
            for file in pgen_files
        ]

    return plotting_data


def plot_pgen_distributions(plotting_data, analysis_dir):
    all_pgen_values = []
    labels = []

    for model, experiments in plotting_data.items():
        for file_name, pgen_values in experiments:
            all_pgen_values.append(pgen_values)
            labels.append(f"{model}_{file_name}")

    fig = ff.create_distplot(all_pgen_values, labels, show_hist=False, show_rug=True)
    fig.update_layout(title='Density Plot of Multiple Models and Experiments',
                      xaxis_title='Pgen values',
                      yaxis_title='Density')

    fig.write_html(f"{analysis_dir}/pgen_distribution.html")


