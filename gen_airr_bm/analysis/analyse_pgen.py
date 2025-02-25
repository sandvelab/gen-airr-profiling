import concurrent.futures
import os

import pandas as pd

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.olga_utils import compute_pgen


def run_pgen_analysis(analysis_config: AnalysisConfig, output_dir: str):
    train_sequences_dir = f"{output_dir}/train_sequences/{analysis_config.model_names}"
    model_sequences_dir = f"{output_dir}/generated_sequences/{analysis_config.model_names}"

    olga_inputs = get_datasets(model_sequences_dir, output_dir, analysis_config.model_names, )
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_pgen, seq_file, pgen_file, model) for seq_file, pgen_file, model in olga_inputs]

    # train_pgen = get_pgens_all(train_sequences_dir, output_dir, analysis_config.model_name)
    # model_pgen = get_pgens_all(model_sequences_dir, output_dir, analysis_config.model_name)


def get_datasets(dataset_dir, output_dir, model_name, default_model_name):
    datasets = os.listdir(dataset_dir)
    olga_inputs = []
    for dataset in datasets:
        sequences_file_path = f"{dataset_dir}/{dataset}"
        pgens_file_path = f"{output_dir}/{model_name}/{dataset.strip('.')[0]}_pgen.tsv"
        olga_inputs.append([sequences_file_path, pgens_file_path, default_model_name])

    return olga_inputs

def plot_pgen_distributions(pgens_all, model_name):
    pass