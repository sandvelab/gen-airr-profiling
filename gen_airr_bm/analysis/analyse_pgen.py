import concurrent.futures
import os

import pandas as pd

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.olga_utils import compute_pgen


def run_pgen_analysis(analysis_config: AnalysisConfig):
    train_sequences_dir = f"{analysis_config.root_output_dir}/train_sequences/{analysis_config.model_names}"
    models_dir = f"{analysis_config.root_output_dir}/generated_sequences"

    output_path_helper_data = os.path.join(analysis_config.root_output_dir, "helper_data")
    os.makedirs(output_path_helper_data, exist_ok=True)

    olga_inputs = get_datasets(models_dir, output_path_helper_data, analysis_config.model_names,
                               analysis_config.default_model_name)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda args: compute_pgen(*args), olga_inputs)


def get_datasets(models_dir, output_path_helper_data, model_names, default_model_name):

    olga_inputs = []
    for model_name in model_names:
        os.makedirs(f"{output_path_helper_data}/{model_name}", exist_ok=True)
        datasets = os.listdir(f"{models_dir}/{model_name}")

        for dataset in datasets:
            sequences_file_path = f"{models_dir}/{model_name}/{dataset}"
            sequences_df = pd.read_csv(sequences_file_path, sep='\t')[["junction", "junction_aa", "v_call", "j_call"]]
            sequence_file_path_olga = f"{output_path_helper_data}/{model_name}/{dataset.split('.')[0]}_olga_sequences.tsv"
            sequences_df.to_csv(sequence_file_path_olga, sep='\t', index=False, header=False)

            pgens_file_path = f"{output_path_helper_data}/{model_name}/{dataset.split('.')[0]}_pgen.tsv"
            olga_inputs.append([sequence_file_path_olga, pgens_file_path, default_model_name])
    return olga_inputs


def plot_pgen_distributions(pgens_all, model_name):
    pass