import os
import multiprocessing
from pathlib import Path
import pandas as pd
import yaml
from immuneML.app.ImmuneMLApp import ImmuneMLApp


def main():
    simulations_dir = "simulated_data"
    helper_dir = "helper_data"
    os.makedirs(simulations_dir, exist_ok=True)
    os.makedirs(helper_dir, exist_ok=True)

    with multiprocessing.Pool() as pool:
        pool.map(execute, range(5))

    model_configs_dir = "model_configs"
    output_dir = "final/models"

    for i in range(5):
        for model in ["PWM"]:
            os.makedirs(f"{model_configs_dir}/{model}", exist_ok=True)
            write_immuneml_config(f"generative_models/{model}.yaml", f"{simulations_dir}/frequent_{i}.tsv", f"{model_configs_dir}/{model}/frequent_{i}.yaml")
            write_immuneml_config(f"generative_models/{model}.yaml", f"{simulations_dir}/rare_{i}.tsv", f"{model_configs_dir}/{model}/rare_{i}.yaml")

    for model in ["PWM"]:
        model_configs_path = f"{model_configs_dir}/{model}"
        immuneml_inputs = os.listdir(model_configs_path)
        for input in immuneml_inputs:
            run_immuneml_command(f"{model_configs_path}/{input}", f"{output_dir}/{model}/{input.strip('.yaml')}")

def run_immuneml_app(input_file, output_dir):
    app = ImmuneMLApp(specification_path=Path(input_file), result_path=Path(output_dir))
    app.run()

def run_immuneml_command(input_file, output_dir):
    command = f"immune-ml {input_file} {output_dir} &"
    exit_code = os.system(command)
    if exit_code != 0:
        raise RuntimeError(f"Running immuneML failed:{command}.")

def write_immuneml_config(input_model_template, input_simulated_data, output_config_file):
    with open(input_model_template, 'r') as file:
        model_template_config = yaml.safe_load(file)

    model_template_config['definitions']['datasets']['dataset']['params']['path'] = input_simulated_data

    with open(output_config_file, 'w') as file:
        yaml.safe_dump(model_template_config, file)

def execute(i):
    generate_rare_and_frequent_olga_sequences(25000, "humanTRB", i, f"helper_data/olga_sequences_{i}.tsv",
                                                  f"helper_data/pgens_file_path_{i}.tsv",
                                                  f"simulated_data/frequent_{i}.tsv",
                                                  f"simulated_data/rare_{i}.tsv")

def generate_rare_and_frequent_olga_sequences(number_of_sequences, model, seed, sequnces_file_path, pgens_file_path,
                                            frequent_sequences_file_path, rare_sequences_file_path):
    """
    This function first generates pure Olga sequences and then computes lower and upper 25% of the sequences based
    on pgen values. As result a file with frequent sequences and a file with rare sequences are generated.
    Both files contain number_of_sequences sequences.

    :param number_of_sequences: number of sequences to generate for each rare and frequent group
    :param model: olga model to use for generating sequences (for example: humanTRB)
    :param seed: seed for random number generator
    :param sequnces_file_path: path to the file where the olga generated sequences will be stored
    :param pgens_file_path: path to the file where the pgen values will be stored
    :param frequent_sequences_file_path: path to the file where the frequent sequences will be stored
    :param rare_sequences_file_path: path to the file where the rare sequences will be stored
    :return:
    """
    number_of_olga_sequences = number_of_sequences * 4
    generate_pure_olga_sequences(number_of_olga_sequences, model, sequnces_file_path, seed)
    column_names_sequences = ["nucleotide", "junction_aa", "v_call", "j_call"]
    olga_sequences = pd.read_csv(sequnces_file_path, sep='\t', names=column_names_sequences)

    compute_pgen(sequnces_file_path, pgens_file_path, model)
    column_names_pgens = ["junction_aa", "pgen"]
    pgens = pd.read_csv(pgens_file_path, sep='\t', names=column_names_pgens)
    pgens.sort_values(by="pgen", inplace=True)

    rare_sequences = pgens.iloc[:number_of_sequences]["junction_aa"]
    rare_filtered = olga_sequences[olga_sequences["junction_aa"].isin(rare_sequences)]
    rare_filtered.to_csv(rare_sequences_file_path, sep='\t', index=False)

    frequent_sequences = pgens.iloc[number_of_olga_sequences - number_of_sequences:]["junction_aa"]
    frequent_filtered = olga_sequences[olga_sequences["junction_aa"].isin(frequent_sequences)]
    frequent_filtered.to_csv(frequent_sequences_file_path, sep='\t', index=False)

def generate_pure_olga_sequences(number_of_sequences, model, output_file_path, seed):
    """
    This function generates number_of_sequences sequences using Olga tool and stores them in output_file_path.
    :param number_of_sequences:  number of sequences to generate
    :param model: olga model to use for generating sequences (for example: humanTRB)
    :param output_file_path: path to the file where the olga generated sequences will be stored
    :param seed: seed for random number generator
    :return:
    """
    command = ('olga-generate_sequences --' + model + ' -o ' + output_file_path + ' -n ' + str(number_of_sequences)
               + ' --seed ' + str(seed))
    exit_code = os.system(command)
    if exit_code != 0:
        raise RuntimeError(f"Running olga tool failed:{command}.")


def compute_pgen(sequences_file_path, pgens_file_path, model):
    """
    This function computes pgen values for the sequences in sequences_file_path and stores them in pgens_file_path.
    :param sequences_file_path: path to the file with olga sequences
    :param pgens_file_path: path to the file where the pgen values will be stored
    :param model: olga model used for generating sequences (for example: humanTRB)
    :return:
    """
    command = 'olga-compute_pgen --' + model + ' -i ' + sequences_file_path + ' -o ' + pgens_file_path \
              + ' --seq_type_out aaseq --seq_in 1 --v_in 2 --j_in 3 --display_off'
    exit_code = os.system(command)
    if exit_code != 0:
        raise RuntimeError(f"Running olga tool failed:{command}.")


if __name__ == "__main__":
    main()
