import os

import numpy as np
import pandas as pd

from gen_airr_bm.core.data_generation_config import DataGenerationConfig
from gen_airr_bm.utils.olga_utils import compute_pgen


#TODO: This function can be rafactored
def simulate_rare_and_frequent_olga_sequences(config: DataGenerationConfig):
    """
    This function first generates pure Olga sequences and then computes lower and upper 25% of the sequences based
    on pgen values. As result a file with frequent sequences and a file with rare sequences are generated.
    Both files contain number_of_sequences sequences.

    :param config: DataGenerationConfig object with the following attributes:
    :return:
    """
    number_of_sequences = config.n_samples
    output_path = config.output_dir
    default_model_name = config.default_model_name
    seed = config.seed
    os.makedirs(output_path, exist_ok=True)
    output_path_helper_data = os.path.join(output_path, "helper_data")
    os.makedirs(output_path_helper_data, exist_ok=True)
    sequnces_file_path = os.path.join(output_path_helper_data, "olga_sequences.tsv")
    pgens_file_path = os.path.join(output_path_helper_data, "pgens.tsv")
    frequent_sequences_file_path = os.path.join(output_path, "frequent.tsv")
    rare_sequences_file_path = os.path.join(output_path, "rare.tsv")

    number_of_olga_sequences = 4 * number_of_sequences
    simulate_pure_olga_sequences(number_of_olga_sequences, default_model_name, sequnces_file_path, seed)
    column_names_sequences = ["junction_aa", "v_call", "j_call"]
    olga_sequences = pd.read_csv(sequnces_file_path, sep='\t', names=column_names_sequences)
    sequences_file_path_drop_nucleotides = os.path.join(output_path_helper_data, "olga_sequences_drop_nucleotides.tsv")
    olga_sequences.to_csv(sequences_file_path_drop_nucleotides, sep='\t', index=False, header=False)

    compute_pgen(sequences_file_path_drop_nucleotides, pgens_file_path, default_model_name)
    column_names_pgens = ["junction_aa", "pgen"]
    pgens = pd.read_csv(pgens_file_path, sep='\t', names=column_names_pgens)
    pgens.sort_values(by="pgen", inplace=True)

    rare_sequences = pgens.iloc[:number_of_sequences]["junction_aa"]
    rare_filtered = olga_sequences[olga_sequences["junction_aa"].isin(rare_sequences)]
    rare_filtered.to_csv(rare_sequences_file_path, sep='\t', index=False)

    frequent_sequences = pgens.iloc[number_of_olga_sequences - number_of_sequences:]["junction_aa"]
    frequent_filtered = olga_sequences[olga_sequences["junction_aa"].isin(frequent_sequences)]
    frequent_filtered.to_csv(frequent_sequences_file_path, sep='\t', index=False)


def simulate_experimental_and_olga_sequences(number_of_sequences, model, seed, olga_sequences_file_path,
                                             experimental_data_file_path, experimental_sampled_data_file_path):
    """
    This function generates number_of_sequences sequences using Olga tool and samples number_of_sequences from
    experimental data and stores them in mixed_sequences_file_path.
    :param number_of_sequences: number of sequences to generate and sample from experimental data
    :param model: olga model to use for generating sequences (for example: humanTRB)
    :param seed: seed for random number generator
    :param olga_sequences_file_path: path to the file where the olga generated sequences will be stored
    :param experimental_data_file_path: path to the file with experimental data. The file should contain columns with
    the following names: "sequence_aa", "v_call", "j_call" and should be in tsv format.
    :param experimental_sampled_data_file_path: path to the file where the sampled experimental sequences will be stored
    :return:
    """
    simulate_pure_olga_sequences(number_of_sequences, model, olga_sequences_file_path, seed)

    columns_to_read = ["sequence_aa", "v_call", "j_call"]
    experimental_data = pd.read_csv(experimental_data_file_path, sep='\t', usecols=columns_to_read)
    if len(experimental_data) < number_of_sequences:
        raise ValueError(f"Not enough sequences! Requested {number_of_sequences}, but only {len(experimental_data)} "
                         f"available.")

    np.random.seed(seed)
    experimental_sequences = experimental_data.sample(n=number_of_sequences, random_state=seed)
    experimental_sequences.to_csv(experimental_sampled_data_file_path, sep='\t', index=False)


def preprocess_experimental_data(config: DataGenerationConfig):
    """
    This function preprocesses experimental data by sampling number_of_sequences sequences from the input data.
    :param config: DataGenerationConfig object with the following attributes:
    :return:
    """
    number_of_sequences = config.n_samples
    output_path = config.output_dir
    input_path = config.data_file
    seed = config.seed
    input_columns = config.input_columns

    train_dir = os.path.join(output_path, "train")
    test_dir = os.path.join(output_path, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    input_path_file_name = os.path.basename(input_path)
    experimental_train_file_path = os.path.join(train_dir, input_path_file_name)
    experimental_test_file_path = os.path.join(test_dir, input_path_file_name)
    experimental_data = pd.read_csv(input_path, sep='\t', usecols=input_columns)
    experimental_data = experimental_data.drop_duplicates()

    # we need at least 2 * number_of_sequences sequences to split them into train and test
    if len(experimental_data) < 2 * number_of_sequences:
        raise ValueError(f"Not enough sequences! Requested {2*number_of_sequences}, but only {len(experimental_data)} "
                         f"available.")

    experimental_sequences = experimental_data.sample(n=2*number_of_sequences, random_state=seed)
    experimental_train = experimental_sequences.iloc[:number_of_sequences].reset_index(drop=True)
    experimental_test = experimental_sequences.iloc[number_of_sequences:].reset_index(drop=True)
    experimental_train.to_csv(experimental_train_file_path, sep='\t', index=False)
    experimental_test.to_csv(experimental_test_file_path, sep='\t', index=False)


def preprocess_experimental_umi_data(config: DataGenerationConfig):
    """
    This function preprocesses experimental data by sampling number_of_sequences sequences from the input data.
    :param config: DataGenerationConfig object with the following attributes:
    :return:
    """
    number_of_sequences = config.n_samples
    output_path = config.output_dir
    input_path = config.data_file
    seed = config.seed
    input_columns = config.input_columns

    train_dir = os.path.join(output_path, "train")
    test_dir = os.path.join(output_path, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    input_path_file_name = os.path.basename(input_path)
    experimental_train_file_path = os.path.join(train_dir, input_path_file_name)
    experimental_test_file_path = os.path.join(test_dir, input_path_file_name)
    experimental_data = pd.read_csv(input_path, sep='\t', usecols=input_columns)
    experimental_data = experimental_data[~experimental_data.junction_aa.str.contains("\*")]
    experimental_data['v_call'] = experimental_data['v_call'].str.split(',').str[0]
    experimental_data['j_call'] = experimental_data['j_call'].str.split(',').str[0]
    experimental_data = experimental_data.dropna(subset=["junction_aa", "v_call", "j_call", "umi_count", "locus"])
    experimental_data = experimental_data.loc[experimental_data.index.repeat(experimental_data["umi_count"])]

    # we need at least 2 * number_of_sequences sequences to split them into train and test
    if len(experimental_data) < 2 * number_of_sequences:
        raise ValueError(f"Not enough sequences! Requested {2*number_of_sequences}, but only {len(experimental_data)} "
                         f"available.")

    experimental_sequences = experimental_data.sample(n=2*number_of_sequences, random_state=seed)
    experimental_train = experimental_sequences.iloc[:number_of_sequences].reset_index(drop=True)
    experimental_test = experimental_sequences.iloc[number_of_sequences:].reset_index(drop=True)
    experimental_train.to_csv(experimental_train_file_path, sep='\t', index=False)
    experimental_test.to_csv(experimental_test_file_path, sep='\t', index=False)


def simulate_pure_olga_sequences(number_of_sequences, model, output_file_path, seed):
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
