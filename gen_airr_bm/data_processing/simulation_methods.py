import os

import numpy as np
import pandas as pd


#TODO: This function can be rafactored
def simulate_rare_and_frequent_olga_sequences(number_of_sequences, model, seed, output_path):
    """
    This function first generates pure Olga sequences and then computes lower and upper 25% of the sequences based
    on pgen values. As result a file with frequent sequences and a file with rare sequences are generated.
    Both files contain number_of_sequences sequences.

    :param number_of_sequences: number of sequences to generate for each rare and frequent group
    :param model: olga model to use for generating sequences (for example: humanTRB)
    :param seed: seed for random number generator
    :param output_path: path to the directory where the output files will be stored
    :return:
    """
    os.makedirs(output_path, exist_ok=True)
    output_path_helper_data = os.path.join(output_path, "helper_data")
    os.makedirs(output_path_helper_data, exist_ok=True)
    sequnces_file_path = os.path.join(output_path_helper_data, "olga_sequences.tsv")
    pgens_file_path = os.path.join(output_path_helper_data, "pgens.tsv")
    frequent_sequences_file_path = os.path.join(output_path, "frequent.tsv")
    rare_sequences_file_path = os.path.join(output_path, "rare.tsv")

    number_of_olga_sequences = 4 * number_of_sequences
    simulate_pure_olga_sequences(number_of_olga_sequences, model, sequnces_file_path, seed)
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


def main():
    number_of_sequences = 1000
    model = "humanTRB"
    seed = 42
    os.makedirs("results", exist_ok=True)
    sequnces_file_path = "/results"
    simulate_rare_and_frequent_olga_sequences(number_of_sequences, model, seed, sequnces_file_path)


if __name__ == "__main__":
    main()