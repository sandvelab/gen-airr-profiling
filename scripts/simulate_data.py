import os

import pandas as pd


def generate_frequency_based_olga_sequences(number_of_sequences, model, seed, sequnces_file_path, pgens_file_path,
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
    column_names_sequences = ["nucleotide", "amino_acid", "v_resolved", "j_resolved"]
    olga_sequences = pd.read_csv(sequnces_file_path, sep='\t', names=column_names_sequences)

    compute_pgen(sequnces_file_path, pgens_file_path, model)
    column_names_pgens = ["amino_acid", "pgen"]
    pgens = pd.read_csv(pgens_file_path, sep='\t', names=column_names_pgens)
    pgens.sort_values(by="pgen", inplace=True)

    rare_sequences = pgens.iloc[:number_of_sequences]["amino_acid"]
    rare_filtered = olga_sequences[olga_sequences["amino_acid"].isin(rare_sequences)]
    rare_filtered.to_csv(rare_sequences_file_path, sep='\t', index=False)

    frequent_sequences = pgens.iloc[number_of_olga_sequences - number_of_sequences:]["amino_acid"]
    frequent_filtered = olga_sequences[olga_sequences["amino_acid"].isin(frequent_sequences)]
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


def main():
    number_of_sequences = 100000
    model = 'humanTRB'
    output_file_path = 'output_file_path.tsv'
    seed = 42
    pgens_file_path = 'pgens_file_path.tsv'
    frequent_sequences_file_path = 'frequent_sequences_file_path.tsv'
    rare_sequences_file_path = 'rare_sequences_file_path.tsv'
    generate_frequency_based_olga_sequences(number_of_sequences, model, seed,
                                            output_file_path, pgens_file_path, frequent_sequences_file_path,
                                            rare_sequences_file_path)


if __name__ == "__main__":
    main()
