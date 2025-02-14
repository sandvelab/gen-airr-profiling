import os

import pandas as pd


def generate_frequency_based_olga_sequences(number_of_rare_sequences, number_of_frequent_sequences, model, seed,
                                            sequnces_file_path, pgens_file_path, frequent_sequences_file_path,
                                            rare_sequences_file_path):
    n_seq = (number_of_rare_sequences + number_of_frequent_sequences) * 2
    generate_pure_olga_sequences(n_seq, model, sequnces_file_path, seed)
    compute_pgen(sequnces_file_path, pgens_file_path, model)
    pgens = pd.read_csv(pgens_file_path, sep='\t')
    pgens.sort_values(by='pgen', inplace=True)


def generate_pure_olga_sequences(number_of_sequences, model, output_file_path, seed):
    command = ('olga-generate_sequences --' + model + ' -o ' + output_file_path + ' -n ' + str(number_of_sequences)
                   + ' --seed ' + str(seed))
    exit_code = os.system(command)
    if exit_code != 0:
        raise RuntimeError(f"Running olga tool failed:{command}.")


def compute_pgen(sequences_file_path, pgens_file_path, model):
    command = 'olga-compute_pgen --' + model + ' -i ' + sequences_file_path + ' -o ' + pgens_file_path \
              + ' --seq_type_out aaseq --seq_in 1 --v_in 2 --j_in 3 --display_off'
    exit_code = os.system(command)
    if exit_code != 0:
        raise RuntimeError(f"Running olga tool failed:{command}.")


def main():
    number_of_rare_sequences = 100
    number_of_frequent_sequences = 100
    model = 'humanTRB'
    output_file_path = 'output_file_path.tsv'
    seed = 42
    pgens_file_path = 'pgens_file_path.tsv'
    frequent_sequences_file_path = 'frequent_sequences_file_path.tsv'
    rare_sequences_file_path = 'rare_sequences_file_path.tsv'
    generate_frequency_based_olga_sequences(number_of_rare_sequences, number_of_frequent_sequences, model, seed,
                                            output_file_path, pgens_file_path, frequent_sequences_file_path,
                                            rare_sequences_file_path)

if __name__ == "__main__":
    main()
