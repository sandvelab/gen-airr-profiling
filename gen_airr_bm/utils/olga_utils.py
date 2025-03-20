import os

import pandas as pd


def compute_pgen(sequences_file_path, pgens_file_path, default_model_name):
    """
    This function computes pgen values for the sequences in sequences_file_path and stores them in pgens_file_path.
    :param sequences_file_path: path to the file with olga sequences
    :param pgens_file_path: path to the file where the pgen values will be stored
    :param default_model_name: olga model used for generating sequences (for example: humanTRB)
    :return:
    """

    sequences_df = pd.read_csv(sequences_file_path, sep='\t', header=None)
    if sequences_df[1].isnull().values.any() or sequences_df[2].isnull().values.any():
        command = 'olga-compute_pgen --' + default_model_name + ' -i ' + sequences_file_path + ' -o ' + pgens_file_path \
              + ' --seq_type_out aaseq --seq_in 0 --display_off'
    else:
        command = 'olga-compute_pgen --' + default_model_name + ' -i ' + sequences_file_path + ' -o ' + pgens_file_path \
                  + ' --seq_type_out aaseq --seq_in 0 --v_in 1 --j_in 2 --display_off'

    exit_code = os.system(command)
    if exit_code != 0:
        raise RuntimeError(f"Running olga tool failed:{command}.")
