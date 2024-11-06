import pandas as pd
import argparse
import os


def split_by_cdr3_length(data_file, output_dir):
    """
    Generate new files for each sequence length in the data file.
    """
    data = pd.read_csv(data_file, sep='\t')

    # Filter on sequence length
    max_len = data['sequence_aa'].apply(len).max()
    min_len = data['sequence_aa'].apply(len).min()

    #TO DO: this should be created by snakemake
    if not os.path.exists(str(output_dir)):
        os.makedirs(str(output_dir))

    for i in range(min_len, max_len + 1):
        data[data['sequence_aa'].apply(len) == i].to_csv(f'{str(output_dir)}/batch1_len_{i}.tsv', sep='\t', index=False)
