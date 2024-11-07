import pandas as pd
import argparse
import os


def filter_by_cdr3_length(data_file, output_file, sequence_length):
    """
    Generate new filtered data file for given sequence length.
    """
    data = pd.read_csv(data_file, sep='\t')

    # Filter on sequence length
    data[data['sequence_aa'].apply(len) == int(sequence_length)].to_csv(output_file, sep='\t', index=False)
