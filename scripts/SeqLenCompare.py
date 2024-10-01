import argparse
import pandas as pd
from scipy.stats import entropy
import numpy as np


def load_csv(file_path):
    # Load the TSV file into a DataFrame
    return pd.read_csv(file_path)

def kl_seq_len_compare(file1, file2):

    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)

    # Make sure both distributions have the same support (i.e., same sequence lengths)
    # Merge the two datasets to align the sequence lengths
    merged_data = pd.merge(data1, data2, on='sequence_lengths', how='outer').fillna(0)
    print(merged_data)

    # Recompute the normalized counts after merging
    P = merged_data['counts_x'] / merged_data['counts_x'].sum()
    Q = merged_data['counts_y'] / merged_data['counts_y'].sum()

    # Compute KL Divergence
    kl_divergence = entropy(P, Q)
    return kl_divergence


def main():
    parser = argparse.ArgumentParser(description='Compute KL Divergence between two sequence length distribution CSV files.')
    parser.add_argument('file1', type=str, help='Path to the first CSV file.')
    parser.add_argument('file2', type=str, help='Path to the second CSV file.')
    parser.add_argument('output_file', type=str, default='.', help='Output directory for the results.')

    args = parser.parse_args()

    # Compute KL divergence
    kl_divergence = kl_seq_len_compare(args.file1, args.file2)

    # Output the results
    with open(args.output_file, 'w') as f:
        f.write(str(kl_divergence))


if __name__ == "__main__":
    main()
