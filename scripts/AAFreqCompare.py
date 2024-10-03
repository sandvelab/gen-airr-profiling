import argparse
import pandas as pd
from scipy.stats import entropy
import numpy as np


def load_tsv(file_path):
    # Load the TSV file into a DataFrame
    return pd.read_csv(file_path, sep='\t')

def kl_aa_compare(file1, file2):
    # Load the distributions
    df1 = load_tsv(file1)
    df2 = load_tsv(file2)

    # Ensure the files have the same amino acid and position
    merged_df = pd.merge(df1[['amino acid', 'position', 'relative frequency']],
                         df2[['amino acid', 'position', 'relative frequency']],
                         on=['amino acid', 'position'],
                         suffixes=('_p', '_q'))

    # Get the relative frequency columns
    p = merged_df['relative frequency_p'].values
    q = merged_df['relative frequency_q'].values

    # Avoid division by zero by adding a small value
    q = np.where(q == 0, 1e-100, q)

    # Compute KL divergence
    kl_divergence = entropy(p, q)
    return kl_divergence


def main():
    parser = argparse.ArgumentParser(description='Compute KL Divergence between two amino acid frequency distribution TSV files.')
    parser.add_argument('file1', type=str, help='Path to the first TSV file.')
    parser.add_argument('file2', type=str, help='Path to the second TSV file.')
    parser.add_argument('output_file', type=str, default='.', help='Output directory for the results.')

    args = parser.parse_args()

    # Compute KL divergence
    kl_divergence = kl_aa_compare(args.file1, args.file2)

    # Output the results
    with open(args.output_file, 'w') as f:
        f.write(str(kl_divergence))


if __name__ == "__main__":
    main()
