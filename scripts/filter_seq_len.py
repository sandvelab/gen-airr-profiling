import pandas as pd
import argparse
import os


def filter_on_cdr3_length(data_file, output_dir):
    """
    Generate new files for each sequence length in the data file.
    """
    data = pd.read_csv(data_file, sep='\t')

    # Filter on sequence length
    max_len = data['sequence_aa'].apply(len).max()
    min_len = data['sequence_aa'].apply(len).min()

    for i in range(min_len, max_len + 1):
        data[data['sequence_aa'].apply(len) == i].to_csv(f'{output_dir}/batch1_len_{i}.tsv', sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(description='Filter datafile by CDR3 sequence length')
    parser.add_argument('data_file', type=str, help='Path to the TSV file.')
    parser.add_argument('output_dir', type=str, help='Directory to save the output files.')

    args = parser.parse_args()

    # Generate output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Generate new files for each sequence length
    filter_on_cdr3_length(args.data_file, args.output_dir)


if __name__ == "__main__":
    main()
