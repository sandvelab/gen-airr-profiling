import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy


def read_distribution(file_path):
    # Read the .tsv file
    df = pd.read_csv(file_path, sep='\t')
    return df


def kl_aa_compare(file1, file2):
    # Load the distributions
    df1 = read_distribution(file1)
    df2 = read_distribution(file2)

    # Ensure the files have the same amino acid and position
    merged_df = pd.merge(df1[['amino acid', 'position', 'relative frequency']],
                         df2[['amino acid', 'position', 'relative frequency']],
                         on=['amino acid', 'position'],
                         suffixes=('_p', '_q'))

    # Get the relative frequency columns
    p = merged_df['relative frequency_p'].values
    q = merged_df['relative frequency_q'].values

    # Avoid division by zero or log of zero, by adding a small value
    p = np.where(p == 0, 1e-10, p)
    q = np.where(q == 0, 1e-10, q)

    # Compute KL divergence
    kl_divergence = entropy(p, q)
    return kl_divergence


def seq_len_compare(file1, file2):
    # Load the data into pandas DataFrames
    # Assuming the data is in two separate CSV files: 'file1.csv' and 'file2.csv'
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


def kmer_freq(seqs, kmer):
    return len([seq for seq in seqs if kmer in seq[3:6]])/len(seqs)


# Compare for AA distribution
root_path = Path.cwd()
analysis_path = root_path / 'results/dataset1/analysis'
aa_path = 'my_instruction/analysis_AA_analysis/report/amino_acid_frequency_distribution.tsv'
orig_aa_file = root_path / analysis_path / 'orig' / aa_path
pwm_aa_file = root_path / analysis_path / 'PWM' / aa_path
vae_aa_file = root_path / analysis_path / 'VAE' / aa_path
print(f'KL-Divergence for AA distribution (orig vs. PWM): {kl_aa_compare(orig_aa_file, pwm_aa_file)}')
print(f'KL-Divergence for AA distribution (orig vs. VAE): {kl_aa_compare(orig_aa_file, vae_aa_file)}')

# Compare for sequence length distribution
seq_len_path = 'my_instruction/analysis_SeqLen_analysis/report/sequence_length_distribution.csv'
orig_seq_len_file = root_path / analysis_path / 'orig' / seq_len_path
pwm_seq_len_file = root_path / analysis_path / 'PWM' / seq_len_path
vae_seq_len_file = root_path / analysis_path / 'VAE' / seq_len_path
print(f'KL-Divergence for sequence length distribution (orig vs. PWM): {seq_len_compare(orig_seq_len_file, pwm_seq_len_file)}')
#print(f'KL-Divergence for sequence length distribution (orig vs. VAE): {seq_len_compare(orig_seq_len_file, vae_seq_len_file)}')

# Check for kmer
model_seq_path = 'my_train_gen_model_inst/generated_sequences/batch1.tsv'
orig_seqs_file = root_path / 'results/dataset1/simulation/my_sim_inst/batch1.tsv'
PWM_seqs_file = root_path / 'results/dataset1/models/PWM' / model_seq_path
VAE_seqs_file = root_path / 'results/dataset1/models/VAE' / model_seq_path

train_seqs = pd.read_csv(orig_seqs_file, sep='\t')['sequence_aa']
PWM_gen_seqs = pd.read_csv(PWM_seqs_file, sep='\t')['sequence_aa']
VAE_gen_seqs = pd.read_csv(VAE_seqs_file, sep='\t')['sequence_aa']
print(f'Kmer freq in train data: {kmer_freq(train_seqs, "SLG")}')
print(f'Kmer freq in generated data (PWM): {kmer_freq(PWM_gen_seqs, "SLG")}')
print(f'Kmer freq in generated data (VAE): {kmer_freq(VAE_gen_seqs, "SLG")}')