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
    # Load the distributions
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    print(df1['sequence_lengths'])
    print(df2['sequence_lengths'])

    # Ensure the files have the same sequence lengths
    merged_df = pd.merge(df1[['counts', 'sequence_lengths']],
                         df2[['counts', 'sequence_lengths']],
                         on=['sequence_lengths'],
                         suffixes=('_p', '_q'))
    print(merged_df)
    # convert counts to relative frequency?
    #

    return NotImplemented


def kmer_freq(seqs, kmer):
    return len([seq for seq in seqs if kmer in seq[3:6]])/len(seqs)


# Compare for AA distribution
orig_aa_file = 'reports_output/orig/my_instruction/analysis_AA_analysis/report/amino_acid_frequency_distribution.tsv'
pwm_aa_file = 'reports_output/generated/PWM/my_instruction/analysis_AA_analysis/report/amino_acid_frequency_distribution.tsv'
vae_aa_file = 'reports_output/generated/VAE/my_instruction/analysis_AA_analysis/report/amino_acid_frequency_distribution.tsv'
print(f'KL-Divergence for AA distribution (orig vs. PWM): {kl_aa_compare(orig_aa_file, pwm_aa_file)}')
print(f'KL-Divergence for AA distribution (orig vs. VAE): {kl_aa_compare(orig_aa_file, vae_aa_file)}')

# Check for kmer
root_path = Path.cwd()
simulated_seqs_path = root_path / 'data/SLG_dataset_output/my_sim_inst/batch1.tsv'
PWM_generated_seqs_path = root_path / 'genModel_output/PWM/my_train_gen_model_inst/generated_sequences/batch1.tsv'
VAE_generated_seqs_path = root_path / 'genModel_output/VAE/my_train_gen_model_inst/generated_sequences/batch1.tsv'

train_seqs = pd.read_csv(simulated_seqs_path, sep='\t')['sequence_aa']
PWM_gen_seqs = pd.read_csv(PWM_generated_seqs_path, sep='\t')['sequence_aa']
VAE_gen_seqs = pd.read_csv(VAE_generated_seqs_path, sep='\t')['sequence_aa']
print(f'Kmer freq in train data: {kmer_freq(train_seqs, "SLG")}')
print(f'Kmer freq in generated data (PWM): {kmer_freq(PWM_gen_seqs, "SLG")}')
print(f'Kmer freq in generated data (VAE): {kmer_freq(VAE_gen_seqs, "SLG")}')