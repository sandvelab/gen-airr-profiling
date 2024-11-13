import argparse
import pandas as pd
import numpy as np
import os
import plotly.express as px
from scipy import stats
import logomaker
import matplotlib.pyplot as plt

from immuneML.reports.PlotlyUtil import PlotlyUtil
from statsmodels.stats.multitest import multipletests
from numpy.linalg import eigvalsh


def load_tsv(file_path):
    # Load the TSV file into a DataFrame
    return pd.read_csv(file_path, sep='\t')


def get_aa_counts_frequencies_df(file_path, max_position):
    # Load the data
    df = pd.read_csv(file_path, sep='\t')

    num_sequences = len(df)

    # return None if the dataframe is empty
    if df.empty:
        return None, num_sequences

    # Initialize a list to store the results
    results = []

    # Loop through each position up to max_position
    for pos in range(int(max_position)):
        # Extract amino acids at the current position across all sequences
        aa_series = df['sequence_aa'].str[pos].dropna()

        # Get counts of each amino acid at this position
        aa_counts = aa_series.value_counts().to_dict()
        total_count = len(aa_series)

        # Calculate relative frequencies and store the results
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            count = aa_counts.get(aa, 0)
            relative_freq = count / total_count if total_count > 0 else 0
            results.append({'amino acid': aa, 'position': pos, 'count': count, 'relative frequency': relative_freq})

    # Convert results to a DataFrame
    result_df = pd.DataFrame(results)
    return result_df, num_sequences


def get_aa_color_map():
    return {
        'Y': '#66c5cc', 'W': '#b3de69', 'V': '#dcb0f2', 'T': '#d9d9d9', 'S': '#8dd3c7', 'R': '#fb8072',
        'Q': '#9eb9f3', 'P': '#f89c74', 'N': '#87c55f', 'M': '#fe88b1', 'L': '#c9db74', 'K': '#ffed6f',
        'I': '#b497e7', 'H': '#f6cf71', 'G': '#be82da', 'F': '#80b1d3', 'E': '#fdb462', 'D': '#fccde5',
        'C': '#bc80bd', 'A': '#ccebc5'
    }


def extract_aa_counts_by_pos(df):
    pos_counts = {}
    for i, row in df.iterrows():
        pos = row["position"]
        aa = row["amino acid"]

        if pos not in pos_counts:
            pos_counts[pos] = {aa: row["count"]}
        else:
            if aa not in pos_counts[pos]:
                pos_counts[pos][aa] = row["count"]
            else:
                raise ValueError("Amino acid occur multiple times for same position")

    return pos_counts


def get_aa_counts(aa, pos, pos_counts):
    # get the count of the amino acid at the position
    count_aa = pos_counts[pos][aa]
    # get the count of all other amino acids at the position
    count_other_aa = sum(pos_counts[pos].values()) - count_aa
    return count_aa, count_other_aa


def get_covariance_matrix(df):
    frequency_df = df.pivot(index='position', columns='amino acid', values='relative frequency')
    frequency_df = frequency_df.fillna(0)

    cov_matrix = np.cov(frequency_df.T)   # Transpose the matrix to get each amino acid as a variable
    return cov_matrix


def adjust_alpha_for_dependent_tests(aa_dataframe, positional_counts, alpha=0.05):
    # Sum of the eigenvalues gives the effective number of tests
    eigenvalues = eigvalsh(get_covariance_matrix(aa_dataframe))
    m_eff_aa = np.sum(eigenvalues) / np.max(eigenvalues)
    # Adjust alpha for dependencies between amino acid frequencies
    adjusted_alpha = alpha / (m_eff_aa * len(positional_counts.keys()))
    return adjusted_alpha


def find_significant_amino_acids(df_simulated, df_model):
    # Extract amino acid counts by position
    pos_counts_simulated = extract_aa_counts_by_pos(df_simulated)
    pos_counts_model = extract_aa_counts_by_pos(df_model)

    # TO DO: handle the case when the positions are not the same
    # Check if the positions are the same
    if pos_counts_simulated.keys() != pos_counts_model.keys():
        raise ValueError("The positions are not the same between the two datasets.")

    # Perform Fisher's exact test for each amino acid at each position
    p_values = []
    tests = []
    for pos in pos_counts_simulated.keys():
        for aa in pos_counts_simulated[pos].keys():
            p_value = run_fisher_test(aa, pos, pos_counts_simulated, pos_counts_model)
            p_values.append(p_value)
            tests.append((aa, pos))

    # TO DO: decide whether to adjust alpha for multiple testing
    adjusted_alpha = adjust_alpha_for_dependent_tests(df_simulated, pos_counts_simulated, alpha=0.05)
    #_, adjusted_p_values, _, _ = multipletests(p_values, alpha=adjusted_alpha, method='fdr_bh')
    _, adjusted_p_values, _, _ = multipletests(p_values, method='fdr_bh')

    significant_p_values = {}
    log_fold_changes = {}

    for i, (aa, pos) in enumerate(tests):
        if adjusted_p_values[i] < 0.05:
            significant_p_values[(aa, pos)] = adjusted_p_values[i]
            log_fold_change = calculate_log_fold_change(aa, pos, pos_counts_simulated, pos_counts_model)
            log_fold_changes[" ".join([aa, str(pos)])] = log_fold_change

    return significant_p_values, log_fold_changes


def run_fisher_test(aa, pos, pos_counts_simulated, pos_counts_model):
    count_aa_simulated, count_other_aa_simulated = get_aa_counts(aa, pos, pos_counts_simulated)
    count_aa_model, count_other_aa_model = get_aa_counts(aa, pos, pos_counts_model)

    contingency_table = [[count_aa_simulated, count_other_aa_simulated],
                         [count_aa_model, count_other_aa_model]]
    _, p_value = stats.fisher_exact(contingency_table)

    return p_value


def calculate_log_fold_change(aa, pos, pos_counts_simulated, pos_counts_model):
    count_aa_simulated, count_other_aa_simulated = get_aa_counts(aa, pos, pos_counts_simulated)
    count_aa_model, count_other_aa_model = get_aa_counts(aa, pos, pos_counts_model)
    # compute log fold change
    # current implementation avoids division by zero, it's just a hack. TO DO: find a better way
    count_aa_simulated = count_aa_simulated if count_aa_simulated > 0 else 1e-10
    fold_change = ((count_aa_model / (count_aa_model + count_other_aa_model)) /
                   (count_aa_simulated / (count_aa_simulated + count_other_aa_simulated)))
    fold_change = fold_change if fold_change > 0 else 1e-10
    log_fold_change = np.log2(fold_change)
    return log_fold_change


def make_logo_df(df):
    # Pivot the DataFrame to have positions as rows and amino acids as columns
    frequency_df = df.pivot(index='position', columns='amino acid', values='relative frequency')

    # Fill NaNs with 0 if there are any missing frequencies
    frequency_df = frequency_df.fillna(0)
    return frequency_df


def make_significance_df(df_model, significant_p_values):
    frequency_df = make_logo_df(df_model)
    # Initialize the significance DataFrame with zeros (non-significant by default)
    significance_df = frequency_df.copy()
    significance_df[:] = 0

    # Mark significant positions based on p-value threshold
    for (amino_acid, position), p_value in significant_p_values.items():
        if amino_acid in significance_df.columns and position in significance_df.index:
            significance_df.at[position, amino_acid] = 1

    return significance_df


def plot_logo_simulated(df_simulated, output_dir, num_sequences):
    frequency_df = make_logo_df(df_simulated)

    # Create a color dictionary for the amino acids
    color_dict = get_aa_color_map()

    # Set up a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    logo = logomaker.Logo(frequency_df, ax=ax)
    logo.style_glyphs(color_scheme=color_dict)

    # Customize plot appearance
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.style_xticks(rotation=90, fmt='%d')
    plt.title(f"Amino Acid Frequency Logo for simulated sequences ({num_sequences} sequences)")
    plt.ylabel("Frequency")
    plt.xlabel("Position")
    plt.savefig(f"{output_dir}/simulated_logo.png")


def plot_logo_model(df_model, significant_p_values, model_name, output_dir, num_sequences):
    frequency_df = make_logo_df(df_model)
    significance_df = make_significance_df(df_model, significant_p_values)
    num_significant = (significance_df == 1).sum().sum()

    # Initialize the logo plot
    fig, ax = plt.subplots(figsize=(10, 6))
    color_mapping = {1: 'red', 0: 'gray'}

    for position in frequency_df.index:
        logo = logomaker.Logo(frequency_df.loc[[position]], ax=ax)
        color_dict = {}
        for amino_acid in significance_df.columns:
            is_significant = significance_df.loc[position, amino_acid]
            color_dict[amino_acid] = color_mapping[is_significant]
        logo.style_glyphs(color_scheme=color_dict)
        logo.style_spines(visible=False)
        logo.style_spines(spines=['left', 'bottom'], visible=True)
        logo.style_xticks(rotation=90, fmt='%d')
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    #To do: find better way for shifting xticks instead of adding additional index
    xticks = [frequency_df.index[0]-1] + list(frequency_df.index)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.tick_params(axis='x', rotation=90)
    plt.title(f"Amino Acid Frequency Logo for {model_name} sequences (Sequence count: {num_sequences}, Red: {num_significant} significantly different)")
    plt.ylabel("Frequency")
    plt.xlabel("Position")
    plt.savefig(f"{output_dir}/{model_name}_logo.png")


def plot_logo_with_background_frequencies(model_freq, simulated_freq, output_dir, model_name):
    # TO DO: make two-directional logo
    foreground_df = make_logo_df(model_freq)
    background_df = make_logo_df(simulated_freq)

    # Create logo plot for foreground frequencies
    fig, ax = plt.subplots(figsize=(10, 6))
    return NotImplemented


def plot_log_fold_changes(log_fold_changes, output_dir, model_name):
    # Create a DataFrame from the fold changes
    df_fold_changes = pd.DataFrame(log_fold_changes.items(), columns=['AA + pos', 'log(fold change)'])

    # Create a bar plot
    figure = px.bar(df_fold_changes, x='AA + pos', y='log(fold change)',
                 title=f'Fold changes of significantly different amino acid counts for {model_name} vs. simulated',
                 color_discrete_map=PlotlyUtil.get_amino_acid_color_map(),
                 labels={'log(fold change)': 'log(Fold Change)', 'AA + pos': 'Amino Acid and Position'})

    # Set bar
    figure.update_layout(height=600, width=1000, bargap=0.2)

    # Save the plot
    figure.write_html(f"{output_dir}/log_fold_changes.html")


def main():
    parser = argparse.ArgumentParser(description='Compare two CDR3 length specific amino acid frequency distribution TSV files.')
    parser.add_argument('file1', type=str, help='Path to the first data file.')
    parser.add_argument('file2', type=str, help='Path to the second data file.')
    parser.add_argument('filtered_sequences_lengths', type=int, help='Number of filtered sequences lengths.')
    parser.add_argument('output_dir', type=str, default='.', help='Output directory to save plots.')
    parser.add_argument('model_name', type=str, default='.', help='Name of the model.')

    args = parser.parse_args()

    # Get dataframes for the two files
    df_simulated, num_sequences_simulated = get_aa_counts_frequencies_df(args.file1, args.filtered_sequences_lengths)
    df_model, num_sequences_model = get_aa_counts_frequencies_df(args.file2, args.filtered_sequences_lengths)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # if given length is not present, return log file
    if df_simulated is None or df_model is None:
        with open(args.output_dir + "/aa_freq_plotting.log", "w") as f:
            f.write("No sequence data found for the given length.")
        return

    # Run Fisher's exact test
    significant_p_values, fold_changes = find_significant_amino_acids(df_simulated, df_model)

    # Plot logos
    plot_logo_simulated(df_simulated, args.output_dir, num_sequences_simulated)
    plot_logo_model(df_model, significant_p_values, args.model_name, args.output_dir, num_sequences_model)

    # Plot log fold changes
    plot_log_fold_changes(fold_changes, args.output_dir, args.model_name)


if __name__ == "__main__":
    main()
