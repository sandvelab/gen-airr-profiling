import argparse
import pandas as pd
from scipy.stats import entropy
import numpy as np
import os
import plotly.express as px
from scipy import stats
import logomaker
import matplotlib.pyplot as plt
#from plotnineseqsuite import pyggseqlogo
from pyggseqlogo import ggseqlogo
from plotnineseqsuite import geom_logo

from immuneML.reports.PlotlyUtil import PlotlyUtil


def load_tsv(file_path):
    # Load the TSV file into a DataFrame
    return pd.read_csv(file_path, sep='\t')


def format_positions(positions):
    return [str(int(pos)) if pos.is_integer() else str(pos) for pos in positions.astype(float)]


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


def get_counts(aa, pos, pos_counts):
    # get the count of the amino acid at the position
    count_aa = pos_counts[pos][aa]

    # get the count of all other amino acids at the position
    count_other_aa = sum(pos_counts[pos].values()) - count_aa
    return count_aa, count_other_aa


def fishers_exact_test(simulated_count1, simulated_count2, model_count1, model_count2):
    # Perform Fisher's exact test
    odds_ratio, p_value = stats.fisher_exact([[simulated_count1, model_count1], [simulated_count2, model_count2]])
    return p_value


def run_fishers_exact_test(df_simulated, df_model):
    # Extract amino acid counts by position
    pos_counts_simulated = extract_aa_counts_by_pos(df_simulated)
    pos_counts_model = extract_aa_counts_by_pos(df_model)

    # Check if the positions are the same
    if pos_counts_simulated.keys() != pos_counts_model.keys():
        raise ValueError("The positions are not the same between the two datasets.")

    # Perform Fisher's exact test for each amino acid at each position
    significant_p_values = {}
    log_fold_changes = {}
    for pos in pos_counts_simulated.keys():
        for aa in pos_counts_simulated[pos].keys():

            count_aa_simulated, count_other_aa_simulated = get_counts(aa, pos, pos_counts_simulated)
            count_aa_model, count_other_aa_model = get_counts(aa, pos, pos_counts_model)
            p_value = fishers_exact_test(count_aa_simulated, count_other_aa_simulated,
                                         count_aa_model, count_other_aa_model)

            if p_value < 0.05:
                significant_p_values[(aa, pos)] = p_value
                # compute fold change
                # Get relative frequencies for aa and position
                simulated_aa_rel_freq = df_simulated[(df_simulated['amino acid'] == aa) & (df_simulated['position'] == pos)]['relative frequency'].values[0]
                model_aa_rel_freq = df_model[(df_model['amino acid'] == aa) & (df_model['position'] == pos)]['relative frequency'].values[0]
                #log_fold_change = np.log(model_aa_rel_freq / simulated_aa_rel_freq)
                log_fold_change = np.log((count_aa_model/count_other_aa_model) / (count_aa_simulated/count_other_aa_simulated))
                log_fold_changes[" ".join([aa, str(pos)])] = log_fold_change


    # Filter df by significant p-values
    df_simulated_significant = df_simulated[df_simulated.apply(lambda row: (row['amino acid'], row['position']) in significant_p_values, axis=1)]
    df_model_significant = df_model[df_model.apply(lambda row: (row['amino acid'], row['position']) in significant_p_values, axis=1)]

    # Check that length of dataframe is equal to length of significant p-values
    if len(df_simulated_significant) != len(significant_p_values):
        raise ValueError("Length of filtered dataframe is not equal to length of significant p-values.")

    return df_simulated_significant, df_model_significant, significant_p_values, log_fold_changes


def combined_plot_aa_compare_significant_aa_pos(df_simulated_significant, df_model_significant,
                                                output_dir, model_name):
    # Combine the two dataframes
    df_combine = {"Simulated": df_simulated_significant, model_name: df_model_significant}
    df_combine = pd.concat(df_combine, names=["dataset"]).reset_index(level=0)

    # Sort the dataframe by position
    df_combine.sort_values(by=["position"], ascending=True, inplace=True)
    df_combine['position'] = format_positions(df_combine['position'])

    # Create a stacked bar plot
    figure = px.bar(df_combine, x='position', y='relative frequency', color='amino acid', barmode='group',
                 facet_col='dataset',
                 title=f'Amino acid relative frequencies for significantly different simulated and {model_name} amino acid counts',
                 color_discrete_map=PlotlyUtil.get_amino_acid_color_map(),
                 labels={'relative frequency': 'Relative Frequency', 'position': 'Position', 'amino acid': 'Amino Acid'})

    # Update layout for better visualization
    figure.update_layout(height=600, width=1000, bargap=0.2)
    # Set x-axis category order based on the unique 'position' values in the order they appear in the DataFrame
    unique_positions = df_combine['position'].unique()
    figure.update_xaxes(categoryorder="array", categoryarray=unique_positions)

    # Save the plot
    figure.show()
    figure.write_html(f"{output_dir}/aa_freq_compare.html")


def combined_plot_aa_compare_significant_aa_pos2(df_simulated_significant, df_model_significant,
                                                output_dir, model_name):
    # Combine the two dataframes
    df_combine = {"Simulated": df_simulated_significant, model_name: df_model_significant}
    df_combine = pd.concat(df_combine, names=["dataset"]).reset_index(level=0)

    # Sort the dataframe by position
    df_combine.sort_values(by=["position"], ascending=True, inplace=True)
    df_combine['position'] = format_positions(df_combine['position'])

    # Create a stacked bar plot, colored by amino acid
    figure = px.bar(df_combine, x='position', y='relative frequency', color='amino acid',
                 pattern_shape='dataset', pattern_shape_sequence=["", "/"],# "" for Simulated (regular), "/" for Model (striped)
                 barmode='stack',
                 title='Simulated vs Model amino acid frequencies Stacked by Amino Acid and Grouped by Dataset',
                 color_discrete_map=PlotlyUtil.get_amino_acid_color_map(),
                 labels={'relative frequency': 'Relative Frequency', 'position': 'Position', 'amino acid': 'Amino Acid', 'dataset': 'Dataset'})

    # Group bars by position and dataset
    df_combine['position_with_dataset'] = df_combine['position'].astype(str) + ' (' + df_combine['dataset'] + ')'

    # Update plot to group by position and dataset with pattern distinction for datasets
    figure = px.bar(df_combine, x='position_with_dataset', y='relative frequency', color='amino acid',
                 pattern_shape='dataset', pattern_shape_sequence=["", "/"],
                 barmode='group',
                 title='Simulated vs Model amino acid frequencies Stacked by Amino Acid with Striped Model Bars',
                 color_discrete_map=PlotlyUtil.get_amino_acid_color_map(),
                 labels={'relative frequency': 'Relative Frequency', 'position_with_dataset': 'Position (Dataset)', 'amino acid': 'Amino Acid'})

    # Save the plot
    figure.write_html(f"{output_dir}/aa_freq_compare_2.html")


def combined_plot_aa_compare_significant_aa_pos3(df_simulated_significant, df_model_significant,
                                                output_dir, model_name):
    # Combine the two dataframes
    df_combine = {"Simulated": df_simulated_significant, model_name: df_model_significant}
    df_combine = pd.concat(df_combine, names=["dataset"]).reset_index(level=0)

    # Sort the dataframe by position
    df_combine.sort_values(by=["position"], ascending=True, inplace=True)
    df_combine['position'] = format_positions(df_combine['position'])

    # Create a grouped and stacked bar plot
    fig = px.bar(df_combine, x='position', y='relative frequency', color='dataset', barmode='group',
                 facet_row=None, facet_col=None,
                 title='Simulated vs Model amino acid frequencies Stacked by Amino Acid at Each Position',
                 labels={'relative frequency': 'Relative Frequency', 'position': 'Position', 'amino acid': 'Amino Acid'},
                 text='amino acid')

    # Group bars by dataset
    fig.update_traces(texttemplate='%{text}', textposition='inside')
    fig.update_layout(barmode='group', xaxis={'categoryorder': 'total ascending'},
                      coloraxis_colorbar=dict(title="Amino Acids"))

    # Save the plot
    fig.write_html(f"{output_dir}/aa_freq_compare_3.html")


def plot_fold_changes(log_fold_changes, output_dir, model_name):
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
    #figure.show()
    figure.write_html(f"{output_dir}/log_fold_changes.html")


def make_logo_df(df):
    # Pivot the DataFrame to have positions as rows and amino acids as columns
    frequency_df = df.pivot(index='position', columns='amino acid', values='relative frequency')

    # Fill NaNs with 0 if there are any missing frequencies
    frequency_df = frequency_df.fillna(0)
    return frequency_df

def make_significance_df(frequency_df, significant_p_values):
    # Initialize the significance DataFrame with zeros (non-significant by default)
    significance_df = frequency_df.copy()
    significance_df[:] = False  # Set all values to 0 initially

    # Mark significant positions based on p-value threshold
    for (amino_acid, position), p_value in significant_p_values.items():
        if amino_acid in significance_df.columns and position in significance_df.index:
            significance_df.at[position, amino_acid] = True

    return significance_df

def plot_logo(frequency_df, significance_df):
    # Initialize the logo plot
    fig, ax = plt.subplots(figsize=(10, 5))
    color_mapping = {True: 'red', False: 'gray'}

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
    plt.title("Amino Acid Frequency Logo with Significance Highlighting")
    plt.ylabel("Frequency")
    plt.xlabel("Position")
    plt.show()


def plot_ggseqlogo(frequency_df, significance_df):
    color_dict = significance_df.replace({True: "red", False: "gray"}).to_dict()
    ggseqlogo(frequency_df, colors=color_dict)

def main():
    parser = argparse.ArgumentParser(description='Compare two CDR3 length specific amino acid frequency distribution TSV files.')
    parser.add_argument('file1', type=str, help='Path to the first TSV file.')
    parser.add_argument('file2', type=str, help='Path to the second TSV file.')
    parser.add_argument('output_dir', type=str, default='.', help='Output directory to save plot.')
    parser.add_argument('model_name', type=str, default='.', help='Name of the model.')

    args = parser.parse_args()

    # Load the TSV files
    df_simulated = load_tsv(args.file1)
    df_model = load_tsv(args.file2)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Run Fisher's exact test
    df_simulated_significant, df_model_significant, significant_p_values, fold_changes = run_fishers_exact_test(df_simulated, df_model)
    # Plot the significant amino acids and positions
    combined_plot_aa_compare_significant_aa_pos(df_simulated_significant, df_model_significant,
                                                 args.output_dir, args.model_name)
    combined_plot_aa_compare_significant_aa_pos2(df_simulated_significant, df_model_significant,
                                                args.output_dir, args.model_name)
    combined_plot_aa_compare_significant_aa_pos3(df_simulated_significant, df_model_significant,
                                                args.output_dir, args.model_name)

    # Plot the fold changes
    plot_fold_changes(fold_changes, args.output_dir, args.model_name)


    # Plot logo
    frequency_df = make_logo_df(df_model)
    #print(frequency_df)
    #print(make_significance_df(frequency_df, significant_p_values))

    #plot_ggseqlogo(frequency_df, make_significance_df(frequency_df, significant_p_values))

    plot_logo(frequency_df, make_significance_df(frequency_df, significant_p_values))


if __name__ == "__main__":
    main()
