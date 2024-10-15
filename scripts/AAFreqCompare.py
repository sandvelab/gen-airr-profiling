import argparse
import pandas as pd
from scipy.stats import entropy
import numpy as np
import plotly.express as px


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


def plot_aa_relative_frequencies(file, model_name):
    df = load_tsv(file)
    df.sort_values(by=["position"], ascending=True, inplace=True)
    df['position'] = format_positions(df['position'])

    # Define the custom order of amino acids
    custom_amino_acid_order = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'Y', 'W', 'R',
                               'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'P', 'G', 'C']

    # Assigning colors based on the group
    color_map = {
         'A': '#FFA500', 'V': '#FFA500', 'I': '#FFA500', 'L': '#FFA500', 'M': '#FFA500',
        'F': '#FFA500', 'W': '#FFA500', 'Y': '#FFA500',
        'R': '#FF4500', 'H': '#FF4500', 'K': '#FF4500',  # Positively Charged (Red)
        'D': '#1E90FF', 'E': '#1E90FF',  # Negatively Charged (Blue)
        'S': '#32CD32', 'T': '#32CD32', 'N': '#32CD32', 'Q': '#32CD32',  # Polar Uncharged (Green)
        'P': '#9370DB', 'G': '#9370DB', 'C': '#9370DB',  # Special (Purple)
    }

    # Create a bar plot
    fig = px.bar(
        df,
        x='position',
        y='relative frequency',
        color='amino acid',
        category_orders={"amino acid": custom_amino_acid_order},
        color_discrete_map=color_map,
        title=f'Relative Frequency of Amino Acids at each Position for {model_name} sequences',
        labels={'relative frequency': 'Relative Frequency', 'position': 'Position'},
    )

    # Update layout for better visualization
    fig.update_layout(
        barmode='stack',
        xaxis_title="Position",
        yaxis_title="Relative Frequency",
        legend_title="Amino Acid"
    )

    # Show the plot
    #fig.show()


def format_positions(positions):
    return [str(int(pos)) if pos.is_integer() else str(pos) for pos in positions.astype(float)]


def plot_aa_freq_compare(file1, file2, model_name, output_file):
    # Load the distributions
    df1 = load_tsv(file1)
    df2 = load_tsv(file2)

    for df in [df1, df2]:
        df.sort_values(by=["position"], ascending=True, inplace=True)
        df['position'] = format_positions(df['position'])

    # Filter to only have amino acid A
    aa = "Y"
    df1 = df1[df1['amino acid'] == aa]
    df2 = df2[df2['amino acid'] == aa]

    df_combine = {"Simulated (train)": df1, model_name: df2}
    df_combine = pd.concat(df_combine, names=["dataset"]).reset_index(level=0)

    # Create distribution plot with px
    figure = px.bar(df_combine, x="position", y="relative frequency", color='dataset')

    figure.update_layout(barmode='group', xaxis=dict(tickmode='array', tickvals=df_combine["position"]),
                         yaxis=dict(tickmode='array'),
                         template="plotly_white",
                         title=f"AA frequency distribution of {aa} for simulated train data and {model_name} data",
                         font=dict(size=22))

    #figure.write_html(f"results/dataset1/analyses/aa_res/{model_name}/AAFreqCompare_{AA}.html")
    #figure.show()


def main():
    parser = argparse.ArgumentParser(description='Compute KL Divergence between two amino acid frequency distribution TSV files.')
    parser.add_argument('file1', type=str, help='Path to the first TSV file.')
    parser.add_argument('file2', type=str, help='Path to the second TSV file.')
    parser.add_argument('output_file', type=str, default='.', help='Output directory for the results.')
    parser.add_argument('model_name', type=str, default='.', help='Name of the model.')

    args = parser.parse_args()

    # Compute KL divergence
    kl_divergence = kl_aa_compare(args.file1, args.file2)

    # Output the results
    with open(args.output_file, 'w') as f:
        f.write(str(kl_divergence))

    # Plot the amino acid relative frequencies
    plot_aa_relative_frequencies(args.file1, "train")
    plot_aa_relative_frequencies(args.file2, args.model_name)
    #plot_aa_freq_compare(args.file1, args.file2, args.model_name, args.output_file)


if __name__ == "__main__":
    main()
