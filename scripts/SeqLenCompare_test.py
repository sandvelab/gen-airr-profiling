import argparse
import pandas as pd
import plotly.graph_objects as go


def load_csv(file_path):
    # Load the TSV file into a DataFrame
    return pd.read_csv(file_path)


def create_merged_dataframe(datasets_list):
    # Rename the 'counts' column in each DataFrame to make them unique
    for i, df in enumerate(datasets_list):
        df.rename(columns={'counts': f'counts_{i + 1}'}, inplace=True)

    # Merge all DataFrames on 'sequence_lengths' column and fill missing values with 0
    merged_df = pd.concat(datasets_list, axis=1).fillna(0)

    # If 'sequence_lengths' appears multiple times, keep only one
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    merged_df = merged_df[merged_df['sequence_lengths'] != 0]

    # Calculate the mean and standard deviation across the counts columns for each sequence length
    count_columns = [col for col in merged_df.columns if 'counts_' in col]
    merged_df['mean'] = merged_df[count_columns].mean(axis=1)
    merged_df['std_dev'] = merged_df[count_columns].std(axis=1)

    return merged_df


def plot_test_gen_seq_len_distribution(merged_simulations_df, model_df, model_name, image_file):

    # Create the bar traces separately
    trace1 = go.Bar(
        x=merged_simulations_df['sequence_lengths'],
        y=merged_simulations_df['mean'],
        name='Simulated (test)',
        error_y=dict(
            type='data',
            array=merged_simulations_df['std_dev'],
            visible=True
        )
    )

    trace2 = go.Bar(
        x=model_df['sequence_lengths'],
        y=model_df['counts'],
        name=model_name,
        error_y=dict(
            type='data',
            visible=False  # Explicitly hide the error bars for model data
        )
    )

    # Create the figure with both traces
    figure = go.Figure(data=[trace1, trace2])

    # Update the layout
    x_tick_vals = pd.merge(merged_simulations_df['sequence_lengths'],
                           model_df['sequence_lengths'],
                           how='outer')['sequence_lengths']
    figure.update_layout(barmode='group', xaxis=dict(tickmode='array', tickvals=x_tick_vals),
                         yaxis=dict(tickmode='array'),
                         template="plotly_white",
                         title=f"Sequence Length Distributions of simulated test data and {model_name} data",
                         xaxis_title="Sequence lengths", yaxis_title="Counts",
                         font=dict(size=22))

    figure.write_html(image_file)


def main():
    parser = argparse.ArgumentParser(description='Plot sequence length distributions for test and model data.')
    parser.add_argument('simulated_data_path', nargs='+', type=str, help='Path to the directory.')
    parser.add_argument('generated_data_path', type=str, help='Path to the generated sequences.')
    parser.add_argument('image_output_file', type=str, default='.', help='Output directory for the results.')
    parser.add_argument('model_name', type=str, default='.', help='Name of the model.')

    args = parser.parse_args()

    # Load test data
    all_data = []
    # Walk through the directory
    for path in args.simulated_data_path:
        data = load_csv(path)
        all_data.append(data)
    test_data = create_merged_dataframe(all_data)

    # Load generated data
    generated_data = load_csv(args.generated_data_path)

    # Plot the sequence length distribution
    plot_test_gen_seq_len_distribution(test_data, generated_data, args.model_name, args.image_output_file)


if __name__ == "__main__":
    main()
