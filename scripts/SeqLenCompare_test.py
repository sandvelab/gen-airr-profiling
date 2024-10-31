import argparse
import pandas as pd
import plotly.graph_objects as go


def load_data(file_paths_list):
    all_data = []
    for path in file_paths_list:
        data = pd.read_csv(path)
        all_data.append(data)
    merged_df = create_merged_dataframe(all_data)
    return merged_df


def create_merged_dataframe(datasets_list):
    # Rename the 'counts' column in each DataFrame to make them unique
    for i, df in enumerate(datasets_list):
        df.rename(columns={'counts': f'counts_{i + 1}'}, inplace=True)

    # Merge all DataFrames on 'sequence_lengths' column
    merged_df = None
    for i, df in enumerate(datasets_list):
        if i == 0:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, on='sequence_lengths', how='outer').fillna(0)

    # Convert counts to frequencies
    for i in range(1, len(datasets_list) + 1):
        merged_df[f'counts_{i}'] = merged_df[f'counts_{i}'] / merged_df[f'counts_{i}'].sum()
        # rename the columns from counts to frequencies
        merged_df.rename(columns={f'counts_{i}': f'frequencies_{i}'}, inplace=True)

    # Calculate the mean and standard deviation across the freq columns for each sequence length
    freq_columns = [col for col in merged_df.columns if 'frequencies_' in col]
    merged_df['mean'] = merged_df[freq_columns].mean(axis=1)
    merged_df['std_dev'] = merged_df[freq_columns].std(axis=1)

    return merged_df


def plot_test_gen_seq_len_distribution(merged_simulations_df, generated_data_df, model_name, image_file):

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
        x=generated_data_df['sequence_lengths'],
        y=generated_data_df['mean'],
        name=f'Generated ({model_name})',
        error_y=dict(
            type='data',
            array=merged_simulations_df['std_dev'],
            visible=True  # Explicitly hide the error bars for model data
        )
    )

    # Create the figure with both traces
    figure = go.Figure(data=[trace1, trace2])

    # Update the layout
    x_tick_vals = pd.merge(merged_simulations_df['sequence_lengths'],
                           generated_data_df['sequence_lengths'],
                           how='outer')['sequence_lengths']
    figure.update_layout(barmode='group', xaxis=dict(tickmode='array', tickvals=x_tick_vals),
                         yaxis=dict(tickmode='array'),
                         template="plotly_white",
                         title=f"Sequence Length Distributions of simulated test data and {model_name} data",
                         xaxis_title="Sequence lengths", yaxis_title="Frequency",
                         font=dict(size=22))

    figure.write_html(image_file)


def main():
    parser = argparse.ArgumentParser(description='Plot sequence length distributions for test and model data.')
    parser.add_argument('--simulated_data_path', nargs='+', type=str, required=True,help='Path to the test directory.')
    parser.add_argument('--generated_data_path', nargs='+', type=str, required=True, help='Path to the generated sequences.')
    parser.add_argument('--image_output_file', type=str, default='.', required=True, help='Output directory for the results.')
    parser.add_argument('--model_name', type=str, default='.', required=True, help='Name of the model.')

    args = parser.parse_args()

    test_data = load_data(args.simulated_data_path)
    generated_data = load_data(args.generated_data_path)

    plot_test_gen_seq_len_distribution(test_data, generated_data, args.model_name, args.image_output_file)


if __name__ == "__main__":
    main()
