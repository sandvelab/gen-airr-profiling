import os
import pandas as pd

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.compairr_utils import run_compairr_existence
from gen_airr_bm.utils.file_utils import get_sequence_files

import plotly.graph_objects as go


def run_innovation_distances_analysis(analysis_config: AnalysisConfig) -> None:
    """ Runs innovation sequence distances analysis on the generated innovative and reference sequences.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        None
    """
    print("Running innovation sequence distances analysis")

    innovation_sequences_dir = save_innovative_sequences_for_compairr(analysis_config)
    all_distance_dfs = compute_distances_to_train(analysis_config, innovation_sequences_dir)
    plot_innovation_distances(analysis_config, all_distance_dfs)


def save_innovative_sequences_for_compairr(analysis_config: AnalysisConfig) -> str:
    """ Saves the innovative sequences for each model and dataset in a format suitable for CompAIRR.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        innovative_sequences_dir (str): Directory where the innovative sequences are saved for CompAIRR analysis.
    """
    innovative_sequences_dir = f"{analysis_config.root_output_dir}/innovation_overlap_compairr_sequences_split/"
    os.makedirs(innovative_sequences_dir, exist_ok=True)

    test_reference = 'test'
    for model in analysis_config.model_names:
        os.makedirs(f"{innovative_sequences_dir}/{model}", exist_ok=True)
        comparison_files_dir = get_sequence_files(analysis_config, model, test_reference)
        for ref_file, gen_files in comparison_files_dir.items():
            for gen_file in gen_files:
                file_name = os.path.basename(gen_file)
                innovative_sequences_file = f"{innovative_sequences_dir}/{model}/{file_name}"

                gen_df = pd.read_csv(gen_file, sep='\t')
                ref_file_df = pd.read_csv(ref_file, sep='\t')
                innovative_sequences_df = gen_df[gen_df['junction_aa'].isin(ref_file_df['junction_aa'])]
                innovative_sequences_df.to_csv(innovative_sequences_file, sep='\t', index=False)

    return innovative_sequences_dir


def compute_distances_to_train(analysis_config: AnalysisConfig, innovation_sequences_dir: str) -> tuple:
    """
    Computes the distances from the innovative model sequences and tes sequences to the training sequences
    at distances 1-3 using CompAIRR.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        innovation_sequences_dir (str): Directory where the innovative sequences are saved for CompAIRR analysis.
    Returns:
        all_distance_dfs (dict): Dict of DataFrames per model and test containing counts of innovative sequences
                                  at each distance to the training set.
    """
    all_distance_dfs = {}

    for model in analysis_config.model_names:
        compairr_output_dir = f"{analysis_config.analysis_output_dir}/compairr_output/{model}"
        os.makedirs(compairr_output_dir, exist_ok=True)

        gen_train_overlap_counts = {}
        test_train_overlap_counts = {}  # cached per dataset_name

        for innovation_gen_file_split in os.listdir(f"{innovation_sequences_dir}/{model}"):
            split_base_name = os.path.splitext(innovation_gen_file_split)[0]
            dataset_name = split_base_name.rsplit('_', 1)[0]

            gen_file = f"{innovation_sequences_dir}/{model}/{innovation_gen_file_split}"
            train_file = f"{analysis_config.root_output_dir}/train_compairr_sequences/{dataset_name}.tsv"
            test_file = f"{analysis_config.root_output_dir}/test_compairr_sequences/{dataset_name}.tsv"

            gen_train_counts, gen_n_seqs = _compute_overlap_counts(
                compairr_output_dir=compairr_output_dir,
                search_for_file=gen_file,
                search_in_file=train_file,
                identifier_prefix=f"{split_base_name}_{model}",
                distances=[1, 2, 3]
            )
            gen_train_counts[">3"] = gen_n_seqs - sum(gen_train_counts[str(d)] for d in [1, 2, 3])
            gen_train_counts["n_sequences"] = gen_n_seqs
            gen_train_overlap_counts[split_base_name] = gen_train_counts

            # --- test-train distances (skip if already computed for this dataset) ---
            if dataset_name not in test_train_overlap_counts:
                test_train_counts, test_n_seqs = _compute_overlap_counts(
                    compairr_output_dir=compairr_output_dir,
                    search_for_file=test_file,
                    search_in_file=train_file,
                    identifier_prefix=f"{dataset_name}_test",
                    distances=[1, 2, 3]
                )
                test_train_counts[">3"] = test_n_seqs - sum(test_train_counts[str(d)] for d in [1, 2, 3])
                test_train_counts["n_sequences"] = test_n_seqs
                test_train_overlap_counts[dataset_name] = test_train_counts

        gen_train_df = pd.DataFrame.from_dict(gen_train_overlap_counts, orient='index')
        test_train_df = pd.DataFrame.from_dict(test_train_overlap_counts, orient='index')

        all_distance_dfs[model] = gen_train_df
        all_distance_dfs["test"] = test_train_df

    return all_distance_dfs


def _compute_overlap_counts(compairr_output_dir: str, search_for_file: str, search_in_file: str,
                                                    identifier_prefix: str, distances: list) -> tuple:
    """
    Runs CompAIRR at each distance and returns exclusive per-distance counts and total sequence count.
    """
    counts = {}
    last_result = None

    for d in distances:
        identifier = f"{identifier_prefix}_{d}"
        run_compairr_existence(
            compairr_output_dir=compairr_output_dir,
            search_for_file=search_for_file,
            search_in_file=search_in_file,
            file_identifier=identifier,
            allowed_mismatches=d,
            indels=False
        )
        result = pd.read_csv(f"{compairr_output_dir}/{identifier}_overlap.tsv", sep='\t',
                             names=['sequence_id', 'overlap_count'], header=0)
        n = result[result['overlap_count'] != 0].shape[0]
        counts[str(d)] = n if d == distances[0] else n - counts[str(d - 1)]
        last_result = result

    n_sequences = last_result.shape[0]
    return counts, n_sequences


def plot_innovation_distances(analysis_config: AnalysisConfig, plotting_dfs: pd.DataFrame) -> None:
    """
    Plots the distribution of distances from innovative sequences to the training set and from test sequences to the training set.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        plotting_dfs (pd.DataFrame): DataFrame containing counts of sequences at each distance to the training set.
    Returns:
        None
    """
    fig = make_distance_figure(plotting_dfs, title='Average sequence counts by distance to nearest train sequence')
    png_path = f"{analysis_config.analysis_output_dir}/innovation_distances_plot.png"
    fig.write_image(png_path)
    print(f"Plot saved at: {png_path}")

    dataset_base_names = set()
    for key, df in plotting_dfs.items():
        if key != "test":
            dataset_base_names.update(df.index.str.rsplit('_', n=1).str[0])

    for dataset in sorted(dataset_base_names):
        subset_mask = {model: df.index.str.startswith(dataset)
                       for model, df in plotting_dfs.items()}
        fig = make_distance_figure(plotting_dfs,
                                   title=f'Average sequence counts by distance to nearest train sequence for dataset {dataset}',
                                    subset_mask=subset_mask)
        png_path = f"{analysis_config.analysis_output_dir}/innovation_distances_{dataset}_plot.png"
        fig.write_image(png_path)
        print(f"Plot saved at: {png_path}")


def make_distance_figure(plotting_dfs: pd.DataFrame, title, subset_mask=None) -> go.Figure:
    """
    Creates a Plotly figure comparing the average sequence counts at each distance to the training set for each model and test.
    If subset_mask is provided, it filters the data for each model according to the mask before computing the means and stds for the plot.
    Args:
        plotting_dfs (pd.DataFrame): DataFrame containing counts of sequences at each distance to the training set for each model and test.
        title (str): Title for the plot.
        subset_mask (dict, optional): Dict with model names as keys and boolean masks as values to filter the data for each model before plotting. Defaults to None (no filtering).
    Returns:
        go.Figure: Plotly figure object ready for display or saving.
    """
    distance_cols = ['1', '2', '3', '>3']

    fig = go.Figure()
    for model, df in plotting_dfs.items():
        subset_df = df[subset_mask[model]] if subset_mask else df
        means = subset_df[distance_cols].mean()
        stds = subset_df[distance_cols].std()

        fig.add_trace(go.Scatter(
            x=distance_cols,
            y=means,
            error_y=dict(type='data', array=stds, visible=True),
            mode='lines+markers',
            name=model
        ))

    fig.data = sorted(fig.data, key=lambda trace: (0 if trace.name == "test" else 1, trace.name))
    fig.update_layout(
        title=title,
        xaxis_title='Distance to nearest training sequence',
        yaxis_title='Avg. sequence count',
        template='plotly_white'
    )
    return fig


