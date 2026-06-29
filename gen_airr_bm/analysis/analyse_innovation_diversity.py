import os
import pandas as pd

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.compairr_utils import run_compairr_existence, run_compairr_cluster
from gen_airr_bm.utils.file_utils import get_sequence_files

import plotly.graph_objects as go
import plotly.express as px

from gen_airr_bm.utils.plotting_utils import get_collection_specification_for_title


def run_innovation_diversity_analysis(analysis_config: AnalysisConfig) -> None:
    """ Runs innovation sequence diversity analysis on the generated innovative and reference sequences.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        None
    """
    print("Running innovation sequence diversity analysis")

    innovation_sequences_dir = save_innovative_sequences_for_compairr(analysis_config)
    nn_counts_innovation_dir = f"{analysis_config.analysis_output_dir}/nn_counts_innovation"
    os.makedirs(nn_counts_innovation_dir, exist_ok=True)
    innovation_nn_plotting_data = count_nearest_neighbors(analysis_config, innovation_sequences_dir,
                                                          nn_counts_innovation_dir)
    plot_nn_counts_across_datasets(analysis_config, innovation_nn_plotting_data, nn_counts_innovation_dir,
                                   innovation=True)
    cluster_counts_plotting_data = cluster_innovation_sequences(analysis_config, innovation_sequences_dir)
    plot_cluster_counts(analysis_config, cluster_counts_plotting_data)

    full_gen_sequences_dir = f"{analysis_config.root_output_dir}/novel_generated_compairr_sequences_split"
    nn_counts_full_gen_dir = f"{analysis_config.analysis_output_dir}/nn_counts_full_gen"
    os.makedirs(nn_counts_full_gen_dir, exist_ok=True)
    full_gen_nn_plotting_data = count_nearest_neighbors(analysis_config, full_gen_sequences_dir, nn_counts_full_gen_dir)
    plot_nn_counts_across_datasets(analysis_config, full_gen_nn_plotting_data, nn_counts_full_gen_dir, innovation=False)


def save_innovative_sequences_for_compairr(analysis_config: AnalysisConfig) -> str:
    """ Saves the innovative sequences for each model and dataset in a format suitable for CompAIRR.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        innovative_sequences_dir (str): Directory where the innovative sequences are saved for CompAIRR analysis.
    """
    innovative_sequences_dir = f"{analysis_config.root_output_dir}/innovation_unique_overlap_compairr_sequences_split"
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


def count_nearest_neighbors(analysis_config: AnalysisConfig, sequences_dir: str,
                            output_dir: str) -> dict:
    """
    Computes the distances from the model sequences and test sequences to the training sequences
    at distances 1-3 using CompAIRR.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        sequences_dir (str): Directory where the generated sequences are saved for CompAIRR analysis.
        output_dir (str): Directory to store CompAIRR output dir.
    Returns:
        all_distance_dfs (dict): Dict of DataFrames per model and test containing counts of innovative sequences
                                  at each distance to the training set.
    """
    all_distance_dfs = {}
    for model in analysis_config.model_names:
        compairr_output_dir = f"{output_dir}/compairr_output/{model}"
        os.makedirs(compairr_output_dir, exist_ok=True)

        gen_train_overlap_counts = {}
        test_train_overlap_counts = {}

        for innovation_gen_file_split in os.listdir(f"{sequences_dir}/{model}"):
            dataset_split_name = os.path.splitext(innovation_gen_file_split)[0]
            dataset_name = dataset_split_name.rsplit('_', 1)[0]

            gen_file = f"{sequences_dir}/{model}/{innovation_gen_file_split}"
            train_file = f"{analysis_config.root_output_dir}/train_compairr_sequences/{dataset_name}.tsv"
            test_file = f"{analysis_config.root_output_dir}/test_compairr_sequences/{dataset_name}.tsv"

            gen_train_counts = compute_nearest_neighbor_counts(
                compairr_output_dir=compairr_output_dir,
                search_for_file=gen_file,
                search_in_file=train_file,
                identifier_prefix=f"{dataset_split_name}_{model}",
                distances=[1, 2, 3]
            )
            gen_train_overlap_counts[dataset_split_name] = gen_train_counts

            if dataset_name not in test_train_overlap_counts:
                test_train_counts = compute_nearest_neighbor_counts(
                    compairr_output_dir=compairr_output_dir,
                    search_for_file=test_file,
                    search_in_file=train_file,
                    identifier_prefix=f"{dataset_name}_test",
                    distances=[1, 2, 3]
                )
                test_train_overlap_counts[dataset_name] = test_train_counts

        gen_train_df = pd.DataFrame.from_dict(gen_train_overlap_counts, orient='index')
        test_train_df = pd.DataFrame.from_dict(test_train_overlap_counts, orient='index')

        all_distance_dfs[model] = gen_train_df
        all_distance_dfs["test"] = test_train_df

    return all_distance_dfs


def compute_nearest_neighbor_counts(compairr_output_dir: str, search_for_file: str, search_in_file: str,
                                    identifier_prefix: str, distances: list) -> dict:
    """
    Runs CompAIRR at each distance and returns per-distance counts and total sequence count.
    Args:
        compairr_output_dir (str): Directory to store CompAIRR output files.
        search_for_file (str): Path to the file of sequences for which to search for existence in another sequence set.
        search_in_file (str): Path to the file to search for existence in.
        identifier_prefix (str): Base name for output files.
        distances (list): List of distances to compute counts for (e.g., [1, 2, 3]).
    Returns:
        counts (dict): Dict with distance as key and count of sequences at that distance as value.
    """
    counts = {}
    prev_result = None
    prev_cumulative_n = 0

    for d in distances:
        identifier = f"{identifier_prefix}_d{d}"
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
        cumulative_n = result[result['overlap_count'] != 0].shape[0]
        counts[str(d)] = cumulative_n - prev_cumulative_n
        prev_cumulative_n = cumulative_n
        prev_result = result

    total_n = 0 if prev_result is None else prev_result.shape[0]
    max_d = max(distances) if distances else None
    if max_d is not None:
        counts[f">{max_d}"] = total_n - prev_cumulative_n

    counts["n_sequences"] = total_n

    return counts


def plot_nn_counts_across_datasets(analysis_config: AnalysisConfig, plotting_dfs: dict, output_dir: str,
                                   innovation: bool=False) -> None:
    """
    Plot number of sequences with distance 1, 2, 3, and >3 to the nearest training sequence for each model and test.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        plotting_dfs (dict): Dict of dataFrames containing counts of sequences at each distance to the training set.
        output_dir (str): Directory to save the plots.
        innovation (bool): Whether the plot is for the innovation analysis (True) or the full generated sequences (False).
        Default is False. Used for the plot title.
    Returns:
        None
    """
    if innovation:
        innovation_title_part = "innovative "
        plotting_dfs = {model: df for model, df in plotting_dfs.items() if model != "test"}
    else:
        innovation_title_part = ""

    collection_specification = get_collection_specification_for_title(analysis_config.receptor_type)
    fig = plot_single_dataset(plotting_dfs, title=f'Number of {innovation_title_part}model sequences by distance to nearest <br>train sequence for {collection_specification} Repertoires',
                              xtitle='Distance to nearest training sequence', ytitle='Avg. sequence count',
                              distance_cols=['1', '2', '3', '>3'])
    png_path = f"{output_dir}/distances_plot.png"
    fig.write_image(png_path)
    print(f"Plot saved at: {png_path}")

    dataset_base_names = set()
    for key, df in plotting_dfs.items():
        if key != "test":
            dataset_base_names.update(df.index.str.rsplit('_', n=1).str[0])

    for dataset in sorted(dataset_base_names):
        subset_mask = {model: df.index.str.startswith(dataset)
                       for model, df in plotting_dfs.items()}
        fig = plot_single_dataset(plotting_dfs,
                                  title=f'Number of {innovation_title_part}model sequences by distance to nearest <br>train sequence for {collection_specification} Repertoires <br>({dataset})',
                                  xtitle='Distance to nearest training sequence', ytitle='Avg. sequence count',
                                  distance_cols=['1', '2', '3', '>3'],
                                  subset_mask=subset_mask)
        png_path = f"{output_dir}/distances_{dataset}_plot.png"
        fig.write_image(png_path)
        print(f"Plot saved at: {png_path}")


def plot_single_dataset(plotting_dfs: dict, title, xtitle: str, ytitle: str, distance_cols: list, subset_mask=None) -> go.Figure:
    """
    Creates a Plotly figure comparing either:
        1) the average sequence counts at each distance to the training set for each model and test.
        2) the average number of clusters of innovative sequences at each distance threshold for each model and dataset.
    If subset_mask is provided, it filters the data for each model according to the mask before computing the means and stds for the plot.
    Args:
        plotting_dfs (dict): Dict of dataFrames containing counts of sequences at each distance to the training set for each model and test.
        title (str): Title for the plot.
        xtitle (str): Title for the x-axis.
        ytitle (str): Title for the y-axis.
        distance_cols (list): List of column names in the DataFrames that correspond to the distance categories (e.g., ['1', '2', '3', '>3']).
        subset_mask (dict, optional): Dict with model names as keys and boolean masks as values to filter the data for each model before plotting. Defaults to None (no filtering).
    Returns:
        go.Figure: Plotly figure object ready for display or saving.
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Dark24
    model_names_sorted = sorted(plotting_dfs.keys())
    if "test" in model_names_sorted:
        model_names_sorted = [m for m in model_names_sorted if m != "test"] + ["test"]
    color_map = {model: colors[i % len(colors)] for i, model in enumerate(model_names_sorted)}

    for model in model_names_sorted:
        df = plotting_dfs[model]
        subset_df = df[subset_mask[model]] if subset_mask else df
        means = subset_df[distance_cols].mean()
        stds = subset_df[distance_cols].std()

        fig.add_trace(go.Scatter(
            x=distance_cols,
            y=means,
            # error_y=dict(type='data', array=stds, visible=True),
            mode='lines+markers',
            name=model,
            line=dict(color=color_map[model]),
            marker=dict(color=color_map[model], size=8),
            opacity=0.6,
        ))

    fig.data = sorted(fig.data, key=lambda trace: (0 if trace.name == "test" else 1, trace.name))
    fig.update_layout(
        width=700,
        height=500,
        title={"text": title,
                  "font": {"size": 20},
               'y': 0.93,
               'yanchor': 'top'
               },
        margin=dict(t=100),
        xaxis_title={'text': xtitle, 'font': {'size': 18}},
        yaxis_title={'text': ytitle, 'font': {'size': 18}},
        xaxis=dict(tickfont=dict(size=14)),
        yaxis=dict(tickfont=dict(size=14)),
        legend=dict(font=dict(size=14)),
        template='plotly_white'
    )
    return fig


def cluster_innovation_sequences(analysis_config: AnalysisConfig, innovation_sequences_dir: str) -> dict:
    """
    Clusters the innovative sequences by distance using CompAIRR clustering.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        innovation_sequences_dir (str): Directory where the innovative sequences are saved for CompAIRR analysis.
    Returns:
        n_clusters_by_model (dict): Dict containing the number of clusters for each model and dataset at each distance threshold.
    """
    clustering_dir = f"{analysis_config.analysis_output_dir}/clustering"
    os.makedirs(clustering_dir, exist_ok=True)

    n_clusters_by_model = {}
    for model in analysis_config.model_names:

        n_clusters_by_dataset = {}
        for innovation_gen_file_split in os.listdir(f"{innovation_sequences_dir}/{model}"):
            dataset_split_name = os.path.splitext(innovation_gen_file_split)[0]
            gen_file = f"{innovation_sequences_dir}/{model}/{innovation_gen_file_split}"
            compairr_output_dir = f"{clustering_dir}/compairr_output/{model}"
            os.makedirs(compairr_output_dir, exist_ok=True)

            n_clusters_by_distance = {}
            for dist in [1, 2, 3]:
                results_file = run_compairr_cluster(compairr_output_dir=compairr_output_dir,
                                                    sequences_path=gen_file,
                                                    output_file_identifier=f"{dataset_split_name}_d{dist}",
                                                    distance=dist)
                n_clusters = pd.read_csv(results_file, sep='\t')["#cluster_no"].nunique()
                n_clusters_by_distance[dist] = n_clusters

            n_clusters_by_dataset[dataset_split_name] = n_clusters_by_distance
        n_clusters_by_model[model] = n_clusters_by_dataset

    return n_clusters_by_model


def plot_cluster_counts(analysis_config: AnalysisConfig, num_clusters_by_model: dict) -> None:
    """
    Plots the number of clusters of innovative sequences at each distance threshold for each model and dataset.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        num_clusters_by_model (dict): Dict containing the number of clusters for each model and dataset at each distance threshold.
    Returns:
        None
    """
    clustering_dir = f"{analysis_config.analysis_output_dir}/clustering"

    cluster_dfs = {
        model: pd.DataFrame.from_dict(dataset_dict, orient='index').rename(columns=str)
        for model, dataset_dict in num_clusters_by_model.items()
    }

    collection_specification = get_collection_specification_for_title(analysis_config.receptor_type)
    fig = plot_single_dataset(
        cluster_dfs,
        title=f"Average number of clusters by distance threshold <br>for {collection_specification} Repertoires",
        xtitle='Distance',
        ytitle='Avg. number of clusters',
        distance_cols=['1', '2', '3']
    )
    png_path = f"{clustering_dir}/cluster_counts_plot.png"
    fig.write_image(png_path)
    print(f"Plot saved at: {png_path}")

    dataset_base_names = set()
    for model, df in cluster_dfs.items():
        dataset_base_names.update(df.index.str.rsplit('_', n=1).str[0])

    for dataset in sorted(dataset_base_names):
        subset_mask = {model: df.index.str.startswith(dataset)
                       for model, df in cluster_dfs.items()}
        fig = plot_single_dataset(
            cluster_dfs,
            title=f'Average number of clusters by distance threshold <br>for {collection_specification} Repertoires ({dataset})',
            xtitle='Distance',
            ytitle='Avg. number of clusters',
            distance_cols=['1', '2', '3'],
            subset_mask=subset_mask
        )
        png_path = f"{clustering_dir}/cluster_counts_{dataset}_plot.png"
        fig.write_image(png_path)
        print(f"Plot saved at: {png_path}")
