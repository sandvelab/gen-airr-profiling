import os

import numpy as np
import pandas as pd
#TODO: We need to find more elegant solution for setting the backend
import matplotlib
matplotlib.use('Agg')
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import linkage, leaves_list

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.compairr_utils import deduplicate_and_merge_two_datasets, run_compairr_existence


def run_phenotype_analysis(analysis_config: AnalysisConfig):
    """ Run phenotype clustering analysis on the generated sequences of two different phenotypes.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        None
    """
    print(f"Running phenotype analysis for {analysis_config}")
    if len(analysis_config.model_names) != 1:
        raise ValueError("Phenotype analysis only supports one model")
    model_name = analysis_config.model_names[0]
    compairr_sequences_dir = f"{analysis_config.root_output_dir}/generated_compairr_sequences/{model_name}"

    similarities_matrix, dataset_names = calculate_similarities_matrix(analysis_config, compairr_sequences_dir,)

    similarities_df = pd.DataFrame(similarities_matrix, index=dataset_names, columns=dataset_names)
    plot_cluster_heatmap(analysis_config, similarities_df, model_name)


# TODO: There's a lot of file operations here, we should consider refactoring this function a bit more
def calculate_similarities_matrix(analysis_config, sequences_dir):
    """ Calculate the Jaccard similarities matrix between all pairs of datasets in the given directory.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        sequences_dir (str): Directory containing the generated sequence datasets.
    Returns:
        tuple: A tuple containing the similarities matrix (list of lists) and the list of dataset names.
    """
    output_dir = analysis_config.analysis_output_dir
    os.makedirs(output_dir, exist_ok=True)

    generated_datasets = sorted([f for f in os.listdir(sequences_dir) if f.endswith('.tsv')])
    generated_datasets_names = [dataset.removesuffix('.tsv') for dataset in generated_datasets]
    if len(generated_datasets) < 2:
        raise ValueError(f"At least two datasets are required for phenotype analysis, but found "
                         f"{len(generated_datasets)} in {sequences_dir}.")

    n = len(generated_datasets)
    similarities_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    helper_dir = f"{output_dir}/compairr_helper_files"
    os.makedirs(helper_dir, exist_ok=True)
    compairr_output_dir = f"{output_dir}/compairr_output"
    os.makedirs(compairr_output_dir, exist_ok=True)

    for i in range(n):
        dataset1 = generated_datasets[i]
        dataset1_path = f"{sequences_dir}/{dataset1}"
        dataset1_name = dataset1.removesuffix('.tsv')

        for j in range(i, n):
            dataset2 = generated_datasets[j]
            dataset2_path = f"{sequences_dir}/{dataset2}"
            dataset2_name = dataset2.removesuffix('.tsv')

            file_name = f"{dataset1_name}_{dataset2_name}"
            unique_sequences_path = f"{helper_dir}/{file_name}_unique.tsv"
            concat_sequences_path = f"{helper_dir}/{file_name}_concat.tsv"

            if i == j:
                jaccard_similarity = 1.0
            else:
                if os.path.exists(unique_sequences_path) and os.path.exists(concat_sequences_path):
                    print(f"Compairr helper files already exist for {file_name}. Skipping execution.")
                else:
                    deduplicate_and_merge_two_datasets(dataset1_path, dataset2_path, unique_sequences_path, concat_sequences_path)

                run_compairr_existence(compairr_output_dir, unique_sequences_path, concat_sequences_path, file_name,
                                       allowed_mismatches=analysis_config.allowed_mismatches, indels=analysis_config.indels)

                overlap_df = pd.read_csv(f"{compairr_output_dir}/{file_name}_overlap.tsv", sep='\t')
                n_nonzero_rows = overlap_df[(overlap_df['dataset_1'] != 0) & (overlap_df['dataset_2'] != 0)].shape[0]

                union = pd.read_csv(unique_sequences_path, sep='\t').shape[0]
                jaccard_similarity = n_nonzero_rows / union if union > 0 else 0.0

            similarities_matrix[i][j] = jaccard_similarity
            similarities_matrix[j][i] = jaccard_similarity

    return similarities_matrix, generated_datasets_names


def plot_cluster_heatmap(analysis_config: AnalysisConfig, similarities_matrix, model_name):
    """ Plot a clustered heatmap of the similarities matrix using seaborn.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        similarities_matrix (pd.DataFrame): DataFrame containing the similarities matrix.
        model_name (str): Name of the model for labeling purposes.
    Returns:
        None
    """
    output_dir = analysis_config.analysis_output_dir
    tsv_path = f"{output_dir}/similarities_matrix.tsv"
    if not os.path.exists(tsv_path):
        similarities_matrix.to_csv(tsv_path, sep='\t')

    Z = linkage(similarities_matrix.values, method="average", metric="euclidean")
    leaf_order = leaves_list(Z)

    clustered = similarities_matrix.iloc[leaf_order, :].iloc[:, leaf_order]
    annotation_text = np.round(clustered.values, 3).astype(str)
    clustered.index = [name.rsplit('_', 1)[0] for name in clustered.index]
    clustered.columns = [name.rsplit('_', 1)[0] for name in clustered.columns]

    # blues = px.colors.sequential.Blues_r
    # blues_cut = blues[2:-2]
    thermal = px.colors.sequential.thermal_r[:-2]

    # Build thermal colors but compressed into 0–0.1
    custom_colorscale = []
    for i, c in enumerate(thermal):
        position = i / (len(thermal) - 1)
        custom_colorscale.append((position * 0.1, c))

    # Add fixed color at value 1.0
    custom_colorscale.append((1.0, "white"))  # or any other color

    fig = go.Figure(
        data=go.Heatmap(
            z=clustered.values,
            x=clustered.columns,
            y=clustered.index,
            colorscale=custom_colorscale,
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Similarity"),
            text=annotation_text,
            texttemplate="%{text}",
            textfont=dict(color="black", size=17),
        )
    )

    fig.update_layout(
        title={"text": f"Pairwise Jaccard Similarity Between {model_name.upper()} {analysis_config.receptor_type} "
              f"Sets (Hamming Distance = {analysis_config.allowed_mismatches})",
               "font": {'size': 20}},
        width=1000,
        height=900,
        xaxis=dict(tickangle=45),
        template="plotly_white",
        coloraxis_colorbar=dict(
            tickvals=[0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 1.0],
            ticktext=["0", "0.01", "0.02", "0.03", "0.05", "0.07", "0.1", "1.0"],
            lenmode="fraction",
            len=1.0
        )
    )

    png_path = f"{output_dir}/cluster_heatmap.png"
    fig.write_image(png_path)
    print(f"Saved clustered heatmap: {png_path}")
