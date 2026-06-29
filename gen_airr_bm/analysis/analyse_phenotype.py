import os
import numpy as np
import pandas as pd
#TODO: We need to find more elegant solution for setting the backend
import matplotlib
from scipy.spatial.distance import squareform

from gen_airr_bm.utils.plotting_utils import get_collection_specification_for_title, wrap_title

matplotlib.use('Agg')
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import linkage, leaves_list

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.constants.dataset_split import DatasetSplit
from gen_airr_bm.utils.compairr_utils import deduplicate_and_merge_two_datasets, run_compairr_existence


def run_phenotype_analysis(analysis_config: AnalysisConfig):
    """ Run phenotype clustering analysis on the generated or train sequences of two different phenotypes.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        None
    """
    print(f"Running phenotype analysis for {analysis_config}")
    if len(analysis_config.model_names) != 1:
        raise ValueError("Phenotype analysis only supports one model")
    model_name = analysis_config.model_names[0]

    if model_name == DatasetSplit.TRAIN.value:
        compairr_sequences_dir = f"{analysis_config.root_output_dir}/{model_name}_compairr_sequences"
    else:
        if analysis_config.receptor_type == 'BCR UMI':
            sequences_dir_name = f"generated_compairr_sequences"
        else:
            sequences_dir_name = f"novel_generated_compairr_sequences"
        compairr_sequences_dir = f"{analysis_config.root_output_dir}/{sequences_dir_name}/{model_name}"

    similarities_matrix, dataset_names = calculate_similarities_matrix(analysis_config, compairr_sequences_dir,)

    similarities_df = pd.DataFrame(similarities_matrix, index=dataset_names, columns=dataset_names)

    phenotypes = [extract_phenotype(analysis_config, name) for name in dataset_names]
    subjects = [extract_subject(analysis_config, name) for name in dataset_names]

    map_phenotype = compute_map(similarities_df, phenotypes)
    map_subject = compute_map(similarities_df, subjects)
    save_ranking_analysis(similarities_df, phenotypes, subjects, analysis_config.analysis_output_dir)
    save_map_metrics(
        output_dir=analysis_config.analysis_output_dir,
        model_name=model_name,
        receptor_type=analysis_config.receptor_type,
        allowed_mismatches=analysis_config.allowed_mismatches,
        map_phenotype=map_phenotype,
        map_subject=map_subject,
        n_repertoires=len(dataset_names),
    )

    plot_cluster_heatmap(analysis_config, similarities_df, model_name, map_phenotype, map_subject)


def extract_phenotype(analysis_config: AnalysisConfig, name: str):
    """Extract phenotype label from a dataset filename.
    Args:
        analysis_config: AnalysisConfig object containing the receptor type information.
        name: dataset filename,
    Returns:
        str: phenotype label extracted from the filename.
    """
    receptor_type = analysis_config.receptor_type
    parts = name.split('_')
    if receptor_type == 'TCR':
        return parts[2]
    elif receptor_type == 'BCR':
        if parts[1].lower() in ['pln', 'iln']:
            return 'LN'
        return parts[1]
    elif receptor_type == 'BCR UMI':
        return parts[0]
    else:
        raise ValueError(f"Unknown receptor_type: {receptor_type}")


def extract_subject(analysis_config: AnalysisConfig, name: str):
    """Extract subject label from a dataset filename.
    Args:
        analysis_config: AnalysisConfig object containing the receptor type information.
        name: dataset filename,
    Returns:
        str: subject label extracted from the filename.
    """
    receptor_type = analysis_config.receptor_type
    parts = name.split('_')
    if receptor_type == 'TCR':
        return parts[0]
    elif receptor_type == 'BCR':
        return parts[0]
    elif receptor_type == 'BCR UMI':
        return parts[1]
    else:
        raise ValueError(f"Unknown receptor_type: {receptor_type}")


def compute_map(similarities_df, labels):
    """Compute Mean Average Precision over rankings induced by similarity.

    For each repertoire (row), rank other items by similarity (descending) and compute
    Average Precision over items sharing the repertoire's label. MAP is the mean across repertoires.

    Args:
        similarities_df: pd.DataFrame, square similarity matrix (higher = more similar)
        labels: list of labels, same order as rows/columns of similarities_df
    Returns:
        float: MAP score in [0, 1]
    """
    sim = similarities_df.values.copy().astype(float)
    n = len(labels)
    np.fill_diagonal(sim, -np.inf)  # exclude self from ranking

    aps = []
    for i in range(n):
        repertoire_label = labels[i]
        # number of relevant items (same label, excluding self)
        n_relevant = sum(1 for j in range(n) if j != i and labels[j] == repertoire_label)
        if n_relevant == 0:
            continue  # skip repertoires with no peers

        # rank all other items by similarity, descending
        ranking = np.argsort(-sim[i])

        hits = 0
        sum_precisions = 0.0
        for rank, idx in enumerate(ranking, start=1):
            if labels[idx] == repertoire_label:
                hits += 1
                sum_precisions += hits / rank
                if hits == n_relevant:
                    break  # all relevant items found

        aps.append(sum_precisions / n_relevant)

    return float(np.mean(aps)) if aps else 0.0


# TODO: There's a lot of file operations here, we should consider refactoring this function a bit more
def calculate_similarities_matrix(analysis_config, sequences_dir):
    """ Calculate the Jaccard similarities matrix between all pairs of datasets in the given directory.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        sequences_dir (str): Directory containing the sequence datasets.
    Returns:
        tuple: A tuple containing the similarities matrix (list of lists) and the list of dataset names.
    """
    output_dir = analysis_config.analysis_output_dir
    os.makedirs(output_dir, exist_ok=True)

    sequence_datasets = sorted([f for f in os.listdir(sequences_dir) if f.endswith('.tsv')])
    sequence_datasets_names = [dataset.removesuffix('.tsv') for dataset in sequence_datasets]
    if len(sequence_datasets) < 2:
        raise ValueError(f"At least two datasets are required for phenotype analysis, but found "
                         f"{len(sequence_datasets)} in {sequences_dir}.")

    n = len(sequence_datasets)
    similarities_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    helper_dir = f"{output_dir}/compairr_helper_files"
    os.makedirs(helper_dir, exist_ok=True)
    compairr_output_dir = f"{output_dir}/compairr_output"
    os.makedirs(compairr_output_dir, exist_ok=True)

    for i in range(n):
        dataset1 = sequence_datasets[i]
        dataset1_path = f"{sequences_dir}/{dataset1}"
        dataset1_name = dataset1.removesuffix('.tsv')

        for j in range(i, n):
            dataset2 = sequence_datasets[j]
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

    return similarities_matrix, sequence_datasets_names


def plot_cluster_heatmap(analysis_config: AnalysisConfig, similarities_matrix, model_name, map_phenotype, map_subject):
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

    distance_matrix = 1.0 - similarities_matrix.values
    np.fill_diagonal(distance_matrix, 0.0)
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method="average")
    leaf_order = leaves_list(Z)

    clustered = similarities_matrix.iloc[leaf_order, :].iloc[:, leaf_order]
    if analysis_config.receptor_type == 'BCR':
        annotation_text = np.round(clustered.values, 4).astype(str)
    else:
        annotation_text = np.round(clustered.values, 3).astype(str)
    clustered.index = [name.rsplit('_', 1)[0] for name in clustered.index]
    clustered.columns = [name.rsplit('_', 1)[0] for name in clustered.columns]

    z_values = clustered.values.copy()
    np.fill_diagonal(z_values, np.nan)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=clustered.columns,
            y=clustered.index,
            colorscale=px.colors.sequential.thermal_r[:-2],
            zmin=0.0,
            zmax=0.16,
            colorbar=dict(
                title="Jaccard similarity",
                tickvals=[0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16],
                ticktext=["0", "0.02", "0.04", "0.06", "0.08", "0.10", "0.12", "0.14", "0.16"],
                lenmode="fraction",
                len=1.0,
            ),
            text=annotation_text,
            texttemplate="%{text}",
            textfont=dict(color="black", size=17),
        )
    )

    collection_specification = get_collection_specification_for_title(analysis_config.receptor_type)
    title_text = wrap_title(f"Pairwise Jaccard Similarity Between {model_name.upper()} {collection_specification} "
                            f"Repertoires", width=55) + f"<br><sub>MAP phenotype = {map_phenotype:.3f} | MAP subject = {map_subject:.3f}</sub>"


    fig.update_layout(
        title={
            "text": title_text,
            "font": {"size": 30},
            'y': 0.95,
            'yanchor': 'top'
        },
        margin=dict(t=140),
        plot_bgcolor="white",
        width=1000,
        height=900,
        xaxis=dict(tickangle=45),
        font=dict(size=16),
        template="plotly_white",
    )

    png_path = f"{output_dir}/cluster_heatmap.png"
    fig.write_image(png_path)
    print(f"Saved clustered heatmap: {png_path}")


def save_ranking_analysis(similarities_df, phenotypes, subjects, output_dir):
    """For each repertoire, save a ranked list of all other repertoires
    with their similarity, rank, and label match info.

    Args:
        similarities_df: pd.DataFrame, square similarity matrix with repertoire names as index/columns
        phenotypes: list of phenotype labels corresponding to the rows/columns of similarities_df
        subjects: list of subject labels corresponding to the rows/columns of similarities_df
        output_dir: directory to save the rankings CSV

    Returns:
        pd.DataFrame: DataFrame containing the rankings and metadata for each repertoire.
    """
    sim = similarities_df.values
    names = list(similarities_df.index)
    n = len(names)

    rows = []
    for i in range(n):
        repertoire_name = names[i]
        repertoire_pheno = phenotypes[i]
        repertoire_subject = subjects[i]

        # Get similarities to all other repertoires (exclude self)
        sims_to_others = [(j, sim[i, j]) for j in range(n) if j != i]
        # Sort by similarity descending
        sims_to_others.sort(key=lambda x: x[1], reverse=True)

        for rank, (j, similarity) in enumerate(sims_to_others, start=1):
            rows.append({
                'repertoire': repertoire_name,
                'repertoire_phenotype': repertoire_pheno,
                'repertoire_subject': repertoire_subject,
                'rank': rank,
                'neighbor': names[j],
                'neighbor_phenotype': phenotypes[j],
                'neighbor_subject': subjects[j],
                'similarity': similarity,
                'same_phenotype': phenotypes[j] == repertoire_pheno,
                'same_subject': subjects[j] == repertoire_subject,
            })

    df = pd.DataFrame(rows)
    csv_path = f"{output_dir}/rankings.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved rankings: {csv_path}")
    return df


def save_map_metrics(output_dir, model_name, receptor_type, allowed_mismatches,
                     map_phenotype, map_subject, n_repertoires):
    """Save MAP scores to a TSV file.
    Args:
        output_dir: Directory to save the metrics file.
        model_name: Name of the model being analyzed.
        receptor_type: Type of receptor (e.g., TCR, BCR).
        allowed_mismatches: Number of mismatches allowed in similarity calculation.
        map_phenotype: MAP score for phenotype classification.
        map_subject: MAP score for subject classification.
        n_repertoires: Number of repertoires analyzed.

    Returns:
        None
    """
    metrics = pd.DataFrame({
        'model': [model_name],
        'receptor_type': [receptor_type],
        'allowed_mismatches': [allowed_mismatches],
        'n_repertoires': [n_repertoires],
        'map_phenotype': [map_phenotype],
        'map_subject': [map_subject],
    })

    metrics_path = f"{output_dir}/map_metrics.tsv"
    metrics.to_csv(metrics_path, sep='\t', index=False)
    print(f"Saved MAP metrics: {metrics_path}")
