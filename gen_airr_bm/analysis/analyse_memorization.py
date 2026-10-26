import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.compairr_utils import run_compairr_existence, run_sequence_deduplication
from gen_airr_bm.utils.file_utils import get_sequence_files, get_reference_files
from gen_airr_bm.utils.plotting_utils import (get_collection_specification_for_title, get_repertoire_dot_traces,
                                             plot_repertoire_statistics)
from gen_airr_bm.utils.statistics_utils import (aggregate_repertoire_scores, check_one_repertoire_per_donor,
                                               describe_error_bars, run_repertoire_statistics,
                                               summarise_repertoire_scores)

SUBSET_INDEX_PATTERN = re.compile(r"_(\d+)\.tsv$")


def run_memorization_analysis(analysis_config: AnalysisConfig) -> None:
    """ Runs the memorization analysis. The memorization score of every generated subset is averaged within its
    repertoire before the models are summarised over repertoires (see statistics_utils). The models are also compared
    at the repertoire level: with each other, in the contrasts given in the config and, for UMI data, with the overlap
    between the train and test sequences. These comparisons are descriptive, or include confidence intervals and tests
    if statistical_tests is set, which requires a donor_pattern and one repertoire per donor.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        None
    """
    print("Running memorization analysis...")

    output_dir = analysis_config.analysis_output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_reference = "train" if "train" in analysis_config.reference_data else None
    test_reference = "test" if "test" in analysis_config.reference_data else None
    if train_reference is None:
        raise ValueError("Train data must be included in reference_data for memorization analysis.")

    scores_df = get_model_memorization_scores(analysis_config, output_dir, train_reference)
    repertoire_scores = aggregate_repertoire_scores(scores_df, "memorization_score", analysis_config.donor_pattern,
                                                    repertoire_column="file")
    if analysis_config.statistical_tests:
        check_one_repertoire_per_donor(repertoire_scores, analysis_config.donor_pattern)

    # Train and test only share sequences in UMI data; the other collections contain unique clones split into train
    # and test, so their overlap is not a meaningful reference
    if test_reference is not None and "UMI" in analysis_config.receptor_type:
        reference_scores = get_reference_memorization_scores(analysis_config, output_dir)
    else:
        reference_scores = None

    plot_results(repertoire_scores, reference_scores, output_dir, "memorization", analysis_config.receptor_type,
                 analysis_config.collection)
    run_memorization_statistics(analysis_config, repertoire_scores, reference_scores)


def get_model_memorization_scores(analysis_config: AnalysisConfig, output_dir: str,
                                  train_reference: str) -> pd.DataFrame:
    """ Get memorization scores (sequence overlap with the train sequences) of every generated subset of each model.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        output_dir (str): Directory to save intermediate and final results.
        train_reference (str): Reference data split to use for training (e.g., "train").
    Returns:
        pd.DataFrame: Columns model, file (the repertoire), subset, memorization_score and novel_sequences, one row
        per model and generated subset.
    """
    model_memorization_scores = []
    for model_name in analysis_config.model_names:
        comparison_files_dir = get_sequence_files(analysis_config, model_name, train_reference)
        for ref_file, gen_files in comparison_files_dir.items():
            memorization_scores, memorization_n = get_memorization_scores(analysis_config, ref_file, gen_files,
                                                                          output_dir, model_name)
            file_name = os.path.splitext(os.path.basename(ref_file))[0]
            model_memorization_scores.extend(
                (model_name, file_name, get_subset_index(gen_file), score, n)
                for gen_file, score, n in zip(gen_files, memorization_scores, memorization_n)
            )

    model_memorization_scores_df = pd.DataFrame(model_memorization_scores,
                                                columns=["model", "file", "subset", "memorization_score",
                                                         "novel_sequences"])
    model_memorization_scores_df.to_csv(f"{output_dir}/memorization_scores.tsv", sep="\t", index=False, header=True)

    return model_memorization_scores_df


def get_subset_index(gen_file: str) -> int | float:
    """ Returns the index of a generated subset from its file name, e.g. 3 for rep_1_3.tsv, or nan if there is none. """
    match = SUBSET_INDEX_PATTERN.search(os.path.basename(gen_file))
    return int(match.group(1)) if match else np.nan


def get_reference_memorization_scores(analysis_config: AnalysisConfig, output_dir: str) -> pd.DataFrame:
    """ Get the memorization score (sequence overlap) between the train and test sequences of each repertoire.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        output_dir (str): Directory to save intermediate and final results.
    Returns:
        pd.DataFrame: Columns repertoire and score, one row per repertoire.
    """
    rows = []
    for train_file, test_file in sorted(get_reference_files(analysis_config)):
        ref_score, _ = get_memorization_scores(analysis_config, train_file, [test_file], output_dir, "reference")
        rows.append({"repertoire": os.path.splitext(os.path.basename(train_file))[0], "score": ref_score[0]})

    return pd.DataFrame(rows, columns=["repertoire", "score"])


def get_memorization_scores(analysis_config: AnalysisConfig, train_file: str, test_or_gen_files: list[str], output_dir: str, name: str) -> tuple:
    """ Compute memorization scores (sequence overlap) between a train reference file and generated files or between
    train and the corresponding test set.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        train_file (str): Path to the train file.
        test_or_gen_files (list[str]): List of paths to generated files or list of one test file.
        output_dir (str): Directory to save intermediate and final results.
        name (str): Name of the model being evaluated or "reference".
    Returns:
        list: A list of memorization scores (sequence overlap) for each model.
    """
    memorization_scores = []
    memorization_counts = []
    compairr_output_dir = f"{output_dir}/compairr_output"
    for file in test_or_gen_files:
        memorization_ratio, memorization_n = compute_overlap_score(analysis_config, train_file, file, compairr_output_dir, name)
        memorization_scores.append(memorization_ratio)
        memorization_counts.append(memorization_n)

    return memorization_scores, memorization_counts


def compute_overlap_score(analysis_config: AnalysisConfig, train_file: str, test_or_gen_file: str, compairr_output_dir: str, name: str) -> tuple:
    """ Compute overlap score between two datasets using CompAIRR.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        train_file (str): Path to the reference train set.
        test_or_gen_file (str): Path to the model-generated set or to corresponding test set.
        compairr_output_dir (str): Directory to save CompAIRR output.
        name (str): Name of the model being evaluated or "reference".
    Returns:
        float: Overlap score between the two datasets.
    """
    dataset_name = os.path.splitext(os.path.basename(test_or_gen_file))[0]
    file_identifier = f"{dataset_name}_{name}"

    run_compairr_existence(compairr_output_dir, test_or_gen_file, train_file, file_identifier,
                           allowed_mismatches=analysis_config.allowed_mismatches, indels=analysis_config.indels)
    compairr_result = pd.read_csv(f"{compairr_output_dir}/{file_identifier}_overlap.tsv", sep='\t',
                                  names=['sequence_id', 'overlap_count'], header=0)
    n_nonzero_rows = compairr_result[(compairr_result['overlap_count'] != 0)].shape[0]
    ratio = n_nonzero_rows / len(compairr_result)
    novel_sequences = len(compairr_result) - n_nonzero_rows

    return ratio, novel_sequences


def plot_results(repertoire_scores: pd.DataFrame, reference_scores: pd.DataFrame | None, fig_dir: str, file_name: str,
                 receptor_type: str, collection: str = None) -> None:
    """ Plot the mean memorization score of each model over repertoires, with SD error bars, one dot per repertoire
    and the train vs. test reference line.
    Args:
        repertoire_scores (pd.DataFrame): Scores per model and repertoire as returned by aggregate_repertoire_scores.
        reference_scores (pd.DataFrame | None): Train vs. test scores per repertoire, with columns repertoire and score.
        fig_dir (str): Directory to save the plot.
        file_name (str): Name of the output plot and tsv files (without extension).
        receptor_type (str): Type of receptor being analyzed ("BCR", "TCR").
        collection (str): Optional collection label for the title (e.g. "D"), inferred from receptor_type if None.
    Returns:
        None
    """
    os.makedirs(fig_dir, exist_ok=True)
    png_path = os.path.join(fig_dir, file_name) + ".png"
    tsv_path = os.path.join(fig_dir, file_name) + ".tsv"

    summary = summarise_repertoire_scores(repertoire_scores, statistical_tests=False)
    summary = summary.sort_values("mean", ascending=False, kind="stable")
    models, scores, errors = summary["source"].tolist(), summary["mean"].tolist(), summary["sd"].tolist()
    mean_reference_score = float(reference_scores["score"].mean()) if reference_scores is not None else None

    pd.DataFrame({"model": models, "mean_overlap_score": scores, "std": errors}).to_csv(tsv_path, sep="\t",
                                                                                         index=False)
    with open(os.path.join(fig_dir, file_name) + "_mean_ref.tsv", "w") as f:
        f.write(f"{mean_reference_score}\n")

    dots = {"train": {model: list(zip(model_scores["donor"], model_scores["score"]))
                      for model, model_scores in repertoire_scores.dropna(subset=["score"]).groupby("source")}}
    dot_traces = get_repertoire_dot_traces(dots, ["train"], models)
    fig = go.Figure([go.Bar(
        x=models,
        y=scores,
        error_y=dict(type='data', array=errors, visible=True),
        marker=dict(color=px.colors.qualitative.Safe[0]),
        offsetgroup="train",
        showlegend=False,
    )] + dot_traces)

    collection_specification = get_collection_specification_for_title(receptor_type, collection)
    fig.update_layout(
        title=dict(
            text=f"Mean Memorization Score for Generated {collection_specification}<br>Repertoires",
            font=dict(size=20)
        ),
        xaxis=dict(
            title=dict(text="Model", font=dict(size=20)),
            tickangle=-45,
            tickfont=dict(size=18)
        ),
        yaxis=dict(
            title=dict(text="Mean Memorization Ratio", font=dict(size=20)),
            tickfont=dict(size=18)
        ),
        xaxis_tickangle=-45,
        template="plotly_white",
        colorway=px.colors.qualitative.Safe,
        legend=dict(tracegroupgap=0),
        # Room for the two-line title and the error bar caption above the plot
        margin=dict(t=150),
        # The figure grows with the donor legend, which is only shown if donors have several repertoires
        height=max(500, 300 + 30 * sum(trace.showlegend is not False for trace in dot_traces)),
    )
    fig.add_annotation(text=describe_error_bars(summary["n_repertoires"].max(), summary["n_donors"].max()),
                       xref="paper", yref="paper", x=1, y=1.02, showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(size=14))

    if mean_reference_score is not None:
        fig.add_hline(
            y=mean_reference_score,
            line=dict(color="black", dash="dash"),
            annotation_text=f"Train vs. Test = {mean_reference_score:.3f}",
            annotation_position="top right",
            annotation_font=dict(size=18, color="black")
        )

    fig.write_image(png_path, scale=3)

    print(f"Plot saved as png at: {png_path}")


def run_memorization_statistics(analysis_config: AnalysisConfig, repertoire_scores: pd.DataFrame,
                                reference_scores: pd.DataFrame | None) -> None:
    """ Runs the repertoire-level comparisons of the memorization scores and plots the comparisons with the train vs.
    test reference (UMI data only) and the contrasts from the config.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        repertoire_scores (pd.DataFrame): Scores per model and repertoire as returned by aggregate_repertoire_scores.
        reference_scores (pd.DataFrame | None): Train vs. test scores per repertoire, with columns repertoire and score.
    Returns:
        None
    """
    output_prefix = f"{analysis_config.analysis_output_dir}/statistics/memorization"
    results = run_repertoire_statistics(repertoire_scores, analysis_config.model_names, output_prefix,
                                        analysis_config.statistical_tests, reference_scores, analysis_config.contrasts,
                                        analysis_config.donor_pattern)
    plot_repertoire_statistics(results, output_prefix, "memorization",
                               get_collection_specification_for_title(analysis_config.receptor_type,
                                                                      analysis_config.collection),
                               score_description="Memorization of Train Sequences",
                               x_label="Difference in Memorization Ratio")
