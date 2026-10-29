import os
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial import distance
from scipy.stats import spearmanr, pearsonr

from gen_airr_bm.analysis.analyse_innovation_umi import symlog_transform
from gen_airr_bm.constants.dataset_split import DatasetSplit
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_sequence_files, get_reference_files, split_subset_name
from gen_airr_bm.utils.plotting_utils import (get_collection_specification_for_title, plot_grouped_avg_scores,
                                             plot_repertoire_statistics)
from gen_airr_bm.utils.statistics_utils import (aggregate_repertoire_scores, check_one_repertoire_per_donor,
                                               run_repertoire_statistics, summarise_scores_by_reference)

METRIC_NAME = "clone_frequency"
SCORE_COLUMNS = ["metric", "model", "reference", "dataset", "subset", "jsd"]
REFERENCE_PREFIX = "reference_"


def run_clone_frequencies_analysis(analysis_config: AnalysisConfig) -> None:
    """ Runs clone frequency analysis on the generated and reference sequences.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        None
    """
    print("Running clone frequency analysis on UMI data")

    output_dir = analysis_config.analysis_output_dir
    os.makedirs(output_dir, exist_ok=True)

    collect_and_plot_clone_frequencies(analysis_config, output_dir)


def collect_and_plot_clone_frequencies(analysis_config: AnalysisConfig, output_dir: str) -> None:
    """ Collects clone frequency data and generates plots. The JSD of every generated subset is averaged within its
    repertoire before the models are summarised over repertoires (see statistics_utils). Against the test sequences,
    the models are also compared at the repertoire level: with the train vs. test reference, with each other, and in
    the contrasts given in the config. These comparisons are descriptive, or include confidence intervals and tests if
    statistical_tests is set, which requires a donor_pattern and one repertoire per donor.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        output_dir (str): Directory to save the generated plots.
    Returns:
        None
    """
    scores = []

    for model_name in analysis_config.model_names:
        for reference in analysis_config.reference_data:
            frequencies_dfs = collect_clone_frequencies_models(analysis_config, model_name, reference)

            model_output_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            jsd_scores = plot_frequencies_by_dataset(frequencies_dfs, model_output_dir, reference, model_name)
            for subset_name, jsd in jsd_scores.items():
                repertoire, subset = split_subset_name(subset_name)
                scores.append({"metric": METRIC_NAME, "model": model_name, "reference": reference,
                               "dataset": repertoire, "subset": subset, "jsd": jsd})
            jsd_scores_avg = np.mean(list(jsd_scores.values()))

            plot_frequencies_combined(frequencies_dfs, model_output_dir, reference, model_name, jsd_scores_avg)

    ref_frequencies = collect_clone_frequencies_reference(analysis_config)
    ref_output_dir = os.path.join(output_dir, "reference_comparison")
    os.makedirs(ref_output_dir, exist_ok=True)
    ref_jsd_scores = plot_frequencies_by_dataset(ref_frequencies, ref_output_dir, "train", "test")
    reference_scores = pd.DataFrame([{"repertoire": name.removeprefix(REFERENCE_PREFIX), "score": jsd}
                                     for name, jsd in sorted(ref_jsd_scores.items())],
                                    columns=["repertoire", "score"])
    reference_score = float(reference_scores["score"].mean()) if len(reference_scores) else None
    plot_frequencies_combined(ref_frequencies, ref_output_dir, "train", "test", reference_score,
                              filter_combined_rep=False)

    scores_df = pd.DataFrame(scores, columns=SCORE_COLUMNS)
    scores_df.to_csv(f"{output_dir}/clone_frequency_jsd_scores.tsv", sep="\t", index=False)
    if scores_df.empty:
        return
    if analysis_config.statistical_tests:
        check_one_repertoire_per_donor(aggregate_repertoire_scores(scores_df, "jsd", analysis_config.donor_pattern),
                                       analysis_config.donor_pattern)

    summarize_and_plot_all(analysis_config, scores_df, reference_score)
    run_clone_frequency_statistics(analysis_config, scores_df, reference_scores)


def summarize_and_plot_all(analysis_config: AnalysisConfig, scores_df: pd.DataFrame,
                           reference_score: float | None) -> None:
    """ Average the JSDs over the generated subsets of each repertoire, summarise them over repertoires and plot the
    grouped averages, with one dot per repertoire.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores_df (pd.DataFrame): One row per model, reference, dataset and generated subset, with the JSD in jsd.
        reference_score (float | None): Mean train vs. test JSD over repertoires, plotted as a reference line.
    Returns:
        None
    """
    mean_scores, std_scores, repertoire_scores, error_bar_caption = \
        summarise_scores_by_reference(scores_df, METRIC_NAME, analysis_config.donor_pattern)

    plot_grouped_avg_scores(
        analysis_config=analysis_config,
        mean_scores_by_ref=mean_scores,
        std_scores_by_ref=std_scores,
        file_name="clone_frequency_jsd_comparison",
        distribution_type="clone frequency",
        scoring_method="JSD",
        reference_score=reference_score,
        error_bar_caption=error_bar_caption,
        repertoire_scores_by_ref=repertoire_scores
    )


def run_clone_frequency_statistics(analysis_config: AnalysisConfig, scores_df: pd.DataFrame,
                                   reference_scores: pd.DataFrame) -> None:
    """ Runs the repertoire-level comparisons against the test sequences and plots the comparisons with the train vs.
    test reference and the contrasts from the config. The train sequences are left out, since the models were fitted on
    them.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores_df (pd.DataFrame): One row per model, reference, dataset and generated subset, with the JSD in jsd.
        reference_scores (pd.DataFrame): Train vs. test scores per repertoire, with columns repertoire and score.
    Returns:
        None
    """
    if DatasetSplit.TEST.value not in analysis_config.reference_data:
        print("Skipping clone frequency statistics, since they are computed against the test sequences.")
        return

    test_scores = scores_df[scores_df["reference"] == DatasetSplit.TEST.value]
    output_prefix = f"{analysis_config.analysis_output_dir}/statistics/{METRIC_NAME}"
    repertoire_scores = aggregate_repertoire_scores(test_scores, "jsd", analysis_config.donor_pattern)
    results = run_repertoire_statistics(repertoire_scores, analysis_config.model_names, output_prefix,
                                        analysis_config.statistical_tests,
                                        reference_scores if len(reference_scores) else None,
                                        analysis_config.contrasts, analysis_config.donor_pattern)
    plot_repertoire_statistics(results, output_prefix, "clone frequency",
                               get_collection_specification_for_title(analysis_config.receptor_type,
                                                                      analysis_config.collection))


def collect_clone_frequencies_models(analysis_config: AnalysisConfig, model_name: str, reference: str) -> dict:
    """ Collects clone frequency data from the generated and reference sequences per model.
    Returns:
        dict: A dictionary where keys are dataset split names and values are DataFrames containing clone frequencies.
    """
    model_comparison_files_dir = get_sequence_files(analysis_config, model_name, reference)
    dfs = {}
    for ref_file, gen_files in model_comparison_files_dir.items():
        data_name = os.path.basename(ref_file).split('.')[0]
        ref_seqs = get_sequences_from_file(ref_file)

        all_gen_seqs = []
        for gen_file_split in gen_files:
            data_split_name = os.path.splitext(os.path.basename(gen_file_split))[0]
            gen_seqs = get_sequences_from_file(gen_file_split)

            all_gen_seqs.extend(gen_seqs)

            df = get_frequencies_df(ref_seqs, gen_seqs, reference, model_name)
            dfs[data_split_name] = df

        if all_gen_seqs:
            df_pooled = get_frequencies_df(ref_seqs, all_gen_seqs, reference, model_name)
            dfs[f"{data_name}_all"] = df_pooled

    return dfs


def collect_clone_frequencies_reference(analysis_config: AnalysisConfig) -> dict:
    """ Collects clone frequency data from the reference sequences for comparison.
    Returns:
        dict: A dictionary where keys are dataset split names and values are DataFrames containing clone frequencies
    """
    dfs = {}
    reference_comparison_files = get_reference_files(analysis_config)
    for train_file, test_file in reference_comparison_files:
        train_seqs = get_sequences_from_file(train_file)
        test_seqs = get_sequences_from_file(test_file)

        df = get_frequencies_df(train_seqs, test_seqs, "train", "test")
        dfs[f"{REFERENCE_PREFIX}{os.path.basename(train_file).split('.')[0]}"] = df

    return dfs


def get_sequences_from_file(file_path) -> list:
    """ Reads sequences from a file and returns them as a list. """
    data_df = pd.read_csv(file_path, sep='\t', usecols=["junction_aa"])
    return data_df["junction_aa"].tolist()


def get_frequencies_df(sample_1: list, sample_2: list, label_1: str, label_2: str) -> pd.DataFrame:
    """ Creates a DataFrame containing the clone frequencies for two samples.
    Args:
        sample_1 (list): List of sequences from the first sample.
        sample_2 (list): List of sequences from the second sample.
        label_1 (str): Label for the first sample (e.g., "reference").
        label_2 (str): Label for the second sample (e.g., "model").
    Returns:
        pd.DataFrame: A DataFrame with columns for counts and frequencies of each sequence in both samples.
    """
    counts_1 = Counter(sample_1)
    counts_2 = Counter(sample_2)

    df = pd.DataFrame({
        f"count_{label_1}": pd.Series(counts_1),
        f"count_{label_2}": pd.Series(counts_2),
    }).fillna(0).astype(int)

    df[f"freq_{label_1}"] = df[f"count_{label_1}"] / len(sample_1)
    df[f"freq_{label_2}"] = df[f"count_{label_2}"] / len(sample_2)

    return df


# def pseudo_log_transform(x, threshold=1e-3):
#     return np.sign(x) * np.log1p(np.abs(x / threshold))
def pseudo_log_transform(x, linthresh=1e-5, base=10):
    return symlog_transform(x)


def wrap_title(text, width=60):
    return "<br>".join(textwrap.wrap(text, width=width))


def create_scatter_plot(combined_df: pd.DataFrame, name1: str, name2: str, title_text: str,
                        color_by: str = None) -> px.scatter:
    """ Creates a scatter plot for clone frequencies.

    Args:
        combined_df (pd.DataFrame): DataFrame with pseudo_freq columns and optional dataset column.
        name1 (str): The reference dataset label.
        name2 (str): The generative model label.
        title_text (str): Title for the plot.
        color_by (str, optional): Column name to color by (e.g., 'dataset'). None for single color.
        width (int): Plot width in pixels.
        height (int): Plot height in pixels.
    Returns:
        plotly.graph_objs.Figure: The scatter plot figure.
    """
    hover_data = {'sequence': True}
    if color_by:
        hover_data[color_by] = True

    fig = px.scatter(
        combined_df,
        x=f"pseudo_freq_{name2}",
        y=f"pseudo_freq_{name1}",
        color=color_by,
        opacity=0.6,
        hover_name='sequence',
        hover_data=hover_data,
        labels={
            f"pseudo_freq_{name2}": f"{name2.upper()} frequency",
            f"pseudo_freq_{name1}": f"{name1.upper()} frequency",
            color_by: 'Repertoire' if color_by else None
        }
    )

    min_val = min(combined_df[f"pseudo_freq_{name2}"].min(), combined_df[f"pseudo_freq_{name1}"].min())
    max_val = max(combined_df[f"pseudo_freq_{name2}"].max(), combined_df[f"pseudo_freq_{name1}"].max())

    color_palette = px.colors.qualitative.Safe
    fig.update_layout(
        template="plotly_white",
        title={'text': wrap_title(title_text), 'font': {'size': 20}},
        xaxis_title={'text': f"{name2} Frequency (pseudo-log scale)", 'font': {'size': 20}},
        yaxis_title={'text': f"{name1.capitalize()} Frequency (pseudo-log scale)", 'font': {'size': 20}},
        xaxis=dict(tickfont=dict(size=18)),
        yaxis=dict(tickfont=dict(size=18)),
        legend_title="Repertoire" if color_by else None,
        legend=dict(
            title=dict(font=dict(size=18)),
            font=dict(size=18)
        ),
        colorway=color_palette,
    )

    fig.add_shape(
        type="line",
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color="red", dash="dash", width=2)
    )

    threshold = 1 / 450000
    tickvals = np.arange(0, max_val + 1)
    ticktext = ["0"] + [f"{threshold * 10 ** (i - 1):.0e}" for i in tickvals[1:]]

    fig.update_yaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext
    )

    return fig


def plot_frequencies_by_dataset(frequencies: dict, output_dir: str, name1: str, name2: str) -> dict:
    """ Plots the clone frequencies for the generated and reference sequences.
    Args:
        frequencies (dict): Dictionary of DataFrames containing clone frequencies.
        output_dir (str): Directory to save the generated plots.
        name1 (str): The reference dataset label.
        name2 (str): The generative model label.
    Returns:
        dict: The JSD score of every dataset (excluding _all), keyed by dataset name.
    """
    jsd_scores = {}

    for dataset_name, freq_df in frequencies.items():
        df_copy = freq_df.copy()
        df_copy[f"pseudo_freq_{name2}"] = pseudo_log_transform(df_copy[f"freq_{name2}"])
        df_copy[f"pseudo_freq_{name1}"] = pseudo_log_transform(df_copy[f"freq_{name1}"])
        df_copy['sequence'] = df_copy.index

        jsd = distance.jensenshannon(df_copy[f"freq_{name1}"], df_copy[f"freq_{name2}"], base=2) ** 2

        if '_all' not in dataset_name:
            jsd_scores[dataset_name] = jsd

        rho, p = spearmanr(df_copy[f"pseudo_freq_{name1}"], df_copy[f"pseudo_freq_{name2}"])
        title_text = f"Clone Frequencies: {name2.upper()} vs {name1.upper()} ({dataset_name}), ρ = {rho:.3f}"
        fig = create_scatter_plot(df_copy, name1, name2, title_text)
        png_path = os.path.join(output_dir, f"{dataset_name}_{name2}_{name1}_symlog.png")
        fig.write_image(png_path, scale=3)
        print(f"Plot saved as png at: {png_path}.")

    return jsd_scores


def plot_frequencies_combined(frequencies: dict, output_dir: str, name1: str, name2: str, jsd_avg: float,
                              filter_combined_rep: bool = True) -> None:
    """ Plots combined clone frequencies for multiple datasets on a single plot.

    Each dataset is shown with a different color.

    Args:
        frequencies (dict): Dictionary of DataFrames containing clone frequencies.
        output_dir (str): Directory to save the generated plots.
        name1 (str): The reference dataset label.
        name2 (str): The generative model label.
        jsd_avg (float): The mean JSD score to include in the plot title.
        filter_combined_rep (bool): If True, only include datasets with '_all' suffix (full repertoire). If False, include all datasets.
    Returns:
        None
    """
    if filter_combined_rep:
        selected_datasets = {k: v for k, v in frequencies.items() if '_all' in k}
    else:
        selected_datasets = frequencies

    if not selected_datasets:
        print(f"No datasets found for combined plot (filter_all={filter_combined_rep}).")
        return

    combined_data = []
    spearman_rhos = []
    pearson_rhos = []

    for dataset_name, freq_df in selected_datasets.items():
        clean_name = dataset_name.replace('_all', '').replace('reference_', '')

        df_copy = freq_df.copy()
        df_copy[f"pseudo_freq_{name2}"] = pseudo_log_transform(df_copy[f"freq_{name2}"])
        df_copy[f"pseudo_freq_{name1}"] = pseudo_log_transform(df_copy[f"freq_{name1}"])
        df_copy['repertoire'] = clean_name
        df_copy['sequence'] = df_copy.index
        combined_data.append(df_copy)

    frequencies_exclude_all = {k: v for k, v in frequencies.items() if '_all' not in k}
    for dataset_name, freq_df in frequencies_exclude_all.items():
        df_copy = freq_df.copy()
        df_copy[f"pseudo_freq_{name2}"] = pseudo_log_transform(df_copy[f"freq_{name2}"])
        df_copy[f"pseudo_freq_{name1}"] = pseudo_log_transform(df_copy[f"freq_{name1}"])

        rho, p = spearmanr(df_copy[f"pseudo_freq_{name1}"], df_copy[f"pseudo_freq_{name2}"])
        spearman_rhos.append(rho)
        rho, p = pearsonr(df_copy[f"pseudo_freq_{name1}"], df_copy[f"pseudo_freq_{name2}"])
        pearson_rhos.append(rho)

    combined_df = pd.concat(combined_data, ignore_index=False)
    spearman_rho_avg = np.mean(spearman_rhos)
    pearson_rhos_avg = np.mean(pearson_rhos)

    save_scatter_with_metric(combined_df, name1, name2, output_dir, os.path.join(output_dir, f"combined_repertoires_{name2}_{name1}_symlog_spearman.png"),
                             f"Clone Frequencies: {name2.upper()} vs {name1.upper()} (Spearman ρ = {spearman_rho_avg:.3f})")
    save_scatter_with_metric(combined_df, name1, name2, output_dir, os.path.join(output_dir, f"combined_repertoires_{name2}_{name1}_symlog_pearson.png"),
                             f"Clone Frequencies: {name2.upper()} vs {name1.upper()} (Pearson ρ = {pearson_rhos_avg:.3f})")
    save_scatter_with_metric(combined_df, name1, name2, output_dir,
                             os.path.join(output_dir, f"combined_repertoires_{name2}_{name1}_symlog_jsd.png"),
                             f"Clone Frequencies: {name2.upper()} vs {name1.upper()} (Mean JSD = {jsd_avg:.3f})")


def save_scatter_with_metric(combined_df, name1, name2, output_dir, png_path, title_text):
    fig = create_scatter_plot(combined_df, name1, name2, title_text, color_by='repertoire')
    fig.write_image(png_path, scale=3)
    print(f"Combined plot saved as png at: {png_path}")
