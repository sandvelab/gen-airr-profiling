import os
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial import distance

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_sequence_files, get_reference_files


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
    """ Collects clone frequency data and generates plots.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        output_dir (str): Directory to save the generated plots.
    Returns:
        None
    """
    for model_name in analysis_config.model_names:
        for reference in analysis_config.reference_data:
            print(f"Collecting clone frequency data for model: {model_name} and reference: {reference}")
            frequencies_dfs = collect_clone_frequencies_models(analysis_config, model_name, reference)

            model_output_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            plot_frequencies(frequencies_dfs, model_output_dir, reference, model_name)

    ref_frequencies = collect_clone_frequencies_reference(analysis_config)
    ref_output_dir = os.path.join(output_dir, "reference_comparison")
    os.makedirs(ref_output_dir, exist_ok=True)
    plot_frequencies(ref_frequencies, ref_output_dir, "train", "test")


def collect_clone_frequencies_models(analysis_config: AnalysisConfig, model_name: str, reference: str) -> dict:
    """ Collects clone frequency data from the generated and reference sequences per model.
    Returns:
        dict: A dictionary where keys are dataset split names and values are DataFrames containing clone frequencies.
    """
    model_comparison_files_dir = get_sequence_files(analysis_config, model_name, reference)
    dfs = {}
    for ref_file, gen_files in model_comparison_files_dir.items():
        ref_seqs = get_sequences_from_file(ref_file)
        for gen_file_split in gen_files:
            data_split_name = os.path.splitext(os.path.basename(gen_file_split))[0]
            gen_seqs = get_sequences_from_file(gen_file_split)

            df = get_frequencies_df(ref_seqs, gen_seqs, reference, model_name)
            dfs[data_split_name] = df

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
        dfs[f"reference_{os.path.basename(train_file).split('.')[0]}"] = df

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


def pseudo_log_transform(x, threshold=1e-3):
    return np.sign(x) * np.log1p(np.abs(x / threshold))


def plot_frequencies(frequencies: dict, output_dir: str, name1: str, name2: str) -> None:
    """ Plots the clone frequencies for the generated and reference sequences.
    Args:
        frequencies (list): A list of DataFrames, each containing the clone frequencies for a specific dataset.
        output_dir (str): Directory to save the generated plots.
        name1 (str): The reference dataset label.
        name2 (str): The generative model label.
    Returns:
        None
    """
    for dataset_name, freq_df in frequencies.items():
        freq_df[f"pseudo_freq_{name2}"] = pseudo_log_transform(freq_df[f"freq_{name2}"])
        freq_df[f"pseudo_freq_{name1}"] = pseudo_log_transform(freq_df[f"freq_{name1}"])

        jsd = distance.jensenshannon(freq_df[f"freq_{name1}"], freq_df[f"freq_{name2}"])

        fig = px.scatter(
            freq_df,
            x=f"pseudo_freq_{name2}",
            y=f"pseudo_freq_{name1}",
            hover_name=freq_df.index,
            labels={
                f"pseudo_freq_{name2}": f"{name2.upper()} frequency",
                f"pseudo_freq_{name1}": f"{name1.upper()} frequency"
            }
        )

        title_text = f"Clone Frequencies: {name2.upper()} vs {name1.upper()} ({dataset_name}), JSD={jsd:.4f}"
        fig.update_layout(
            template="simple_white",
            width=600,
            height=600,
            title={'text': wrap_title(title_text),
                      'font': {'size': 20}},
            xaxis_title=f"{name1} frequency (pseudo-log scale)",
            yaxis_title=f"{name2} frequency (pseudo-log scale)",
        )

        png_path = os.path.join(output_dir, f"{dataset_name}_{name2}_{name1}.png")
        fig.write_image(png_path)
        print(f"Plot saved as PNG at: {png_path}.")

def wrap_title(text, width=60):
    return "<br>".join(textwrap.wrap(text, width=width))

