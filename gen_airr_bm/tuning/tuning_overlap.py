import glob
import os
from pathlib import Path

import pandas as pd
import plotly.express as px

from gen_airr_bm.core.tuning_config import TuningConfig
from gen_airr_bm.utils.tuning_utils import validate_analyses_data


def run_overlap_tuning(tuning_config: TuningConfig):
    """ Runs parameter tuning by overlap with train and validation reference set.
        Args:
            tuning_config: Configuration for the tuning, including paths and model names.
        Returns:
            None
    """
    print("Tuning model hyperparameters based on overlap with train and validation reference set...")
    validated_analyses_paths = validate_analyses_data(tuning_config, required_analyses=['memorization',
                                                                                        'precision_recall'])
    print(f"Validated analyses for tuning: {validated_analyses_paths}")
    os.makedirs(tuning_config.tuning_output_dir, exist_ok=True)

    memorization_df, memorization_mean_ref_score, precision_df = get_overlap_results(tuning_config)
    overlap_difference_df = compute_overlap_difference(tuning_config, memorization_df, precision_df)

    plot_overlap_results(overlap_difference_df, tuning_config.tuning_output_dir)


def get_overlap_results(tuning_config: TuningConfig) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """ Collects results from memorization and precision analyses for tuning purposes.
    Args:
        tuning_config: Configuration for the tuning, including paths and model names.
    Returns:
        tuple: DataFrame of memorization results, mean reference memorization score, and DataFrame of precision results.

    """
    root_output_dir = tuning_config.root_output_dir
    model_names = tuning_config.model_names
    memorization_path = Path(root_output_dir) / "analyses/memorization" / '_'.join(model_names) / "memorization"
    memorization_df = pd.read_csv(memorization_path / ".tsv", sep="\t")

    with open(memorization_path / "_mean_ref.tsv", "r") as f:
        memorization_mean_ref_score = float(f.readline().strip())

    precision_path = glob.glob(str(root_output_dir) + "/analyses/precision_recall/" + '_'.join(model_names) +
                               "/precision/*.tsv")
    precision_df = pd.read_csv(precision_path[0], sep="\t")

    return memorization_df, memorization_mean_ref_score, precision_df


def compute_overlap_difference(tuning_config: TuningConfig, memorization_df, precision_df):
    """ Computes the difference between the overlap results from memorization and precision analyses.
    Args:
        tuning_config: Configuration for the tuning, including paths and model names.
        memorization_df: DataFrame containing results from memorization analysis.
        precision_df: DataFrame containing results from precision analysis.
    Returns:
        DataFrame: DataFrame containing the overlap difference.
    """
    precision_scores = precision_df[["Model", "Mean_Score"]].sort_values(by="Model")
    memorization_scores = memorization_df[["model", "mean_jaccard_similarity"]].sort_values(by="model")
    overlap_difference = precision_scores["Mean_Score"].values - memorization_scores["mean_jaccard_similarity"].values
    overlap_difference = pd.DataFrame({
        "Model": precision_scores["Model"].values,
        "Overlap_Difference": overlap_difference
    })
    overlap_difference_path = Path(tuning_config.tuning_output_dir) / "overlap_difference.tsv"
    overlap_difference.to_csv(overlap_difference_path, sep="\t", index=False)
    return overlap_difference


def plot_overlap_results(overlap_difference_df, output_dir):
    """ Plots the overlap difference between memorization and precision analyses.
    Args:
        overlap_difference_df: DataFrame containing the overlap difference.
        output_dir: Directory where the plot will be saved.
    Returns:
        None
    """
    best_model = overlap_difference_df.loc[overlap_difference_df["Overlap_Difference"].idxmax(), "Model"]
    best_score = overlap_difference_df["Overlap_Difference"].max()

    df = overlap_difference_df.sort_values(by="Model")

    fig = px.bar(
        df,
        x="Model",
        y="Overlap_Difference",
        title=f"Scores by Model (Highest: {best_model}, {best_score:.3f})",
        text="Overlap_Difference"
    )

    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        yaxis_title="Score",
        xaxis_title="Model",
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    plot_path = Path(output_dir) / "overlap_difference.png"
    fig.write_image(plot_path)
