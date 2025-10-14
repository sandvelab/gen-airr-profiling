import glob
import os
from pathlib import Path

import pandas as pd
import plotly.express as px

from gen_airr_bm.core.tuning_config import TuningConfig
from gen_airr_bm.utils.tuning_utils import validate_analyses_data


def run_reduced_dim_tuning(tuning_config: TuningConfig):
    """ Runs parameter tuning by similarity in reduced dimensionality.
        Args:
            tuning_config: Configuration for the tuning, including paths and model names.
        Returns:
            None
    """
    print("Tuning model hyperparameters based on reduced dimensionality metrics...")

    validated_analyses_paths = validate_analyses_data(tuning_config, required_analyses=['reduced_dimensionality'])
    print(f"Validated analyses for tuning: {validated_analyses_paths}")

    os.makedirs(tuning_config.tuning_output_dir, exist_ok=True)
    summary, best_model = collect_analyses_results(tuning_config)
    save_and_plot_summary(tuning_config, summary, best_model, tuning_config.tuning_output_dir)


def collect_analyses_results(tuning_config: TuningConfig) -> tuple[pd.DataFrame, str]:
    """ Collects results from analyses for tuning purposes. Amino acid results are averaged across all sequence lengths
    before combined with kmer and sequence length results.
        Args:
            tuning_config: Configuration for the tuning, including paths and model names.
        Returns:
            tuple: Summary dataframe of the analyses results and name of the best model based on validation reference.
    """
    root_output_dir = tuning_config.root_output_dir
    model_names = tuning_config.model_names
    reference_data = tuning_config.reference_data
    analyses_dir = (Path(root_output_dir) / "analyses/reduced_dimensionality" / '_'.join(model_names) /
                    '_'.join(reference_data))

    result_files = glob.glob(os.path.join(analyses_dir, "*.tsv"))
    amino_acid_dfs, other_dfs = [], []
    for f in result_files:
        df = pd.read_csv(f, sep="\t")
        df["Reference"] = df["Reference"].str.replace("test", "validation")
        if "aminoacid" in f.split("/")[-1]:
            amino_acid_dfs.append(df)
        else:
            other_dfs.append(df)

    aa_data = pd.concat(amino_acid_dfs, ignore_index=True)
    aa_summary = aa_data.groupby(["Reference", "Model"], as_index=False)[["Mean_Score", "Std_Dev"]].mean()
    dfs = other_dfs + [aa_summary]

    all_data = pd.concat(dfs, ignore_index=True)
    summary = all_data.groupby(["Reference", "Model"], as_index=False)[["Mean_Score", "Std_Dev"]].mean()

    val_summary = summary[summary["Reference"] == "validation"]
    best_model = val_summary.loc[val_summary["Mean_Score"].idxmin()]["Model"]

    return summary, best_model


def save_and_plot_summary(tuning_config: TuningConfig, summary: pd.DataFrame, best_model: str, output_dir: str):
    """ Saves the tuning summary to a TSV file and generates a bar plot.
    Args:
        tuning_config: Configuration for the tuning, including paths and model names.
        summary: DataFrame containing the summary of analyses results.
        best_model: Name of the best model based on validation reference.
        output_dir: Directory where the summary and plot will be saved.
    Returns:
        None
    """
    summary_path = Path(output_dir) / "reduced_dim_summary.tsv"
    summary.to_csv(summary_path, sep="\t")
    print(f"Saved tuning summary for tuning method {tuning_config.tuning_method} to {summary_path}")

    color_palette = px.colors.qualitative.Safe
    summary = summary.sort_values(by="Mean_Score", ascending=False)
    summary = summary[summary["Reference"] == "validation"]

    fig = px.bar(
        summary,
        x="Reference",
        y="Mean_Score",
        color="Model",
        barmode="group",
        title=f"Mean JSD across reduced dimensionality methods. Best model: {best_model}",
        labels={"Mean_Score": "Mean Score", "Reference": "Dataset Type"},
        color_discrete_sequence=color_palette,
    )

    fig.update_layout(
        title_x=0.5,
        yaxis_title="Mean JSD",
        xaxis_title="Model",
        template="plotly_white",
        font=dict(size=11),
    )
    plot_path = Path(output_dir) / "reduced_dim_summary.png"
    fig.write_image(plot_path)
    print(f"Saved tuning summary plot for tuning method {tuning_config.tuning_method} to {plot_path}")
