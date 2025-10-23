import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from gen_airr_bm.core.tuning_config import TuningConfig
from gen_airr_bm.utils.tuning_utils import validate_analyses_data

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gen_airr_bm.utils.tuning_utils import format_value


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
    analyses_dir = (Path(root_output_dir) / "analyses/reduced_dimensionality" / '_'.join(tuning_config.subfolder_name.split()) /
                    '_'.join(reference_data))

    result_files = glob.glob(os.path.join(analyses_dir, "*.tsv"))
    amino_acid_dfs, other_dfs = [], []
    for f in result_files:
        if not f.endswith("_ref.tsv"):
            df = pd.read_csv(f, sep="\t")
            df["Reference"] = df["Reference"].str.replace("test", "validation")
            if "aminoacid" in f.split("/")[-1]:
                amino_acid_dfs.append(df)
            else:
                other_dfs.append(df)

    aa_data = pd.concat(amino_acid_dfs, ignore_index=True)
    aa_summary = aa_data.groupby(["Reference", "Model"], as_index=False)["abs_diff_to_ref"].mean()
    dfs = other_dfs + [aa_summary]

    all_data = pd.concat(dfs, ignore_index=True)
    summary = all_data.groupby(["Reference", "Model"], as_index=False)["abs_diff_to_ref"].mean()

    val_summary = summary[summary["Reference"] == "validation"]
    best_model = val_summary.loc[val_summary["abs_diff_to_ref"].idxmin()]["Model"]

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

    summary = summary.sort_values(by="abs_diff_to_ref")
    summary = summary[summary["Reference"] == "validation"]

    # fig = px.scatter(
    #     summary,
    #     x="Model",
    #     y="abs_diff_to_ref",
    #     title=f"Mean absolute difference to reference JSD score. Best model: {best_model}",
    #     labels={"abs_diff_to_ref": "Mean absolute difference", "Reference": "Dataset Type"},
    # )

    # fig.update_layout(
    #     title_x=0.5,
    #     yaxis_title="Mean abs JSD diff",
    #     xaxis_title="Model",
    #     # template="plotly_white",
    #     font=dict(size=11),
    # )

    #lstm_hyperparameters_path = "data/lstm_hyperparameters.tsv"
    lstm_hyperparameters_path = "data/sonnia_hyperparameters.tsv"
    lstm_hyperparams = pd.read_csv(lstm_hyperparameters_path, sep="\t")
    hyperparams_long = (
        lstm_hyperparams.set_index("Hyperparameters")
        .T
        .reset_index()
        .rename(columns={"index": "Model"})
    )
    summary = summary.merge(hyperparams_long, on="Model", how="left")
    summary_sorted = summary.sort_values("abs_diff_to_ref", ascending=True) #.head(15)

    models = summary_sorted["Model"].tolist()

    # list of hyperparameters to show
    # param_names = ["epochs", "batch_size", "hidden_size", "learning_rate",
    #                "embedding_size", "temperature", "number_layers"]
    param_names = ["epochs", "batch_size", "n_gen_seqs"]

    # extract the values for each hyperparameter (rows)
    param_values = summary_sorted[param_names].T.values  # shape: (n_params, n_models)

    # convert all values to string for display
    param_text = summary_sorted[param_names].applymap(format_value).T.values

    # --- Build subplots ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.03,
        specs=[[{"type": "scatter"}],
               [{"type": "heatmap"}]]
    )

    # --- Scatter plot (top) ---
    fig.add_trace(
        go.Scatter(
            x=models,
            y=summary_sorted["abs_diff_to_ref"],
            mode="markers",
            marker=dict(size=8, color="royalblue"),
            name="Mean abs JSD diff"
        ),
        row=1, col=1
    )

    # --- Heatmap table (bottom) ---
    fig.add_trace(
        go.Heatmap(
            z=np.zeros_like(param_values, dtype=float),  # blank heatmap (we'll only show text)
            x=models,
            y=param_names,
            text=param_text,
            texttemplate="%{text}",
            colorscale=[[0, "white"], [1, "white"]],  # make it look like a table
            showscale=False
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=800,
        width=1300,
        title=f"Mean absolute difference to reference JSD score. Best model: {best_model}",
        font=dict(size=14),
        yaxis_title="Mean abs JSD diff",
        title_x=0.5,
        showlegend=False,
        xaxis=dict(tickangle=45),
    )

    fig.update_yaxes(showgrid=False, row=2, col=1)

    plot_path = Path(output_dir) / "reduced_dim_summary.png"
    fig.write_image(plot_path)
    print(f"Saved tuning summary plot for tuning method {tuning_config.tuning_method} to {plot_path}")
