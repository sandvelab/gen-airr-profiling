import os
from pathlib import Path
import numpy as np
import pandas as pd

from gen_airr_bm.core.tuning_config import TuningConfig
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def format_value(x):
    """Return int if looks like int, otherwise rounded float."""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        # show 5 decimals max, drop trailing zeros
        return f"{x:.5f}".rstrip("0").rstrip(".")
    return str(x)


def validate_analyses_data(tuning_config: TuningConfig, required_analyses: list) -> list:
    """ Validates that the necessary analyses for the tuning method have been run.
    Args:
        tuning_config: Configuration for the tuning, including paths and model names.
        required_analyses (list): List of required analyses to validate.
    Returns:
        list: List of validated analysis paths.
    """
    subfolder_name = tuning_config.subfolder_name
    model_names = tuning_config.model_names
    analyses_dir = Path(tuning_config.root_output_dir) / "analyses"

    validated_analyses_paths = []
    for analysis in required_analyses:
        analysis_path = analyses_dir / analysis / '_'.join(subfolder_name.split())
        if not os.path.exists(analysis_path):
            raise FileNotFoundError(f"Required analysis '{analysis}' with models {model_names} not found in "
                                    f"{analysis_path}. Please run this analysis before tuning using method "
                                    f"'{tuning_config.tuning_method}'.")
        else:
            validated_analyses_paths.append(analysis_path)

    return validated_analyses_paths


def save_and_plot_tuning_results(tuning_config: TuningConfig, analysis_name: str, summary_df: pd.DataFrame,
                                 output_dir: str, plot_title: str, plot_ascending_scores: bool = True) -> None:
    """ Saves the tuning summary to a TSV file and generates a scatter plot with table of hyperparameter values.
    Args:
        tuning_config: Configuration for the tuning, including paths and model names.
        analysis_name: Name of the analysis (e.g., 'aminoacid', 'kmer', 'length', 'precision').
        summary_df: DataFrame containing the summary of analyses results.
        output_dir: Directory where the summary and plot will be saved.
        plot_title: Title for the generated plot.
        plot_ascending_scores: Whether to plot scores in ascending order (default is False).
    Returns:
        None
    """
    summary_path = Path(output_dir) / f"{analysis_name}_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t")
    print(f"Saved tuning summary for tuning method {tuning_config.tuning_method} to {summary_path}")

    if "Reference" in summary_df.columns:
        summary_df = summary_df[summary_df["Reference"] == "test"]
    if "k_value" in summary_df.columns:
        summary_df = summary_df[summary_df["k_value"] == summary_df["k_value"].unique()[0]]

    tuning_hyperparameters = pd.read_csv(tuning_config.hyperparameter_table_path, sep="\t")
    parameter_names = tuning_hyperparameters["Hyperparameters"].tolist()
    hyperparams_long = (
        tuning_hyperparameters.set_index("Hyperparameters")
        .T
        .reset_index()
        .rename(columns={"index": "Model"})
    )
    summary_df = summary_df.merge(hyperparams_long, on="Model", how="left")
    summary_sorted = summary_df.sort_values("Score", ascending=plot_ascending_scores)

    models = summary_sorted["Model"].tolist()
    parameter_values = summary_sorted[parameter_names].T.values
    parameter_text = summary_sorted[parameter_names].applymap(format_value).T.values

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.03,
        specs=[[{"type": "scatter"}],
               [{"type": "heatmap"}]]
    )

    fig.add_trace(
        go.Scatter(
            x=models,
            y=summary_sorted["Score"],
            mode="markers",
            marker=dict(size=8, color="royalblue"),
            name="Mean abs JSD diff"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=np.zeros_like(parameter_values, dtype=float),
            x=models,
            y=parameter_names,
            text=parameter_text,
            texttemplate="%{text}",
            colorscale=[[0, "white"], [1, "white"]],
            showscale=False
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=800,
        width=1300,
        title=plot_title,
        font=dict(size=14),
        yaxis_title="Mean Score",
        title_x=0.5,
        showlegend=False,
        xaxis=dict(tickangle=45),
    )

    fig.update_yaxes(showgrid=False, row=2, col=1)

    plot_path = Path(output_dir) / f"{analysis_name}_summary.png"
    fig.write_image(plot_path)
    print(f"Saved tuning plot for tuning method {tuning_config.tuning_method} to {plot_path}")