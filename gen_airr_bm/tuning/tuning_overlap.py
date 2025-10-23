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


def run_overlap_tuning(tuning_config: TuningConfig) -> None:
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

    mem_df, mem_mean_ref_score, prec_rec_df, prec_mean_ref_score = get_overlap_results(tuning_config)
    overlap_difference_df = compute_overlap_difference(tuning_config, mem_df, prec_rec_df,
                                                       mem_mean_ref_score, prec_mean_ref_score)

    plot_precision_memorization_scatter(overlap_difference_df, tuning_config.tuning_output_dir,
                                        mem_mean_ref_score, prec_mean_ref_score)
    plot_tuning_score_by_k(overlap_difference_df, tuning_config.tuning_output_dir)
    plot_tuning_score(overlap_difference_df, tuning_config.tuning_output_dir)


def get_overlap_results(tuning_config: TuningConfig) -> tuple:
    """ Collects results from memorization and precision analyses for tuning purposes.
    Args:
        tuning_config: Configuration for the tuning, including paths and model names.
    Returns:
        tuple: DataFrame of memorization results, mean reference memorization score, and DataFrame of precision results.
    """
    root_output_dir = tuning_config.root_output_dir
    model_names = tuning_config.model_names
    memorization_path = Path(root_output_dir) / "analyses/memorization" / '_'.join(tuning_config.subfolder_name.split()) / "memorization"
    memorization_df = pd.read_csv(str(memorization_path) + ".tsv", sep="\t")

    with open(str(memorization_path) + "_mean_ref.tsv", "r") as f:
        memorization_mean_ref_score = float(f.readline().strip())

    precision_recall_path = glob.glob(str(root_output_dir) + "/analyses/precision_recall/" + '_'.join(tuning_config.subfolder_name.split()) +
                                        "/test/precision_recall_data.tsv")
    precision_recall_df = pd.read_csv(precision_recall_path[0], sep="\t")
    precision_mean_ref_score = precision_recall_df[precision_recall_df["Model"] == "upper_reference"]["Precision_mean"].values[0]

    return memorization_df, memorization_mean_ref_score, precision_recall_df, precision_mean_ref_score


def compute_overlap_difference(tuning_config: TuningConfig, memorization_df: pd.DataFrame, precision_recall_df: pd.DataFrame,
                               memorization_mean_ref_score: float, precision_mean_ref_score: float) -> pd.DataFrame:
    """ Computes the difference between the overlap results from memorization and precision analyses.
    Args:
        tuning_config: Configuration for the tuning, including paths and model names.
        memorization_df: DataFrame containing results from memorization analysis.
        precision_recall_df: DataFrame containing results from precision analysis.
        memorization_mean_ref_score: Mean reference score from memorization analysis.
        precision_mean_ref_score: Mean reference score from precision analysis.
    Returns:
        DataFrame: DataFrame containing the overlap difference.
    """
    precision_scores = precision_recall_df[precision_recall_df[["Model", "Precision_mean"]].sort_values(by="Model")
                                    ["Model"] != "upper_reference"]
    recall_scores = precision_recall_df[precision_recall_df[["Model", "Recall_mean"]].sort_values(by="Model")
                                    ["Model"] != "upper_reference"]
    recall_mean_ref_score = precision_recall_df[precision_recall_df["Model"] == "upper_reference"]["Recall_mean"].values[0]
    memorization_scores = memorization_df[["model", "mean_overlap_score"]].sort_values(by="model")

    rows = []
    for k in tuning_config.k_values:
        abs_memorization_diff = abs(memorization_scores["mean_overlap_score"].values - memorization_mean_ref_score)
        abs_precision_diff = abs(precision_scores["Precision_mean"].values - precision_mean_ref_score)
        abs_recall_diff = abs(recall_scores["Recall_mean"].values - recall_mean_ref_score)
        overlap_difference_k_scaled = abs_precision_diff + k * abs_memorization_diff
        overlap_difference = abs_precision_diff + abs_recall_diff + abs_memorization_diff
        for model, score_k, score, prec, mem, abs_prec, abs_mem in zip(precision_scores["Model"].values,
                                                                       overlap_difference_k_scaled,
                                                                       overlap_difference,
                                                                       precision_scores["Precision_mean"].values,
                                                                       memorization_scores["mean_overlap_score"].values,
                                                                       abs_precision_diff,
                                                                       abs_memorization_diff):
            rows.append({
                "model": model,
                "overlap_difference_k_scaled": score_k,
                "overlap_difference": score,
                "realism": prec,
                "memorization": mem,
                "abs_realism_diff": abs_prec,
                "abs_memorization_diff": abs_mem,
                "k_value": k
            })
    overlap_difference_df = pd.DataFrame(rows)

    overlap_difference_path = Path(tuning_config.tuning_output_dir) / "combined_overlap_scores.tsv"
    overlap_difference_df.to_csv(overlap_difference_path, sep="\t", index=False)
    return overlap_difference_df


def plot_precision_memorization_scatter(overlap_difference_df: pd.DataFrame, output_dir: str,
                                        memorization_mean_ref_score: float, precision_mean_ref_score: float) -> None:
    """ Plots the precision and memorization scores as scatter plots.
    Args:
        overlap_difference_df: DataFrame containing the overlap difference results.
        output_dir: Directory to save the plots.
        memorization_mean_ref_score: Mean reference score from memorization analysis.
        precision_mean_ref_score: Mean reference score from precision analysis.
    Returns:
        None
    """
    scatterplot_df = overlap_difference_df.groupby("model", as_index=False).agg({
        "realism": "mean",
        "memorization": "mean"
    })

    scatterplot_df["model_sort"] = scatterplot_df["model"].apply(lambda x: int(x.split('_')[-1]) if '_' in x and
                                                                 x.split('_')[-1].isdigit() else x)
    # scatterplot_df = scatterplot_df.sort_values(by="model_sort")
    model_sort = scatterplot_df["model"].tolist()

    fig = px.scatter(
        scatterplot_df,
        x="realism",
        y="memorization",
        color="model",
        symbol="model",
        color_discrete_sequence=px.colors.qualitative.Dark24,
        category_orders={"model": model_sort},
    )

    fig.update_traces(marker=dict(size=6))
    fig.update_layout(
        title={'text': f"Realism and memorization scores by model. Red dashed lines indicate reference scores.",
               'font': {'size': 15}},
        width=750,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=8),
        ),
        margin=dict(b=120)
    )

    x_max = max(scatterplot_df["realism"].max(), precision_mean_ref_score)
    y_max = max(scatterplot_df["memorization"].max(), memorization_mean_ref_score)
    x_min = min(scatterplot_df["realism"].min(), precision_mean_ref_score)
    y_min = min(scatterplot_df["memorization"].min(), memorization_mean_ref_score)

    fig.add_shape(
        type="line",
        x0=precision_mean_ref_score, x1=precision_mean_ref_score,
        y0=y_min, y1=y_max,
        line=dict(color="red", width=0.8, dash="dash")
    )

    fig.add_shape(
        type="line",
        x0=x_min, x1=x_max,
        y0=memorization_mean_ref_score, y1=memorization_mean_ref_score,
        line=dict(color="red", width=0.8, dash="dash")
    )

    fig.add_annotation(
        x=precision_mean_ref_score, y=y_min,
        text=f"x = {precision_mean_ref_score:.2f}",
        showarrow=False,
        font=dict(color="red", size=10),
        xshift=-30
    )

    fig.add_annotation(
        x=x_min, y=memorization_mean_ref_score,
        text=f"y = {memorization_mean_ref_score:.2f}",
        showarrow=False,
        font=dict(color="red", size=10),
        yshift=10,
        xshift=30
    )

    plot_path = Path(output_dir) / "realism_memorization_scatter.png"
    fig.write_image(plot_path)


def plot_tuning_score_by_k(overlap_difference_df: pd.DataFrame, output_dir: str) -> None:
    """ Plots the overlap tuning scores by model and k-value.
    Args:
        overlap_difference_df: DataFrame containing the overlap difference results.
        output_dir: Directory to save the plots.
    Returns:
        None
    """
    overlap_difference_df["model_sort"] = overlap_difference_df["model"].apply(
        lambda x: int(x.split('_')[-1]) if '_' in x and
                                           x.split('_')[-1].isdigit() else x)
    # overlap_difference_df = overlap_difference_df.sort_values(by="model_sort")
    overlap_difference_df = overlap_difference_df.sort_values(by="overlap_difference_k_scaled")
    overlap_difference_df["k_value"] = overlap_difference_df["k_value"].astype(str)
    k_sorted = sorted(overlap_difference_df["k_value"].unique())
    fig = px.scatter(
        overlap_difference_df,
        x="model",
        y="overlap_difference_k_scaled",
        color="k_value",
        category_orders={"k_value": k_sorted},
        title="Overlap Tuning Score Based on Realism and Scaled Memorization",
        hover_data=["abs_realism_diff", "abs_memorization_diff"]
    )

    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Overlap Score",
        legend_title="k value"
    )

    plot_path = Path(output_dir) / "overlap_score_by_k.png"
    fig.write_image(plot_path)


def plot_tuning_score(overlap_difference_df: pd.DataFrame, output_dir: str) -> None:
    """ Plots the overlap tuning scores by model.
    Args:
        overlap_difference_df: DataFrame containing the overlap difference results.
        output_dir: Directory to save the plots.
    Returns:
        None
    """
    overlap_difference_df = overlap_difference_df.sort_values(by="overlap_difference")
    overlap_difference_df = overlap_difference_df[overlap_difference_df["k_value"] ==
                                                  overlap_difference_df["k_value"].unique()[0]]
    # fig = px.scatter(
    #     overlap_difference_df,
    #     x="model",
    #     y="overlap_difference",
    #     title="Overlap Tuning Score Based on Realism, Coverage, and Memorization",
    #     hover_data=["abs_realism_diff", "abs_memorization_diff"]
    # )
    #
    # fig.update_layout(
    #     xaxis_title="Model",
    #     yaxis_title="Overlap Score",
    # )

    # lstm_hyperparameters_path = "data/lstm_hyperparameters.tsv"
    lstm_hyperparameters_path = "data/sonnia_hyperparameters.tsv"
    lstm_hyperparams = pd.read_csv(lstm_hyperparameters_path, sep="\t")
    hyperparams_long = (
        lstm_hyperparams.set_index("Hyperparameters")
        .T
        .reset_index()
        .rename(columns={"index": "model"})
    )
    summary = overlap_difference_df.merge(hyperparams_long, on="model", how="left")
    summary_sorted = summary.sort_values("overlap_difference", ascending=True)  # .head(15)

    models = summary_sorted["model"].tolist()

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
            y=summary_sorted["overlap_difference"],
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
        title=f"Tuning score based on realism, coverage, and memorization",
        font=dict(size=14),
        yaxis_title="overlap score",
        title_x=0.5,
        showlegend=False,
        xaxis=dict(tickangle=45),
    )

    fig.update_yaxes(showgrid=False, row=2, col=1)

    plot_path = Path(output_dir) / "overlap_score.png"
    fig.write_image(plot_path)
