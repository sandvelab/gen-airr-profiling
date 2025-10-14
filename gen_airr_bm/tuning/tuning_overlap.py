import glob
import os
from pathlib import Path

import pandas as pd
import plotly.express as px

from gen_airr_bm.core.tuning_config import TuningConfig
from gen_airr_bm.utils.tuning_utils import validate_analyses_data


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

    memorization_df, memorization_mean_ref_score, precision_df, precision_mean_ref_score = get_overlap_results(tuning_config)
    overlap_difference_df = compute_overlap_difference(tuning_config, memorization_df, precision_df,
                                                       memorization_mean_ref_score, precision_mean_ref_score)

    plot_precision_memorization_scatter(overlap_difference_df, tuning_config.tuning_output_dir,
                                        memorization_mean_ref_score, precision_mean_ref_score)
    plot_tuning_score_by_k(overlap_difference_df, tuning_config.tuning_output_dir)


def get_overlap_results(tuning_config: TuningConfig) -> tuple:
    """ Collects results from memorization and precision analyses for tuning purposes.
    Args:
        tuning_config: Configuration for the tuning, including paths and model names.
    Returns:
        tuple: DataFrame of memorization results, mean reference memorization score, and DataFrame of precision results.
    """
    root_output_dir = tuning_config.root_output_dir
    model_names = tuning_config.model_names
    memorization_path = Path(root_output_dir) / "analyses/memorization" / '_'.join(model_names) / "memorization"
    memorization_df = pd.read_csv(str(memorization_path) + ".tsv", sep="\t")

    with open(str(memorization_path) + "_mean_ref.tsv", "r") as f:
        memorization_mean_ref_score = float(f.readline().strip())

    mean_precision_ref_path = glob.glob(str(root_output_dir) + "/analyses/precision_recall/" + '_'.join(model_names) +
                                        "/test/precision_recall_data.tsv")
    precision_df = pd.read_csv(mean_precision_ref_path[0], sep="\t")
    precision_mean_ref_score = precision_df[precision_df["Model"] == "upper_reference"]["Precision_mean"].values[0]

    return memorization_df, memorization_mean_ref_score, precision_df, precision_mean_ref_score


def compute_overlap_difference(tuning_config: TuningConfig, memorization_df: pd.DataFrame, precision_df: pd.DataFrame,
                               memorization_mean_ref_score: float, precision_mean_ref_score: float) -> pd.DataFrame:
    """ Computes the difference between the overlap results from memorization and precision analyses.
    Args:
        tuning_config: Configuration for the tuning, including paths and model names.
        memorization_df: DataFrame containing results from memorization analysis.
        precision_df: DataFrame containing results from precision analysis.
        memorization_mean_ref_score: Mean reference score from memorization analysis.
        precision_mean_ref_score: Mean reference score from precision analysis.
    Returns:
        DataFrame: DataFrame containing the overlap difference.
    """
    precision_scores = precision_df[precision_df[["Model", "Precision_mean"]].sort_values(by="Model")
                                    ["Model"] != "upper_reference"]
    memorization_scores = memorization_df[["model", "mean_overlap_score"]].sort_values(by="model")

    rows = []
    for k in tuning_config.k_values:
        abs_memorization_diff = abs(memorization_scores["mean_overlap_score"].values - memorization_mean_ref_score)
        abs_precision_diff = abs(precision_scores["Precision_mean"].values - precision_mean_ref_score)
        overlap_difference = 1 - (abs_precision_diff + k * abs_memorization_diff)
        for model, score, prec, mem, abs_prec, abs_mem in zip(precision_scores["Model"].values,
                                                              overlap_difference,
                                                              precision_scores["Precision_mean"].values,
                                                              memorization_scores["mean_overlap_score"].values,
                                                              abs_precision_diff,
                                                              abs_memorization_diff):
            rows.append({
                "model": model,
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
    scatterplot_df = scatterplot_df.sort_values(by="model_sort")
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

    fig.add_shape(
        type="line",
        x0=precision_mean_ref_score, x1=precision_mean_ref_score,
        y0=scatterplot_df["memorization"].min(), y1=scatterplot_df["memorization"].max(),
        line=dict(color="red", width=0.8, dash="dash")
    )

    fig.add_shape(
        type="line",
        x0=scatterplot_df["realism"].min(), x1=scatterplot_df["realism"].max(),
        y0=memorization_mean_ref_score, y1=memorization_mean_ref_score,
        line=dict(color="red", width=0.8, dash="dash")
    )

    fig.add_annotation(
        x=precision_mean_ref_score, y=scatterplot_df["memorization"].max(),
        text=f"x = {precision_mean_ref_score:.2f}",
        showarrow=False,
        font=dict(color="red", size=10),
        xshift=-30
    )

    fig.add_annotation(
        x=scatterplot_df["realism"].min(), y=memorization_mean_ref_score,
        text=f"y = {memorization_mean_ref_score:.2f}",
        showarrow=False,
        font=dict(color="red", size=10),
        yshift=10
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
    overlap_difference_df = overlap_difference_df.sort_values(by="model_sort")
    overlap_difference_df["k_value"] = overlap_difference_df["k_value"].astype(str)
    k_sorted = sorted(overlap_difference_df["k_value"].unique())
    fig = px.scatter(
        overlap_difference_df,
        x="model",
        y="overlap_difference",
        color="k_value",
        category_orders={"k_value": k_sorted},
        title="Overlap Score by Model and k-value",
        hover_data=["abs_realism_diff", "abs_memorization_diff"]
    )

    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Overlap Score",
        legend_title="k value"
    )

    plot_path = Path(output_dir) / "overlap_score_by_k.png"
    fig.write_image(plot_path)
