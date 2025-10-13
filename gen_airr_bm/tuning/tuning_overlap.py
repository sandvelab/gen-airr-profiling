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
    overlap_difference_df = compute_overlap_difference(tuning_config, memorization_df, precision_df,
                                                       memorization_mean_ref_score)

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
    memorization_df = pd.read_csv(str(memorization_path) + ".tsv", sep="\t")

    with open(str(memorization_path) + "_mean_ref.tsv", "r") as f:
        memorization_mean_ref_score = float(f.readline().strip())

    precision_path = glob.glob(str(root_output_dir) + "/analyses/precision_recall/" + '_'.join(model_names) +
                               "/precision/*.tsv")
    precision_df = pd.read_csv(precision_path[0], sep="\t")

    return memorization_df, memorization_mean_ref_score, precision_df


def compute_overlap_difference(tuning_config: TuningConfig, memorization_df, precision_df, memorization_mean_ref_score):
    """ Computes the difference between the overlap results from memorization and precision analyses.
    Args:
        tuning_config: Configuration for the tuning, including paths and model names.
        memorization_df: DataFrame containing results from memorization analysis.
        precision_df: DataFrame containing results from precision analysis.
    Returns:
        DataFrame: DataFrame containing the overlap difference.
    """
    precision_scores = precision_df[["Model", "Mean_Score"]].sort_values(by="Model")
    memorization_scores = memorization_df[["model", "mean_overlap_score"]].sort_values(by="model")

    rows = []
    for k in tuning_config.k_values:
        memorization_scores["abs_mem_diffs"] = abs(memorization_scores["mean_overlap_score"].values - memorization_mean_ref_score)
        overlap_difference = precision_scores["Mean_Score"].values - k * memorization_scores["abs_mem_diffs"].values
        for model, diff in zip(precision_scores["Model"].values, overlap_difference):
            rows.append({
                "Model": model,
                "Overlap_Difference": diff,
                "precision": precision_scores.loc[precision_scores["Model"] == model, "Mean_Score"].values[0],
                "abs_mem_diff": memorization_scores.loc[memorization_scores["model"] == model, "abs_mem_diffs"].values[0],
                "k_value": k
            })
    overlap_difference_df = pd.DataFrame(rows)

    overlap_difference_path = Path(tuning_config.tuning_output_dir) / "overlap_difference.tsv"
    overlap_difference_df.to_csv(overlap_difference_path, sep="\t", index=False)
    return overlap_difference_df


def plot_overlap_results(overlap_difference_df, output_dir):
    """ Plots the overlap difference results.
    Args:
        overlap_difference_df: DataFrame containing the overlap difference results.
        output_dir: Directory to save the plots.
    Returns:
        None
    """
    scatterplot_df = overlap_difference_df.groupby("Model", as_index=False).agg({
        "precision": "mean",
        "abs_mem_diff": "mean"
    })

    scatterplot_df["Model_Sort"] = scatterplot_df["Model"].apply(lambda x: int(x.split('_')[-1]) if '_' in x and
                                                                 x.split('_')[-1].isdigit() else x)
    scatterplot_df = scatterplot_df.sort_values(by="Model_Sort")
    model_sort = scatterplot_df["Model"].tolist()

    fig1 = px.scatter(
        scatterplot_df,
        x="precision",
        y="abs_mem_diff",
        color="Model",
        symbol="Model",
        color_discrete_sequence=px.colors.qualitative.Dark24,
        category_orders={"Model": model_sort},
        title="Mean Precision vs Absolute Memorization Difference per Model"
    )

    fig1.update_traces(marker=dict(size=6))
    fig1.update_layout(
        width=800,
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

    plot1_path = Path(output_dir) / "overlap_scatter.png"
    fig1.write_image(plot1_path)

    overlap_difference_df["Model_Sort"] = overlap_difference_df["Model"].apply(lambda x: int(x.split('_')[-1]) if '_' in x and
                                                                x.split('_')[-1].isdigit() else x)
    overlap_difference_df = overlap_difference_df.sort_values(by="Model_Sort")
    overlap_difference_df["k_value"] = overlap_difference_df["k_value"].astype(str)
    k_sorted = sorted(overlap_difference_df["k_value"].unique())
    fig2 = px.scatter(
        overlap_difference_df,
        x="Model",
        y="Overlap_Difference",
        color="k_value",
        category_orders={"k_value": k_sorted},
        title="Overlap Score by Model and k-value",
        hover_data=["precision", "abs_mem_diff"]
    )

    fig2.update_layout(
        xaxis_title="Model",
        yaxis_title="Overlap Score",
        legend_title="k value"
    )

    plot2_path = Path(output_dir) / "overlap_difference_by_k.png"
    fig2.write_image(plot2_path)
