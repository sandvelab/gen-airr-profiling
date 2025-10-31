import glob
import os
from pathlib import Path
import pandas as pd
import plotly.express as px

from gen_airr_bm.core.tuning_config import TuningConfig
from gen_airr_bm.utils.tuning_utils import validate_analyses_data, save_and_plot_tuning_results


def run_overlap_tuning(tuning_config: TuningConfig) -> None:
    """ Runs hyperparameter tuning by overlap with train and validation reference set.
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
    overlap_score_df = compute_overlap_score(tuning_config, mem_df, prec_rec_df)

    plot_precision_memorization_scatter(overlap_score_df, tuning_config.tuning_output_dir, mem_mean_ref_score,
                                        prec_mean_ref_score)
    plot_tuning_score_by_k(overlap_score_df, tuning_config.tuning_output_dir)

    save_and_plot_tuning_results(tuning_config, "overlap", overlap_score_df,
                                 tuning_config.tuning_output_dir, plot_title="Tuning score based on realism")


def get_overlap_results(tuning_config: TuningConfig) -> tuple:
    """ Collects results from memorization and precision analyses for hyperparameter tuning.
    Args:
        tuning_config: Configuration for the tuning, including paths and model names.
    Returns:
        tuple: DataFrame of memorization results, mean reference memorization score, and DataFrame of precision results.
    """
    root_output_dir = tuning_config.root_output_dir
    memorization_path = Path(root_output_dir) / "analyses/memorization" / '_'.join(tuning_config.subfolder_name.split()) / "memorization"
    memorization_df = pd.read_csv(str(memorization_path) + ".tsv", sep="\t")

    with open(str(memorization_path) + "_mean_ref.tsv", "r") as f:
        memorization_mean_ref_score = float(f.readline().strip())

    precision_recall_path = glob.glob(str(root_output_dir) + "/analyses/precision_recall/" + '_'.join(tuning_config.subfolder_name.split()) +
                                      "/test/precision_recall_data.tsv")
    precision_recall_df = pd.read_csv(precision_recall_path[0], sep="\t")
    precision_mean_ref_score = precision_recall_df[precision_recall_df["Model"] == "upper_reference"]["Precision_mean"].values[0]

    return memorization_df, memorization_mean_ref_score, precision_recall_df, precision_mean_ref_score


def compute_overlap_score(tuning_config: TuningConfig, memorization_df: pd.DataFrame,
                          precision_recall_df: pd.DataFrame) -> pd.DataFrame:
    """ Computes the tuning scores from memorization and precision analyses.
    Args:
        tuning_config: Configuration for the tuning, including paths and model names.
        memorization_df: DataFrame containing results from memorization analysis.
        precision_recall_df: DataFrame containing results from precision analysis.
    Returns:
        DataFrame: DataFrame containing the overlap difference.
    """
    precision_scores = precision_recall_df[precision_recall_df[["Model", "Precision_mean"]].sort_values(by="Model")
                                    ["Model"] != "upper_reference"]
    memorization_scores = memorization_df[["model", "mean_overlap_score"]].sort_values(by="model")

    rows = []
    for k in tuning_config.k_values:
        overlap_score_k_scaled = (precision_scores["Precision_mean"].values + k *
                                  memorization_scores["mean_overlap_score"].values)
        overlap_score = precision_scores["Precision_mean"].values
        for model, score_k, score, prec, mem in zip(precision_scores["Model"].values,
                                                    overlap_score_k_scaled,
                                                    overlap_score,
                                                    precision_scores["Precision_mean"].values,
                                                    memorization_scores["mean_overlap_score"].values):
            rows.append({
                "Model": model,
                "Overlap_score_k_scaled": score_k,
                "Score": score,
                "Realism": prec,
                "Memorization": mem,
                "k_value": k
            })
    overlap_score_df = pd.DataFrame(rows)
    return overlap_score_df


def plot_precision_memorization_scatter(overlap_score_df: pd.DataFrame, output_dir: str,
                                        memorization_mean_ref_score: float, precision_mean_ref_score: float) -> None:
    """ Plots the precision and memorization scores as scatter plots.
    Args:
        overlap_score_df: DataFrame containing the overlap difference results.
        output_dir: Directory to save the plots.
        memorization_mean_ref_score: Mean reference score from memorization analysis.
        precision_mean_ref_score: Mean reference score from precision analysis.
    Returns:
        None
    """
    scatterplot_df = overlap_score_df.groupby("Model", as_index=False).agg({
        "Realism": "mean",
        "Memorization": "mean"
    })

    scatterplot_df["model_sort"] = scatterplot_df["Model"].apply(lambda x: int(x.split('_')[-1]) if '_' in x and
                                                                 x.split('_')[-1].isdigit() else x)
    model_sort = scatterplot_df["Model"].tolist()

    fig = px.scatter(
        scatterplot_df,
        x="Realism",
        y="Memorization",
        color="Model",
        symbol="Model",
        color_discrete_sequence=px.colors.qualitative.Dark24,
        category_orders={"Model": model_sort},
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

    x_max = max(scatterplot_df["Realism"].max(), precision_mean_ref_score)
    y_max = max(scatterplot_df["Memorization"].max(), memorization_mean_ref_score)
    x_min = min(scatterplot_df["Realism"].min(), precision_mean_ref_score)
    y_min = min(scatterplot_df["Memorization"].min(), memorization_mean_ref_score)

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
    print(f"Plot saved as PNG at: {plot_path}")


def plot_tuning_score_by_k(overlap_score_df: pd.DataFrame, output_dir: str) -> None:
    """ Plots the overlap tuning scores by model and k-value.
    Args:
        overlap_score_df: DataFrame containing the overlap difference results.
        output_dir: Directory to save the plots.
    Returns:
        None
    """
    overlap_score_df["model_sort"] = overlap_score_df["Model"].apply(
        lambda x: int(x.split('_')[-1]) if '_' in x and
                                           x.split('_')[-1].isdigit() else x)
    overlap_score_df = overlap_score_df.sort_values(by="Overlap_score_k_scaled")
    overlap_score_df["k_value"] = overlap_score_df["k_value"].astype(str)
    k_sorted = sorted(overlap_score_df["k_value"].unique())
    fig = px.scatter(
        overlap_score_df,
        x="Model",
        y="Overlap_score_k_scaled",
        color="k_value",
        category_orders={"k_value": k_sorted},
        title="Overlap Tuning Score Based on Realism and Scaled Memorization"
    )

    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Overlap Score",
        legend_title="k value"
    )

    plot_path = Path(output_dir) / "overlap_score_by_k.png"
    fig.write_image(plot_path)
    print(f"Plot saved as PNG at: {plot_path}")
