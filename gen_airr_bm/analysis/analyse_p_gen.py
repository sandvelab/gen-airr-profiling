import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from gen_airr_bm.constants.dataset_split import DatasetSplit
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.plotting_utils import get_collection_specification_for_title


def run_p_gen_analysis(analysis_config: AnalysisConfig) -> None:
    """ Runs the p_gen analysis: compares the generation probabilities that each model assigns to its train and test
    sequences. The p_gens are computed in the sampling step (compute_p_gens: True) and read from
    <root_output_dir>/{train,test}_p_gens/<model_name>/.
    Train and test sequences do not overlap, so the two sets are compared as distributions, not pairwise. p_gens of
    different model types are not directly comparable, so two per-repertoire, within-model quantities are plotted as
    well: the gap between median train and test log10 p_gen, and the AUC of separating train from test sequences by
    p_gen, which only depends on ranks and is therefore comparable across models.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        None
    """
    print("Running p_gen analysis...")

    output_dir = analysis_config.analysis_output_dir
    os.makedirs(output_dir, exist_ok=True)

    p_gens_df = collect_p_gens(analysis_config)
    summary_df = summarize_p_gens(p_gens_df)
    summary_df.to_csv(f"{output_dir}/p_gen_summary.tsv", sep="\t", index=False)

    plot_p_gen_distributions(p_gens_df, analysis_config.model_names, output_dir, analysis_config.receptor_type)
    plot_per_repertoire_scores(summary_df, "median_log10_p_gen_gap", 0, "0 = no difference",
                               "Median log10 p_gen (train - test)",
                               "Train - Test Gap in Median log10 p_gen", "p_gen_train_test_gap",
                               analysis_config.model_names, output_dir, analysis_config.receptor_type)
    plot_per_repertoire_scores(summary_df, "membership_auc", 0.5, "0.5 = no preference for train sequences",
                               "P(train p_gen > test p_gen)", "Membership Inference AUC from p_gen",
                               "p_gen_membership_auc", analysis_config.model_names, output_dir,
                               analysis_config.receptor_type)


def collect_p_gens(analysis_config: AnalysisConfig) -> pd.DataFrame:
    """ Reads the train and test p_gens of all models into one long table.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        pd.DataFrame: Columns model, split, file, junction_aa and p_gen.
    """
    p_gens = []
    for model_name in analysis_config.model_names:
        for split in DatasetSplit.train_and_test():
            p_gens_dir = Path(analysis_config.root_output_dir) / f"{split.value}_p_gens" / model_name
            p_gens_files = sorted(p_gens_dir.glob("*.tsv"))
            if not p_gens_files:
                raise FileNotFoundError(f"No {split.value} p_gens found for {model_name} in {p_gens_dir}. Run "
                                        f"sampling with compute_p_gens: True for this model first.")
            for p_gens_file in p_gens_files:
                p_gens.append(read_p_gens(p_gens_file).assign(model=model_name, split=split.value,
                                                              file=p_gens_file.stem))

    return pd.concat(p_gens, ignore_index=True)[["model", "split", "file", "junction_aa", "p_gen"]]


def read_p_gens(p_gens_file: Path) -> pd.DataFrame:
    """ Reads one p_gens file written by the sampling step and renames its p_gen column (named after the model by
    immuneML, e.g. p_gen_model) to p_gen.
    Args:
        p_gens_file (Path): Path to the p_gens file.
    Returns:
        pd.DataFrame: Columns junction_aa and p_gen.
    """
    p_gens_df = pd.read_csv(p_gens_file, sep="\t")
    p_gen_columns = [column for column in p_gens_df.columns if column.startswith("p_gen")]
    if len(p_gen_columns) != 1:
        raise ValueError(f"Expected one p_gen column in {p_gens_file}, found {p_gen_columns}.")

    return p_gens_df[["junction_aa", p_gen_columns[0]]].rename(columns={p_gen_columns[0]: "p_gen"})


def summarize_p_gens(p_gens_df: pd.DataFrame) -> pd.DataFrame:
    """ Summarizes the log10 p_gens per model and repertoire for train and test. Sequences with p_gen 0 (e.g. with a
    V/J gene or amino acid the model never saw) are counted and kept in the medians as the lowest values (log10 p_gen
    -inf), so a median is -inf only if at least half of the sequences have p_gen 0.
    Args:
        p_gens_df (pd.DataFrame): Long table from collect_p_gens.
    Returns:
        pd.DataFrame: One row per model and repertoire with sequence counts, zero p_gen counts, median train and
        test log10 p_gen, their difference (train minus test) and the AUC of separating train from test by p_gen.
    """
    summary = []
    for (model, file), file_df in p_gens_df.groupby(["model", "file"], sort=False):
        row = {"model": model, "file": file}
        for split in DatasetSplit.train_and_test():
            p_gens = file_df.loc[file_df["split"] == split.value, "p_gen"].to_numpy()
            row[f"n_{split.value}"] = len(p_gens)
            row[f"n_zero_{split.value}"] = int((p_gens == 0).sum())
            with np.errstate(divide="ignore"):
                row[f"median_log10_p_gen_{split.value}"] = np.median(np.log10(p_gens))
        with np.errstate(invalid="ignore"):
            row["median_log10_p_gen_gap"] = row["median_log10_p_gen_train"] - row["median_log10_p_gen_test"]
        row["membership_auc"] = compute_auc(file_df.loc[file_df["split"] == "train", "p_gen"],
                                            file_df.loc[file_df["split"] == "test", "p_gen"])
        summary.append(row)

    return pd.DataFrame(summary)


def compute_auc(train_p_gens: pd.Series, test_p_gens: pd.Series) -> float:
    """ Computes the membership inference AUC of separating train from test sequences by p_gen (Yeom et al., CSF 2018:
    membership inference by thresholding the model's loss). It is the probability that a random train sequence
    has a higher p_gen than a random test sequence, with ties counted as 1/2 (Mann-Whitney U / (n_train * n_test)).
    0.5 means the model scores train and test sequences alike; values above 0.5 mean it prefers the sequences it was
    trained on. Only ranks are used, so sequences with p_gen 0 are included as tied lowest values.
    Args:
        train_p_gens (pd.Series): p_gens of the train sequences.
        test_p_gens (pd.Series): p_gens of the test sequences.
    Returns:
        float: The AUC.
    """
    n_train, n_test = len(train_p_gens), len(test_p_gens)
    ranks = pd.concat([train_p_gens, test_p_gens], ignore_index=True).rank(method="average")
    train_rank_sum = ranks.iloc[:n_train].sum()

    return (train_rank_sum - n_train * (n_train + 1) / 2) / (n_train * n_test)


def get_title_specification(receptor_type: str) -> str:
    try:
        return get_collection_specification_for_title(receptor_type)
    except ValueError:
        return receptor_type


def plot_p_gen_distributions(p_gens_df: pd.DataFrame, model_names: list, output_dir: str, receptor_type: str) \
        -> None:
    """ Plots split violins of log10 p_gens per model, train on the left and test on the right, pooled over
    repertoires. The inner box marks the median and quartiles. Sequences with p_gen 0 cannot be drawn on a log scale,
    so their share per split is written above each model instead.
    Args:
        p_gens_df (pd.DataFrame): Long table from collect_p_gens.
        model_names (list): Models in plotting order.
        output_dir (str): Directory to save the plot.
        receptor_type (str): Type of receptor being analyzed (e.g. "TCR", "BCR").
    Returns:
        None
    """
    nonzero_df = p_gens_df[p_gens_df["p_gen"] > 0].assign(log10_p_gen=lambda df: np.log10(df["p_gen"]))

    fig = go.Figure()
    for split, side, color in [("train", "negative", px.colors.qualitative.Safe[0]),
                               ("test", "positive", px.colors.qualitative.Safe[1])]:
        split_df = nonzero_df[nonzero_df["split"] == split]
        fig.add_trace(go.Violin(
            x=split_df["model"],
            y=split_df["log10_p_gen"],
            name=split.capitalize(),
            legendgroup=split,
            side=side,
            line_color=color,
            box_visible=True,
            points=False,
            scalemode="width",
        ))

    fig.update_layout(
        title=dict(text=f"Generation Probabilities of Train and Test Sequences<br>"
                        f"{get_title_specification(receptor_type)}", font=dict(size=20)),
        xaxis=dict(title=dict(text="Model", font=dict(size=20)), tickangle=-45, tickfont=dict(size=18),
                   categoryorder="array", categoryarray=model_names,
                   # Plotly does not pad a category axis for violins, so the outer halves would be cut off
                   range=[-0.5, len(model_names) - 0.5]),
        yaxis=dict(title=dict(text="log10 p_gen", font=dict(size=20)), tickfont=dict(size=18)),
        violingap=0.1,
        violinmode="overlay",
        legend=dict(font=dict(size=18)),
        template="plotly_white",
        margin=dict(t=170),
    )
    add_zero_p_gen_annotations(fig, p_gens_df, model_names)

    png_path = os.path.join(output_dir, "p_gen_distributions.png")
    fig.write_image(png_path, scale=3)
    print(f"Plot saved as png at: {png_path}")


def add_zero_p_gen_annotations(fig: go.Figure, p_gens_df: pd.DataFrame, model_names: list) -> None:
    """ Writes the percentage of train and test sequences with p_gen 0 above each model's violins. """
    zero_shares = (p_gens_df["p_gen"] == 0).groupby([p_gens_df["model"], p_gens_df["split"]]).mean() * 100
    for model_name in model_names:
        fig.add_annotation(x=model_name, xref="x", y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"p_gen = 0<br>train {zero_shares.get((model_name, 'train'), 0):.1f}% / "
                                f"test {zero_shares.get((model_name, 'test'), 0):.1f}%",
                           font=dict(size=14))


def plot_per_repertoire_scores(summary_df: pd.DataFrame, score_column: str, reference_value: float,
                               reference_label: str, y_title: str, title: str, file_name: str, model_names: list,
                               output_dir: str, receptor_type: str) -> None:
    """ Plots a per-repertoire score for each model as a box with one point per repertoire, e.g. the gap between
    median train and test log10 p_gen (0 means no difference) or the membership inference AUC (0.5 means no
    difference). Values above the
    reference mean that the model assigns higher probabilities to the sequences it was trained on than to held-out
    sequences. Repertoires with an infinite gap (at least half of a set has p_gen 0) are left out and reported.
    Args:
        summary_df (pd.DataFrame): Summary table from summarize_p_gens.
        score_column (str): Column of summary_df to plot.
        reference_value (float): Value of the score when train and test are scored alike, drawn as a dashed line.
        reference_label (str): Label of the dashed line.
        y_title (str): Title of the y-axis.
        title (str): First line of the plot title; the second line names the collection and notes that each point is
            one repertoire.
        file_name (str): Name of the output png file (without extension).
        model_names (list): Models in plotting order.
        output_dir (str): Directory to save the plot.
        receptor_type (str): Type of receptor being analyzed (e.g. "TCR", "BCR").
    Returns:
        None
    """
    finite_df = summary_df[np.isfinite(summary_df[score_column])]
    if len(finite_df) < len(summary_df):
        print(f"{file_name}: left out {len(summary_df) - len(finite_df)} repertoire(s) with a non-finite "
              f"{score_column}, see p_gen_summary.tsv.")

    fig = go.Figure(go.Box(
        x=finite_df["model"],
        y=finite_df[score_column],
        boxpoints="all",
        jitter=0.3,
        pointpos=0,
        marker=dict(color=px.colors.qualitative.Safe[0]),
        showlegend=False,
    ))
    fig.add_hline(y=reference_value, line=dict(color="black", dash="dash"), annotation_text=reference_label,
                  annotation_position="bottom right", annotation_font=dict(size=16, color="black"))

    fig.update_layout(
        title=dict(text=f"{title}<br>{get_title_specification(receptor_type)}, one point per repertoire",
                   font=dict(size=20)),
        xaxis=dict(title=dict(text="Model", font=dict(size=20)), tickangle=-45, tickfont=dict(size=18),
                   categoryorder="array", categoryarray=model_names),
        yaxis=dict(title=dict(text=y_title, font=dict(size=20)), tickfont=dict(size=18)),
        template="plotly_white",
    )

    png_path = os.path.join(output_dir, f"{file_name}.png")
    fig.write_image(png_path, scale=3)
    print(f"Plot saved as png at: {png_path}")
