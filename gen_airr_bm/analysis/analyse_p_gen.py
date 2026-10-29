import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from gen_airr_bm.constants.dataset_split import DatasetSplit
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.plotting_utils import get_collection_specification_for_title, plot_paired_comparisons, \
    plot_repertoire_statistics
from gen_airr_bm.utils.statistics_utils import aggregate_repertoire_scores, check_one_repertoire_per_donor, \
    compare_paired, run_repertoire_statistics, summarise_repertoire_scores, write_repertoire_statistics

# The per-repertoire scores: their description in plot titles, a short label for the axes, their value if a model
# scores its train and test sequences alike, and whether they can be compared between models. The gap in log10 p_gen
# cannot, since p_gens of different model types are on different scales; the AUC only depends on ranks.
P_GEN_SCORES = {
    "median_log10_p_gen_gap": ("Train - Test Gap in Median log10 p_gen", "Gap in Median log10 p_gen", 0.0, False),
    "membership_auc": ("Membership Inference AUC from p_gen", "Membership AUC", 0.5, True),
}


def run_p_gen_analysis(analysis_config: AnalysisConfig) -> None:
    """ Runs the p_gen analysis: compares the generation probabilities that each model assigns to its train and test
    sequences. The p_gens are computed in the sampling step (compute_p_gens: True) and read from
    <root_output_dir>/{train,test}_p_gens/<model_name>/.
    Train and test sequences do not overlap, so the two sets are compared as distributions, not pairwise. p_gens of
    different model types are not directly comparable, so two per-repertoire, within-model quantities are plotted as
    well: the gap between median train and test log10 p_gen, and the AUC of separating train from test sequences by
    p_gen, which only depends on ranks and is therefore comparable across models. Both are compared with their value
    if train and test are scored alike, and the AUC also between models, at the repertoire level (see
    run_p_gen_statistics).
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
    if analysis_config.statistical_tests:
        check_one_repertoire_per_donor(aggregate_repertoire_scores(summary_df, "membership_auc",
                                                                   analysis_config.donor_pattern,
                                                                   repertoire_column="file"),
                                       analysis_config.donor_pattern)

    plot_p_gen_distributions(p_gens_df, analysis_config.model_names, output_dir, analysis_config.receptor_type,
                             analysis_config.collection)
    plot_per_repertoire_scores(summary_df, "median_log10_p_gen_gap", 0, "0 = no difference",
                               "Median log10 p_gen (train - test)",
                               "Train - Test Gap in Median log10 p_gen", "p_gen_train_test_gap",
                               analysis_config.model_names, output_dir, analysis_config.receptor_type,
                               analysis_config.collection)
    plot_per_repertoire_scores(summary_df, "membership_auc", 0.5, "0.5 = no preference for train sequences",
                               "P(train p_gen > test p_gen)", "Membership Inference AUC from p_gen",
                               "p_gen_membership_auc", analysis_config.model_names, output_dir,
                               analysis_config.receptor_type, analysis_config.collection)
    run_p_gen_statistics(analysis_config, summary_df)


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


def get_title_specification(receptor_type: str, collection: str = None) -> str:
    try:
        return get_collection_specification_for_title(receptor_type, collection)
    except ValueError:
        return receptor_type


def plot_p_gen_distributions(p_gens_df: pd.DataFrame, model_names: list, output_dir: str, receptor_type: str,
                             collection: str = None) -> None:
    """ Plots the density of log10 p_gens of train (solid) and test (dashed) sequences, pooled over repertoires, with
    one panel per model in a grid of at most max_cols panels per row. p_gens of different model types are on different
    scales, so each panel has its own x-axis. Panels are colored by model family, so that e.g. TCRpeg_kmer and
    TCRpeg_overlap share a color. Sequences with p_gen 0 cannot be drawn on a log scale, so their share per split is
    written in the panel title instead.
    Args:
        p_gens_df (pd.DataFrame): Long table from collect_p_gens.
        model_names (list): Models in plotting order, one panel each.
        output_dir (str): Directory to save the plot.
        receptor_type (str): Type of receptor being analyzed (e.g. "TCR", "BCR").
        collection (str): Optional collection label for the plot title (e.g. "D"); inferred from the receptor type
            if not given.
    Returns:
        None
    """
    n_cols, n_rows = get_grid_shape(len(model_names))
    zero_shares = (p_gens_df["p_gen"] == 0).groupby([p_gens_df["model"], p_gens_df["split"]]).mean() * 100
    subplot_titles = [f"{model_name}<br><sup>p_gen = 0: train {zero_shares.get((model_name, 'train'), 0):.1f}% / "
                      f"test {zero_shares.get((model_name, 'test'), 0):.1f}%</sup>" for model_name in model_names]
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, horizontal_spacing=0.05,
                        vertical_spacing=0.4 / n_rows)

    nonzero_df = p_gens_df[p_gens_df["p_gen"] > 0].assign(log10_p_gen=lambda df: np.log10(df["p_gen"]))
    family_colors = get_family_colors(model_names)
    for i, model_name in enumerate(model_names):
        row, col = i // n_cols + 1, i % n_cols + 1
        model_df = nonzero_df[nonzero_df["model"] == model_name]
        if model_df.empty:
            continue
        grid = get_density_grid(model_df["log10_p_gen"].to_numpy())
        for split, dash in [("train", "solid"), ("test", "dash")]:
            density = compute_density(model_df.loc[model_df["split"] == split, "log10_p_gen"].to_numpy(), grid)
            if density is None:
                print(f"p_gen_distributions: too few distinct {split} p_gens for {model_name} to draw a density.")
                continue
            fig.add_trace(go.Scatter(x=grid, y=density, mode="lines", showlegend=False,
                                     line=dict(color=family_colors[get_model_family(model_name)], dash=dash,
                                               width=3)),
                          row=row, col=col)

    # Line style legend in neutral grey, as the colors only identify the model family of each panel
    for split, dash in [("Train", "solid"), ("Test", "dash")]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=split,
                                 line=dict(color="grey", dash=dash, width=3)), row=1, col=1)

    fig.update_xaxes(title=dict(text="log10 p_gen", font=dict(size=24)), tickfont=dict(size=20))
    fig.update_yaxes(tickfont=dict(size=20))
    fig.update_yaxes(title=dict(text="Density", font=dict(size=24)), col=1)
    fig.update_annotations(font=dict(size=24))
    fig.update_layout(
        title=dict(text=f"Generation Probabilities of Train and Test Sequences<br>"
                        f"{get_title_specification(receptor_type, collection)}", font=dict(size=28)),
        legend=dict(font=dict(size=24)),
        template="plotly_white",
        width=max(700, 450 * n_cols),
        height=150 + 480 * n_rows,
        margin=dict(t=230),
    )

    png_path = os.path.join(output_dir, "p_gen_distributions.png")
    fig.write_image(png_path, scale=3)
    print(f"Plot saved as png at: {png_path}")


def get_grid_shape(n_panels: int, max_cols: int = 5) -> tuple[int, int]:
    """ Returns the number of columns and rows of a grid with at most max_cols panels per row. """
    n_cols = min(n_panels, max_cols)
    return n_cols, -(-n_panels // n_cols)


def get_model_family(model_name: str) -> str:
    """ Returns the model family, i.e. the model name without the tuning criterion or grid search suffix, e.g.
    TCRpeg for TCRpeg_kmer and tcrsep for tcrsep_01. """
    return model_name.split("_")[0]


def get_family_colors(model_names: list) -> dict:
    """ Assigns one color of the Safe palette to each model family, in order of first appearance. """
    families = list(dict.fromkeys(get_model_family(model_name) for model_name in model_names))
    colors = px.colors.qualitative.Safe
    return {family: colors[i % len(colors)] for i, family in enumerate(families)}


def get_density_grid(values: np.ndarray, n_points: int = 200) -> np.ndarray:
    """ Returns evenly spaced points over the range of the values, padded by 10% of the range (at least 0.5) on
    each side so that the density tails are not cut off. """
    padding = max(0.1 * (values.max() - values.min()), 0.5)
    return np.linspace(values.min() - padding, values.max() + padding, n_points)


def compute_density(values: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    """ Estimates the density of the values on the grid with a Gaussian kernel density estimate (Scott's rule for the
    bandwidth). Returns None if there are fewer than two distinct values, for which no density can be estimated. """
    if len(np.unique(values)) < 2:
        return None
    return gaussian_kde(values)(grid)


def plot_per_repertoire_scores(summary_df: pd.DataFrame, score_column: str, reference_value: float,
                               reference_label: str, y_title: str, title: str, file_name: str, model_names: list,
                               output_dir: str, receptor_type: str, collection: str = None) -> None:
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
        collection (str): Optional collection label for the plot title (e.g. "D"); inferred from the receptor type
            if not given.
    Returns:
        None
    """
    finite_df = summary_df[np.isfinite(summary_df[score_column])]
    if len(finite_df) < len(summary_df):
        print(f"{file_name}: left out {len(summary_df) - len(finite_df)} repertoire(s) with a non-finite "
              f"{score_column}, see p_gen_summary.tsv.")

    colour = px.colors.qualitative.Safe[0]
    fig = go.Figure(go.Box(
        x=finite_df["model"],
        y=finite_df[score_column],
        boxpoints="all",
        jitter=0.3,
        pointpos=0,
        width=0.5,
        line=dict(width=1.2, color=colour),
        fillcolor="rgba(136, 204, 238, 0.25)",
        marker=dict(color=colour, size=5, opacity=0.7),
        showlegend=False,
    ))
    # The reference is explained above the plot, so that the label does not cover the boxes
    fig.add_hline(y=reference_value, line=dict(color="black", dash="dash", width=1.5))
    fig.add_annotation(text=f"Dashed line: {reference_label}", xref="paper", yref="paper", x=1, y=1.02,
                       xanchor="right", yanchor="bottom", showarrow=False, font=dict(size=16))

    fig.update_layout(
        title=dict(text=f"{title}<br>{get_title_specification(receptor_type, collection)}, one point per repertoire",
                   font=dict(size=20)),
        xaxis=dict(title=dict(text="Model", font=dict(size=20)), tickangle=-45, tickfont=dict(size=18),
                   categoryorder="array", categoryarray=model_names),
        yaxis=dict(title=dict(text=y_title, font=dict(size=20)), tickfont=dict(size=18)),
        template="plotly_white",
        # Room for the two-line title and the reference caption above the plot
        margin=dict(t=150),
        # The figure grows with the number of models, so that the boxes are not squeezed together
        width=max(700, 200 + 90 * len(model_names)),
        height=650,
    )

    png_path = os.path.join(output_dir, f"{file_name}.png")
    fig.write_image(png_path, scale=3)
    print(f"Plot saved as png at: {png_path}")


def run_p_gen_statistics(analysis_config: AnalysisConfig, summary_df: pd.DataFrame) -> None:
    """ Runs the repertoire-level comparisons of the per-repertoire p_gen scores. Each model is compared with the
    value the score takes if train and test are scored alike (a gap of 0, an AUC of 0.5), which pairs every repertoire
    with that constant, so that the test is a one-sample t-test across repertoires. The AUC is also compared between
    models, in the contrasts from the config and for all pairs. These comparisons are descriptive, or include
    confidence intervals and tests if statistical_tests is set. Repertoires with a non-finite gap are left out.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        summary_df (pd.DataFrame): Summary table from summarize_p_gens.
    Returns:
        None
    """
    collection_specification = get_title_specification(analysis_config.receptor_type, analysis_config.collection)
    for score_column, (description, label, null_value, comparable) in P_GEN_SCORES.items():
        output_prefix = f"{analysis_config.analysis_output_dir}/statistics/p_gen_{score_column}"
        finite_df = summary_df.assign(score=summary_df[score_column].where(np.isfinite(summary_df[score_column])))
        repertoire_scores = aggregate_repertoire_scores(finite_df, "score", analysis_config.donor_pattern,
                                                        repertoire_column="file")
        models = [model for model in analysis_config.model_names if model in set(repertoire_scores["source"])]
        if comparable:
            results = run_repertoire_statistics(repertoire_scores, models, output_prefix,
                                                analysis_config.statistical_tests, None, analysis_config.contrasts,
                                                analysis_config.donor_pattern)
            plot_repertoire_statistics(results, output_prefix, score_column, collection_specification,
                                       score_description=description, x_label=f"Difference in {label}")
        else:
            write_repertoire_statistics({"repertoire_scores": repertoire_scores,
                                         "summary": summarise_repertoire_scores(repertoire_scores,
                                                                                analysis_config.statistical_tests)},
                                        output_prefix)

        null_source = f"{null_value:g}"
        null_scores = repertoire_scores.drop_duplicates("repertoire").assign(source=null_source, score=null_value)
        comparisons_df, differences_df = compare_paired(pd.concat([repertoire_scores, null_scores], ignore_index=True),
                                                        [(model, null_source) for model in models],
                                                        analysis_config.statistical_tests)
        comparisons_df.to_csv(f"{output_prefix}_vs_null.tsv", sep="\t", index=False)
        differences_df.to_csv(f"{output_prefix}_vs_null_differences.tsv", sep="\t", index=False)
        plot_paired_comparisons(comparisons_df, f"{output_prefix}_vs_null",
                                f"{description}: Models Compared with {null_source} (Train and Test Scored Alike) in "
                                f"{collection_specification} Repertoires", f"{label} − {null_source}", differences_df)
