import os
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gen_airr_bm.analysis.analyse_network import calculate_jsd, validate_references
from gen_airr_bm.constants.dataset_split import DatasetSplit
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.compairr_utils import deduplicate_single_dataset, run_compairr_cluster
from gen_airr_bm.utils.file_utils import get_reference_files, get_sequence_files, get_subset_index, is_up_to_date
from gen_airr_bm.utils.plotting_utils import (
    get_collection_specification_for_title,
    plot_grouped_avg_scores,
    plot_repertoire_statistics,
)
from gen_airr_bm.utils.statistics_utils import (aggregate_repertoire_scores, check_one_repertoire_per_donor,
                                               run_repertoire_statistics, summarise_scores_by_reference)

METRIC_NAME = "cluster_size"
SCORE_COLUMNS = ["metric", "model", "reference", "dataset", "subset", "jsd"]
SUMMARY_COLUMNS = ["source", "dataset", "subset", "n_sequences", "n_clusters", "clusters_per_sequence",
                   "singleton_share", "largest_cluster_share"]


def run_clonal_clusters_analysis(analysis_config: AnalysisConfig) -> None:
    """ Compares the clonal cluster structure of generated and reference repertoires. The unique CDR3 amino acid
    sequences of every train, test and generated set are clustered by single linkage, where two sequences are linked if
    they have the same length and differ by at most allowed_mismatches substitutions (CompAIRR cluster, genes ignored).
    This mimics the CDR3-similarity-based grouping of BCR sequences into clonal families, at the amino acid level.

    Each set is then described by the share of its sequences that fall in clusters of size 1, 2, 3-4, 5-8, ...
    (log2 bins), so that both singletons and large lineage-like clusters count. The JSD between these distributions
    compares each generated subset to the train and test sets, with the train vs. test JSD as reference. Summary
    statistics (number of clusters per sequence, share of singletons, share of the largest cluster) are written too.

    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names. Its
            allowed_mismatches is the clustering distance and must be at least 1.
    Returns:
        None
    """
    print("Running clonal clusters analysis")
    validate_references(analysis_config.reference_data)
    if analysis_config.allowed_mismatches < 1:
        raise ValueError("Clonal clusters analysis needs allowed_mismatches >= 1 as the clustering distance.")

    output_dir = analysis_config.analysis_output_dir
    helper_dir = f"{output_dir}/compairr_helper_files"
    compairr_output_dir = f"{output_dir}/compairr_output"
    for directory in [output_dir, helper_dir, compairr_output_dir]:
        os.makedirs(directory, exist_ok=True)

    distance = analysis_config.allowed_mismatches
    scores, summaries, bin_shares = [], [], []
    summarised_files = set()

    for reference in analysis_config.reference_data:
        for model_name in analysis_config.model_names:
            comparison_files = get_sequence_files(analysis_config, model_name, reference)
            for ref_file, gen_files in sorted(comparison_files.items()):
                dataset_name = os.path.splitext(os.path.basename(ref_file))[0]
                ref_bins = bin_cluster_sizes(get_cluster_sizes(ref_file, reference, helper_dir, compairr_output_dir,
                                                               distance))
                for gen_file in sorted(gen_files, key=get_subset_index):
                    subset = get_subset_index(gen_file)
                    gen_sizes = get_cluster_sizes(gen_file, model_name, helper_dir, compairr_output_dir, distance)
                    gen_bins = bin_cluster_sizes(gen_sizes)
                    scores.append({"metric": METRIC_NAME, "model": model_name, "reference": reference,
                                   "dataset": dataset_name, "subset": subset,
                                   "jsd": calculate_jsd(gen_bins, ref_bins, model_name, reference)[0]})
                    if gen_file not in summarised_files:
                        summarised_files.add(gen_file)
                        summaries.append(summarise_cluster_sizes(gen_sizes, model_name, dataset_name, subset))
                        bin_shares.extend(to_share_rows(gen_bins, model_name, dataset_name, subset))

    reference_scores = []
    for train_file, test_file in sorted(get_reference_files(analysis_config)):
        dataset_name = os.path.splitext(os.path.basename(train_file))[0]
        train_sizes = get_cluster_sizes(train_file, DatasetSplit.TRAIN.value, helper_dir, compairr_output_dir, distance)
        test_sizes = get_cluster_sizes(test_file, DatasetSplit.TEST.value, helper_dir, compairr_output_dir, distance)
        reference_scores.append({"repertoire": dataset_name,
                                 "score": calculate_jsd(bin_cluster_sizes(test_sizes), bin_cluster_sizes(train_sizes),
                                                        DatasetSplit.TEST.value, DatasetSplit.TRAIN.value)[0]})
        for split, sizes in [(DatasetSplit.TRAIN.value, train_sizes), (DatasetSplit.TEST.value, test_sizes)]:
            summaries.append(summarise_cluster_sizes(sizes, split, dataset_name, np.nan))
            bin_shares.extend(to_share_rows(bin_cluster_sizes(sizes), split, dataset_name, np.nan))

    scores_df = pd.DataFrame(scores, columns=SCORE_COLUMNS)
    reference_scores_df = pd.DataFrame(reference_scores, columns=["repertoire", "score"])
    summaries_df = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    bin_shares_df = pd.DataFrame(bin_shares, columns=["source", "dataset", "subset", "cluster_size_bin", "share"])

    scores_df.to_csv(f"{output_dir}/cluster_size_jsd_scores.tsv", sep="\t", index=False)
    reference_scores_df.to_csv(f"{output_dir}/cluster_size_jsd_train_vs_test.tsv", sep="\t", index=False)
    summaries_df.to_csv(f"{output_dir}/cluster_summary_statistics.tsv", sep="\t", index=False)
    bin_shares_df.to_csv(f"{output_dir}/cluster_size_bin_shares.tsv", sep="\t", index=False)

    if analysis_config.statistical_tests:
        check_one_repertoire_per_donor(aggregate_repertoire_scores(scores_df, "jsd", analysis_config.donor_pattern),
                                       analysis_config.donor_pattern)

    plot_cluster_size_jsd(analysis_config, scores_df, reference_scores_df)
    run_cluster_size_statistics(analysis_config, scores_df, reference_scores_df)
    plot_cluster_summary_statistics(analysis_config, summaries_df)
    plot_cluster_size_distributions(analysis_config, bin_shares_df)


def get_cluster_sizes(sequences_path: str, label: str, helper_dir: str, compairr_output_dir: str,
                      distance: int) -> pd.Series:
    """ Clusters the unique CDR3 amino acid sequences of one file with CompAIRR and returns the cluster sizes.
    The deduplicated file and the CompAIRR output of an earlier run are reused while they are newer than their input.
    Args:
        sequences_path (str): Path to the AIRR TSV file with the sequences (train, test or generated).
        label (str): Name of the set, e.g. train, test or the model name, used to name the output files.
        helper_dir (str): Directory for the deduplicated sequence files.
        compairr_output_dir (str): Directory for the CompAIRR output.
        distance (int): Maximum number of substitutions between linked sequences.
    Returns:
        pd.Series: One entry per cluster, with the number of unique sequences in it.
    """
    file_name = f"{os.path.splitext(os.path.basename(sequences_path))[0]}_{label}"
    unique_path = f"{helper_dir}/{file_name}_unique.tsv"
    if not is_up_to_date(unique_path, [sequences_path]):
        deduplicate_single_dataset(sequences_path, unique_path)

    output_identifier = f"{file_name}_d{distance}"
    results_path = f"{compairr_output_dir}/{output_identifier}.tsv"
    # run_compairr_cluster reuses any existing output, so remove output computed from older sequences first
    if os.path.exists(results_path) and not is_up_to_date(results_path, [unique_path]):
        os.remove(results_path)
    results_path = run_compairr_cluster(compairr_output_dir=compairr_output_dir, sequences_path=unique_path,
                                        output_file_identifier=output_identifier, distance=distance)

    clusters = pd.read_csv(results_path, sep="\t", usecols=["#cluster_no"])
    return clusters.groupby("#cluster_no").size()


def cluster_size_bin(size: int) -> int:
    """ Returns the lower bound of the log2 bin of a cluster size: 1, 2, 3 (3-4), 5 (5-8), 9 (9-16), ... """
    return 1 if size <= 1 else 2 ** int(np.ceil(np.log2(size)) - 1) + 1


def cluster_size_bin_label(lower: int) -> str:
    """ Returns the label of the log2 bin that starts at lower, e.g. "1", "2", "3-4", "5-8". """
    return str(lower) if lower <= 2 else f"{lower}-{2 * (lower - 1)}"


def bin_cluster_sizes(cluster_sizes: pd.Series) -> pd.Series:
    """ Counts the sequences in clusters of each log2 size bin.
    Args:
        cluster_sizes (pd.Series): One entry per cluster, with its size.
    Returns:
        pd.Series: Named count and indexed by the lower bound of the bin, with the number of sequences in clusters of
        that bin. calculate_jsd normalises the counts.
    """
    bins = cluster_sizes.map(cluster_size_bin)
    return cluster_sizes.groupby(bins).sum().rename("count").rename_axis("cluster_size_bin")


def summarise_cluster_sizes(cluster_sizes: pd.Series, source: str, dataset: str, subset) -> dict:
    """ Summarises the cluster sizes of one set of sequences.
    Args:
        cluster_sizes (pd.Series): One entry per cluster, with its size.
        source (str): Name of the set, e.g. train, test or the model name.
        dataset (str): Name of the repertoire.
        subset: Index of the generated subset, or nan for train and test.
    Returns:
        dict: The summary statistics, with the keys of SUMMARY_COLUMNS.
    """
    n_sequences = int(cluster_sizes.sum())
    n_clusters = int(len(cluster_sizes))
    return {"source": source, "dataset": dataset, "subset": subset, "n_sequences": n_sequences,
            "n_clusters": n_clusters,
            "clusters_per_sequence": n_clusters / n_sequences if n_sequences else np.nan,
            "singleton_share": int((cluster_sizes == 1).sum()) / n_sequences if n_sequences else np.nan,
            "largest_cluster_share": int(cluster_sizes.max()) / n_sequences if n_sequences else np.nan}


def to_share_rows(binned_sizes: pd.Series, source: str, dataset: str, subset) -> list[dict]:
    """ Converts binned sequence counts into rows with the share of sequences per bin. """
    total = binned_sizes.sum()
    return [{"source": source, "dataset": dataset, "subset": subset, "cluster_size_bin": int(lower),
             "share": count / total} for lower, count in binned_sizes.items()]


def plot_cluster_size_jsd(analysis_config: AnalysisConfig, scores_df: pd.DataFrame,
                          reference_scores: pd.DataFrame) -> None:
    """ Plots the mean JSD of the cluster size distributions per model and reference, with one observation per
    repertoire, and the mean train vs. test JSD as reference line. """
    mean_scores, std_scores, error_bar_caption = \
        summarise_scores_by_reference(scores_df, METRIC_NAME, analysis_config.donor_pattern)
    plot_grouped_avg_scores(
        analysis_config=analysis_config,
        mean_scores_by_ref=mean_scores,
        std_scores_by_ref=std_scores,
        distribution_type="cluster size",
        file_name="all_datasets_cluster_size",
        scoring_method="JSD",
        reference_score=float(reference_scores["score"].mean()) if len(reference_scores) else None,
        error_bar_caption=error_bar_caption
    )


def run_cluster_size_statistics(analysis_config: AnalysisConfig, scores_df: pd.DataFrame,
                                reference_scores: pd.DataFrame) -> None:
    """ Compares the models at the repertoire level against the test sequences, as in the network analysis: with the
    train vs. test reference and the contrasts from the config. Descriptive unless statistical_tests is set. """
    if DatasetSplit.TEST.value not in analysis_config.reference_data:
        print("Skipping cluster size statistics, since they are computed against the test sequences.")
        return

    test_scores = scores_df[scores_df["reference"] == DatasetSplit.TEST.value]
    output_prefix = f"{analysis_config.analysis_output_dir}/statistics/{METRIC_NAME}"
    repertoire_scores = aggregate_repertoire_scores(test_scores, "jsd", analysis_config.donor_pattern)
    results = run_repertoire_statistics(repertoire_scores, analysis_config.model_names, output_prefix,
                                        analysis_config.statistical_tests, reference_scores, analysis_config.contrasts,
                                        analysis_config.donor_pattern)
    plot_repertoire_statistics(results, output_prefix, "cluster size",
                               get_collection_specification_for_title(analysis_config.receptor_type,
                                                                      analysis_config.collection))


def summarise_by_repertoire(df: pd.DataFrame, value_columns: list, group_columns: list) -> pd.DataFrame:
    """ Averages the values over the generated subsets of each repertoire, and then over repertoires, so that the
    repertoires are the observations. Returns the mean and the standard deviation over repertoires. """
    per_repertoire = df.groupby(group_columns + ["dataset"], as_index=False)[value_columns].mean()
    return per_repertoire.groupby(group_columns)[value_columns].agg(["mean", "std"])


def ordered_sources(analysis_config: AnalysisConfig) -> list[str]:
    """ Returns the train and test sets followed by the models, in the order of the config. """
    return [DatasetSplit.TRAIN.value, DatasetSplit.TEST.value] + list(analysis_config.model_names)


def plot_cluster_summary_statistics(analysis_config: AnalysisConfig, summaries_df: pd.DataFrame) -> None:
    """ Plots the summary statistics of the clusters per model, train and test, as the mean over repertoires with the
    standard deviation over repertoires as error bars. """
    statistics = {"clusters_per_sequence": "Clusters per Sequence",
                  "singleton_share": "Share of Singleton Sequences",
                  "largest_cluster_share": "Share of Sequences in the Largest Cluster"}
    summary = summarise_by_repertoire(summaries_df, list(statistics), ["source"])
    summary.to_csv(f"{analysis_config.analysis_output_dir}/cluster_summary_statistics_by_source.tsv", sep="\t")
    sources = [s for s in ordered_sources(analysis_config) if s in summary.index]

    fig = make_subplots(rows=1, cols=len(statistics), subplot_titles=list(statistics.values()))
    for col, statistic in enumerate(statistics, start=1):
        colors = ["#4d4d4d" if s in (DatasetSplit.TRAIN.value, DatasetSplit.TEST.value) else "#4c78a8"
                  for s in sources]
        fig.add_trace(go.Bar(x=sources, y=summary.loc[sources, (statistic, "mean")],
                             error_y=dict(type="data", array=summary.loc[sources, (statistic, "std")].fillna(0)),
                             marker_color=colors, showlegend=False), row=1, col=col)
        fig.update_xaxes(tickangle=-45, row=1, col=col)
    collection = get_collection_specification_for_title(analysis_config.receptor_type, analysis_config.collection)
    fig.update_layout(title=f"Clonal Clusters (CDR3 Distance ≤ {analysis_config.allowed_mismatches}) of "
                            f"{collection} Repertoires<br><sup>Mean over repertoires, error bars show the standard "
                            f"deviation over repertoires</sup>",
                      template="plotly_white", height=550, width=1500)
    fig.write_image(f"{analysis_config.analysis_output_dir}/cluster_summary_statistics.png")


def plot_cluster_size_distributions(analysis_config: AnalysisConfig, bin_shares_df: pd.DataFrame) -> None:
    """ Plots, for each model, the share of sequences per cluster size bin next to the test and train sets, as the
    mean over repertoires. Bins a set has no clusters in count as a share of 0 in the mean, and a mean of 0 is not
    drawn on the log axis. """
    sources = [s for s in ordered_sources(analysis_config) if s in set(bin_shares_df["source"])]
    bins = sorted(bin_shares_df["cluster_size_bin"].unique())
    full_index = pd.MultiIndex.from_frame(
        bin_shares_df[["source", "dataset", "subset"]].drop_duplicates()).to_frame(index=False)
    complete = (full_index.merge(pd.DataFrame({"cluster_size_bin": bins}), how="cross")
                .merge(bin_shares_df, how="left", on=["source", "dataset", "subset", "cluster_size_bin"])
                .fillna({"share": 0.0}))
    mean_shares = summarise_by_repertoire(complete, ["share"], ["source", "cluster_size_bin"])[("share", "mean")]
    labels = [cluster_size_bin_label(b) for b in bins]

    models = [s for s in sources if s not in (DatasetSplit.TRAIN.value, DatasetSplit.TEST.value)]
    n_cols = 4
    n_rows = int(np.ceil(len(models) / n_cols)) or 1
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=models, shared_yaxes=True)
    for i, model in enumerate(models):
        row, col = i // n_cols + 1, i % n_cols + 1
        for source, style in [(DatasetSplit.TEST.value, dict(color="black", dash="dash")),
                              (DatasetSplit.TRAIN.value, dict(color="grey", dash="dot")),
                              (model, dict(color="#4c78a8"))]:
            if source not in mean_shares.index.get_level_values("source"):
                continue
            # A share of 0 has no place on the log axis, so it is left as a gap
            shares = [mean_shares.get((source, b), 0.0) for b in bins]
            fig.add_trace(go.Scatter(x=labels, y=[s if s > 0 else None for s in shares],
                                     mode="lines+markers", line=style, name=source if source != model else "generated",
                                     legendgroup=source if source != model else "generated", showlegend=i == 0),
                          row=row, col=col)
    fig.update_yaxes(type="log", title_text="Share of sequences", col=1)
    fig.update_xaxes(title_text="Cluster size")
    collection = get_collection_specification_for_title(analysis_config.receptor_type, analysis_config.collection)
    fig.update_layout(title=f"Cluster Size Distributions (CDR3 Distance ≤ {analysis_config.allowed_mismatches}) of "
                            f"{collection} Repertoires<br><sup>Mean share of sequences per cluster size over "
                            f"repertoires</sup>",
                      template="plotly_white", height=320 * n_rows + 120, width=1500)
    fig.write_image(f"{analysis_config.analysis_output_dir}/cluster_size_distributions.png")
