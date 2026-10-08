import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial.distance import jensenshannon

from gen_airr_bm.constants.dataset_split import DatasetSplit
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_generated_sequences_dir_name, get_reference_files, get_sequence_files
from gen_airr_bm.utils.plotting_utils import (get_collection_specification_for_title, plot_grouped_avg_scores,
                                             plot_paired_comparisons, title_case, wrap_title)
from gen_airr_bm.utils.statistics_utils import (aggregate_repertoire_scores, check_one_repertoire_per_donor,
                                               describe_error_bars, run_repertoire_statistics,
                                               summarise_repertoire_scores)

GENE_CALL_COLUMNS = ["v_call", "j_call"]

# Each metric is defined by the gene calls it compares and the resolution they are compared at. Gene level keeps the
# gene within the family (TRBV6-1), family level collapses it (TRBV6). Family level is reported in addition to gene
# level because the experimental data contains calls that were only resolved to the family (TRBV20), which at gene
# level can never match the resolved call a model generates (TRBV20-1).
GENE_USAGE_METRICS = {
    "v gene": (["v_call"], "gene"),
    "j gene": (["j_call"], "gene"),
    "vj pairing": (["v_call", "j_call"], "gene"),
    "v family": (["v_call"], "family"),
    "j family": (["j_call"], "family"),
}

SUBSET_INDEX_PATTERN = re.compile(r"_(\d+)\.tsv$")
GENE_ABBREVIATIONS = {"v", "j", "vj"}


def run_gene_usage_analysis(analysis_config: AnalysisConfig) -> None:
    """ Runs the germline gene usage analysis: compares V gene, J gene and V-J pairing frequencies of the novel
    generated sequences with those of the train and test sequences of the same repertoire. Models that do not
    generate gene calls are skipped and reported in skipped_models.txt. Against the test sequences, the models are
    also compared at the repertoire level (see statistics_utils): with the train vs. test reference, with each other,
    and in the contrasts given in the config. These comparisons are descriptive, or include confidence intervals and
    tests if statistical_tests is set, which requires one repertoire per donor.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        None
    """
    print("Running gene usage analysis...")
    os.makedirs(analysis_config.analysis_output_dir, exist_ok=True)

    if isinstance(analysis_config.reference_data, str):
        analysis_config.reference_data = [analysis_config.reference_data]

    models_with_gene_calls, models_without_gene_calls = split_models_by_gene_calls(analysis_config)
    report_skipped_models(analysis_config, models_without_gene_calls)
    if not models_with_gene_calls:
        print("None of the models generate gene calls, skipping gene usage analysis.")
        return

    distributions_cache = {}
    scores_df = compute_gene_usage_scores(analysis_config, models_with_gene_calls, distributions_cache)
    scores_df.to_csv(f"{analysis_config.analysis_output_dir}/gene_usage_jsd_scores.tsv", sep="\t", index=False)
    if analysis_config.statistical_tests:
        check_one_repertoire_per_donor(aggregate_repertoire_scores(scores_df, "jsd", analysis_config.donor_pattern))

    frequencies_df = compute_gene_usage_frequencies(analysis_config, models_with_gene_calls, distributions_cache)
    frequencies_df.to_csv(f"{analysis_config.analysis_output_dir}/gene_usage_frequencies.tsv", sep="\t", index=False)

    for metric_name in GENE_USAGE_METRICS:
        mean_scores_by_ref, std_scores_by_ref, error_bars_by_ref, repertoire_scores_by_ref, error_bar_caption = \
            summarise_scores_by_reference(analysis_config, scores_df, metric_name)
        reference_scores = compute_train_test_reference_scores(analysis_config, metric_name, distributions_cache)
        reference_score = float(reference_scores["score"].mean()) if reference_scores is not None else None
        plot_grouped_avg_scores(analysis_config, mean_scores_by_ref, std_scores_by_ref,
                               f"gene_usage_{metric_name.replace(' ', '_')}_grouped", get_metric_label(metric_name),
                               "JSD", reference_score, error_bars_by_ref, error_bar_caption, repertoire_scores_by_ref)
        plot_gene_usage_frequencies(analysis_config, frequencies_df, metric_name)
        run_gene_usage_statistics(analysis_config, scores_df, metric_name, models_with_gene_calls, reference_scores)


def get_metric_label(metric_name: str) -> str:
    """ Returns the metric name as used in plot titles, with the gene abbreviations capitalised, for example
    "VJ Pairing" for "vj pairing".
    Args:
        metric_name (str): The metric name as used in GENE_USAGE_METRICS.
    Returns:
        str: The title-cased metric label.
    """
    return title_case(" ".join(word.upper() if word in GENE_ABBREVIATIONS else word for word in metric_name.split()))


def split_models_by_gene_calls(analysis_config: AnalysisConfig) -> tuple[list, list]:
    """ Splits the models into those that generate gene calls and those that do not.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        tuple[list, list]: Models with gene calls and models without gene calls, both in config order.
    """
    models_with_gene_calls, models_without_gene_calls = [], []
    for model in analysis_config.model_names:
        if model_generates_gene_calls(analysis_config, model):
            models_with_gene_calls.append(model)
        else:
            models_without_gene_calls.append(model)
    return models_with_gene_calls, models_without_gene_calls


def model_generates_gene_calls(analysis_config: AnalysisConfig, model: str) -> bool:
    """ Checks whether a model generates gene calls, based on the first of its generated files.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        model (str): The name of the generative model to check.
    Returns:
        bool: True if the generated sequences come with V or J gene calls.
    """
    gen_dir = (f"{analysis_config.root_output_dir}/"
               f"{get_generated_sequences_dir_name(analysis_config)}/{model}")
    gen_files = sorted(os.listdir(gen_dir))
    if not gen_files:
        return False
    gene_calls = read_gene_calls(os.path.join(gen_dir, gen_files[0]))
    return bool(len(gene_calls))


def report_skipped_models(analysis_config: AnalysisConfig, models_without_gene_calls: list) -> None:
    """ Writes the models that were skipped because they do not generate gene calls to a file.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        models_without_gene_calls (list): Names of the models without gene calls.
    Returns:
        None
    """
    if not models_without_gene_calls:
        return
    print(f"Skipping models without generated gene calls: {', '.join(models_without_gene_calls)}")
    with open(f"{analysis_config.analysis_output_dir}/skipped_models.txt", "w") as file:
        file.write("Models skipped because they do not generate V or J gene calls:\n")
        file.writelines(f"{model}\n" for model in models_without_gene_calls)


def read_gene_calls(file_path: str) -> pd.DataFrame:
    """ Reads the V and J gene calls from a sequence file and normalises them to IMGT names without alleles.
    Sequences without both a V and a J call are dropped.
    Args:
        file_path (str): Path to the sequence file in tsv format.
    Returns:
        pd.DataFrame: Dataframe with the normalised v_call and j_call columns.
    """
    available_columns = pd.read_csv(file_path, sep="\t", nrows=0).columns
    columns_to_read = [column for column in GENE_CALL_COLUMNS if column in available_columns]
    if len(columns_to_read) < len(GENE_CALL_COLUMNS):
        return pd.DataFrame(columns=GENE_CALL_COLUMNS)

    gene_calls = pd.read_csv(file_path, sep="\t", usecols=columns_to_read, dtype=str)
    for column in GENE_CALL_COLUMNS:
        gene_calls[column] = gene_calls[column].map(normalise_gene_call)
    return gene_calls.dropna(subset=GENE_CALL_COLUMNS)


def normalise_gene_call(gene_call) -> str | None:
    """ Normalises a gene call to an IMGT gene name without the allele, for example TRBV5-5*01 to TRBV5-5. Dropping
    the allele is what makes the calls comparable: the experimental data keeps the allele, while the models generate
    gene names without it. Calls with several genes are reduced to the first one.
    Args:
        gene_call: The gene call as written in the sequence file.
    Returns:
        str | None: The normalised gene name, or None if there is no call.
    """
    if not isinstance(gene_call, str) or not gene_call.strip():
        return None
    return gene_call.split(",")[0].strip().split("*")[0] or None


def get_gene_family(gene_name: str) -> str:
    """ Returns the gene family of an IMGT gene name, for example TRBV6 for TRBV6-1 and TRBV20 for TRBV20/OR9-2.
    Args:
        gene_name (str): The IMGT gene name without the allele.
    Returns:
        str: The gene family.
    """
    return re.split(r"[-/]", gene_name)[0]


def compute_usage_distribution(gene_calls: pd.DataFrame, columns: list, resolution: str) -> dict:
    """ Computes the usage frequencies of the given gene calls. Several columns are combined into a pairing, for
    example TRBV6-1|TRBJ1-5.
    Args:
        gene_calls (pd.DataFrame): Dataframe with the normalised gene calls.
        columns (list): The gene call columns to use.
        resolution (str): "gene" to keep the gene within the family, "family" to collapse it.
    Returns:
        dict: Mapping from gene (or gene pairing) to its frequency.
    """
    if not len(gene_calls):
        return {}
    genes = [gene_calls[column] if resolution == "gene" else gene_calls[column].map(get_gene_family)
             for column in columns]
    counts = Counter("|".join(call) for call in zip(*genes))
    total = sum(counts.values())
    return {gene: count / total for gene, count in counts.items()} if total > 0 else {}


def get_usage_distributions(file_path: str, distributions_cache: dict) -> dict:
    """ Computes the usage distributions of all metrics for a sequence file, reading each file only once.
    Args:
        file_path (str): Path to the sequence file in tsv format.
        distributions_cache (dict): Cache mapping file paths to their usage distributions.
    Returns:
        dict: Mapping from metric name to the usage distribution.
    """
    if file_path not in distributions_cache:
        gene_calls = read_gene_calls(file_path)
        distributions_cache[file_path] = {
            metric_name: compute_usage_distribution(gene_calls, columns, resolution)
            for metric_name, (columns, resolution) in GENE_USAGE_METRICS.items()
        }
    return distributions_cache[file_path]


def compute_jsd(distribution1: dict, distribution2: dict, smooth: float = 1e-10) -> float:
    """ Computes the Jensen-Shannon divergence between two usage distributions.
    Args:
        distribution1 (dict): First usage distribution.
        distribution2 (dict): Second usage distribution.
        smooth (float): Value used for genes that are missing from one of the distributions.
    Returns:
        float: The Jensen-Shannon divergence, or nan if one of the distributions is empty.
    """
    if not distribution1 or not distribution2:
        return np.nan
    genes = set(distribution1) | set(distribution2)
    p = np.array([distribution1.get(gene, smooth) for gene in genes])
    q = np.array([distribution2.get(gene, smooth) for gene in genes])
    return jensenshannon(p, q, base=2) ** 2


def compute_gene_usage_scores(analysis_config: AnalysisConfig, models: list, distributions_cache: dict) -> pd.DataFrame:
    """ Computes the gene usage divergence between every generated subset and every reference dataset.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        models (list): Names of the models to analyse.
        distributions_cache (dict): Cache mapping file paths to their usage distributions.
    Returns:
        pd.DataFrame: One row per metric, model, reference, dataset and generated subset.
    """
    scores = []
    for model in models:
        for reference in analysis_config.reference_data:
            comparison_files = get_sequence_files(analysis_config, model, reference)
            for ref_file, gen_files in comparison_files.items():
                dataset = os.path.splitext(os.path.basename(ref_file))[0]
                ref_distributions = get_usage_distributions(ref_file, distributions_cache)
                for gen_file in sorted(gen_files):
                    gen_distributions = get_usage_distributions(gen_file, distributions_cache)
                    subset = SUBSET_INDEX_PATTERN.search(os.path.basename(gen_file))
                    for metric_name in GENE_USAGE_METRICS:
                        scores.append({"metric": metric_name,
                                       "model": model,
                                       "reference": reference,
                                       "dataset": dataset,
                                       "subset": int(subset.group(1)) if subset else np.nan,
                                       "jsd": compute_jsd(gen_distributions[metric_name],
                                                          ref_distributions[metric_name])})
    return pd.DataFrame(scores)


def summarise_scores_by_reference(analysis_config: AnalysisConfig, scores_df: pd.DataFrame,
                                  metric_name: str) -> tuple[dict, dict, dict, dict, str]:
    """ Averages the divergence scores of a metric over the generated subsets of each repertoire, and summarises them
    over repertoires, so that the repertoires and not the generated subsets are the observations.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including the statistics mode and the donor
            pattern.
        scores_df (pd.DataFrame): Scores as returned by compute_gene_usage_scores.
        metric_name (str): The metric to summarise.
    Returns:
        tuple[dict, dict, dict, dict, str]: As {reference: {model: value}}: the mean and the standard deviation over
        repertoires, the error bars (the confidence interval with statistical tests, otherwise the mean plus and
        minus the standard deviation), and the (donor, score) pairs of the repertoires; and the error bar caption.
    """
    mean_scores_by_ref, std_scores_by_ref, error_bars_by_ref, repertoire_scores_by_ref = (
        defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict))
    n_repertoires, n_donors = 0, 0
    metric_scores = scores_df[scores_df["metric"] == metric_name]
    for reference, reference_scores in metric_scores.groupby("reference"):
        repertoire_scores = aggregate_repertoire_scores(reference_scores, "jsd", analysis_config.donor_pattern)
        summary = summarise_repertoire_scores(repertoire_scores, analysis_config.statistical_tests)
        for row in summary.itertuples():
            mean_scores_by_ref[reference][row.source] = row.mean
            std_scores_by_ref[reference][row.source] = row.sd
            error_bars_by_ref[reference][row.source] = ((row.ci_low, row.ci_high) if analysis_config.statistical_tests
                                                        else (row.mean - row.sd, row.mean + row.sd))
        n_repertoires = max(n_repertoires, summary["n_repertoires"].max())
        n_donors = max(n_donors, summary["n_donors"].max())
        for model, model_scores in repertoire_scores.dropna(subset=["score"]).groupby("source"):
            repertoire_scores_by_ref[reference][model] = list(zip(model_scores["donor"], model_scores["score"]))
    error_bar_caption = describe_error_bars(analysis_config.statistical_tests, n_repertoires, n_donors)
    return (dict(mean_scores_by_ref), dict(std_scores_by_ref), dict(error_bars_by_ref), dict(repertoire_scores_by_ref),
            error_bar_caption)


def compute_train_test_reference_scores(analysis_config: AnalysisConfig, metric_name: str,
                                        distributions_cache: dict) -> pd.DataFrame | None:
    """ Computes the gene usage divergence between the train and test sequences of each repertoire, as a reference
    for how similar two samples of the same repertoire are.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        metric_name (str): The metric to compute the reference scores for.
        distributions_cache (dict): Cache mapping file paths to their usage distributions.
    Returns:
        pd.DataFrame | None: Columns repertoire and score, or None if train and test are not both used.
    """
    if not (DatasetSplit.TRAIN.value in analysis_config.reference_data
            and DatasetSplit.TEST.value in analysis_config.reference_data):
        return None

    rows = [{"repertoire": os.path.splitext(os.path.basename(train_file))[0],
             "score": compute_jsd(get_usage_distributions(test_file, distributions_cache)[metric_name],
                                  get_usage_distributions(train_file, distributions_cache)[metric_name])}
            for train_file, test_file in sorted(get_reference_files(analysis_config))]
    return pd.DataFrame(rows, columns=["repertoire", "score"])


def run_gene_usage_statistics(analysis_config: AnalysisConfig, scores_df: pd.DataFrame, metric_name: str,
                              models: list, reference_scores: pd.DataFrame | None) -> None:
    """ Runs the repertoire-level comparisons of a metric against the test sequences and plots the comparisons with
    the train vs. test reference and the contrasts from the config. The train sequences are left out, since the
    models were fitted on them. With statistical tests, each metric is its own multiple testing family.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores_df (pd.DataFrame): Scores as returned by compute_gene_usage_scores.
        metric_name (str): The metric to test.
        models (list): The models to compare.
        reference_scores (pd.DataFrame | None): Train vs. test scores per repertoire.
    Returns:
        None
    """
    if DatasetSplit.TEST.value not in analysis_config.reference_data:
        print("Skipping gene usage statistics, since they are computed against the test sequences.")
        return

    test_scores = scores_df[(scores_df["metric"] == metric_name)
                            & (scores_df["reference"] == DatasetSplit.TEST.value)]
    metric_file_name = metric_name.replace(" ", "_")
    output_prefix = f"{analysis_config.analysis_output_dir}/statistics/gene_usage_{metric_file_name}"
    repertoire_scores = aggregate_repertoire_scores(test_scores, "jsd", analysis_config.donor_pattern)
    results = run_repertoire_statistics(repertoire_scores, models, output_prefix, analysis_config.statistical_tests,
                                        reference_scores, analysis_config.contrasts, analysis_config.donor_pattern)

    collection_specification = get_collection_specification_for_title(analysis_config.receptor_type,
                                                                     analysis_config.collection)
    metric_label = get_metric_label(metric_name)
    if "vs_train_test" in results:
        plot_paired_comparisons(results["vs_train_test"], f"{output_prefix}_vs_train_test",
                                f"{metric_label} JSD to Test: Models Compared with Train vs. Test in "
                                f"{collection_specification} Repertoires", "Difference in JSD",
                                results["vs_train_test_differences"])
    if "contrasts" in results:
        plot_paired_comparisons(results["contrasts"], f"{output_prefix}_contrasts",
                                f"{metric_label} JSD to Test: Model Contrasts in {collection_specification} "
                                f"Repertoires", "Difference in JSD", results["contrasts_differences"])


def compute_gene_usage_frequencies(analysis_config: AnalysisConfig, models: list,
                                   distributions_cache: dict) -> pd.DataFrame:
    """ Computes the gene usage frequencies of the generated and reference sequences, averaged over the generated
    subsets of each repertoire and over repertoires.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        models (list): Names of the models to analyse.
        distributions_cache (dict): Cache mapping file paths to their usage distributions.
    Returns:
        pd.DataFrame: One row per metric, source, dataset and gene, with the mean frequency.
    """
    frequencies = []
    for model in models:
        gen_dir = (f"{analysis_config.root_output_dir}/"
                   f"{get_generated_sequences_dir_name(analysis_config)}/{model}")
        gen_files = [os.path.join(gen_dir, file) for file in sorted(os.listdir(gen_dir))]
        frequencies.extend(summarise_frequencies(gen_files, model, distributions_cache, is_generated=True))

    for reference in analysis_config.reference_data:
        ref_dir = f"{analysis_config.root_output_dir}/{reference}_compairr_sequences"
        ref_files = [os.path.join(ref_dir, file) for file in sorted(os.listdir(ref_dir))]
        frequencies.extend(summarise_frequencies(ref_files, reference, distributions_cache, is_generated=False))

    return pd.DataFrame(frequencies)


def summarise_frequencies(files: list, source: str, distributions_cache: dict, is_generated: bool) -> list:
    """ Averages the usage frequencies of the given files per repertoire. Files of generated sequences are grouped by
    repertoire over their subsets, reference files are one per repertoire.
    Args:
        files (list): Paths to the sequence files.
        source (str): Label of the data source, either a model name or a reference name.
        distributions_cache (dict): Cache mapping file paths to their usage distributions.
        is_generated (bool): Whether the files are generated subsets, whose names end in a subset index.
    Returns:
        list: Rows with metric, source, dataset, gene and mean frequency.
    """
    distributions_by_dataset = defaultdict(lambda: defaultdict(list))
    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        dataset = file_name.rsplit("_", 1)[0] if is_generated else file_name
        distributions = get_usage_distributions(file, distributions_cache)
        for metric_name, distribution in distributions.items():
            distributions_by_dataset[dataset][metric_name].append(distribution)

    frequencies = []
    for dataset, distributions_by_metric in distributions_by_dataset.items():
        for metric_name, distributions in distributions_by_metric.items():
            genes = {gene for distribution in distributions for gene in distribution}
            for gene in genes:
                frequencies.append({"metric": metric_name,
                                    "source": source,
                                    "dataset": dataset,
                                    "gene": gene,
                                    "frequency": np.mean([distribution.get(gene, 0.0)
                                                          for distribution in distributions])})
    return frequencies


def get_gene_usage_plotting_data(analysis_config: AnalysisConfig, frequencies_df: pd.DataFrame, metric_name: str,
                                 reference: str) -> pd.DataFrame:
    """ Pairs the generated frequency of every gene with its reference frequency, averaged over repertoires. Each
    model gets a row for every gene that it or the reference uses, so genes the model never generates are shown at a
    generated frequency of 0 and genes missing from the reference at a reference frequency of 0.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        frequencies_df (pd.DataFrame): Frequencies as returned by compute_gene_usage_frequencies.
        metric_name (str): The metric to plot.
        reference (str): The reference dataset to compare against, e.g. "test".
    Returns:
        pd.DataFrame: Columns source, gene, frequency_generated and frequency_reference.
    """
    metric_frequencies = frequencies_df[frequencies_df["metric"] == metric_name]
    mean_frequencies = metric_frequencies.groupby(["source", "gene"])["frequency"].mean().reset_index()
    reference_frequencies = mean_frequencies[mean_frequencies["source"] == reference][["gene", "frequency"]]

    model_dfs = []
    for model in analysis_config.model_names:
        model_frequencies = mean_frequencies[mean_frequencies["source"] == model][["gene", "frequency"]]
        if model_frequencies.empty:
            continue
        model_df = model_frequencies.merge(reference_frequencies, on="gene", how="outer",
                                           suffixes=("_generated", "_reference")).fillna(0.0)
        model_df.insert(0, "source", model)
        model_dfs.append(model_df)

    if not model_dfs:
        return pd.DataFrame(columns=["source", "gene", "frequency_generated", "frequency_reference"])
    return pd.concat(model_dfs, ignore_index=True)


def plot_gene_usage_frequencies(analysis_config: AnalysisConfig, frequencies_df: pd.DataFrame,
                                metric_name: str) -> None:
    """ Plots the generated gene usage frequencies against the test (or, if test is not used, the train) frequencies,
    averaged over repertoires.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        frequencies_df (pd.DataFrame): Frequencies as returned by compute_gene_usage_frequencies.
        metric_name (str): The metric to plot.
    Returns:
        None
    """
    reference = (DatasetSplit.TEST.value if DatasetSplit.TEST.value in analysis_config.reference_data
                 else analysis_config.reference_data[0])
    plotting_df = get_gene_usage_plotting_data(analysis_config, frequencies_df, metric_name, reference)
    if plotting_df.empty:
        return

    output_path = (f"{analysis_config.analysis_output_dir}/"
                   f"gene_usage_frequencies_{metric_name.replace(' ', '_')}")
    collection_specification = get_collection_specification_for_title(analysis_config.receptor_type,
                                                                     analysis_config.collection)
    title = (f"{get_metric_label(metric_name)} Usage in Generated vs. {reference.capitalize()} "
             f"{collection_specification} Repertoires")

    fig = px.scatter(plotting_df, x="frequency_reference", y="frequency_generated", color="source",
                     hover_data=["gene"])
    max_frequency = max(plotting_df["frequency_reference"].max(), plotting_df["frequency_generated"].max())
    fig.add_shape(type="line", x0=0, y0=0, x1=max_frequency, y1=max_frequency,
                  line=dict(color="black", dash="dash"))
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(legend_title_text="Model",
                      title={'text': wrap_title(title, width=50), 'font': {'size': 20}, 'y': 0.93,
                             'yanchor': 'top'},
                      margin=dict(t=100),
                      xaxis=dict(title=dict(text=f"{reference.capitalize()} Frequency", font=dict(size=20)),
                                 tickfont=dict(size=18)),
                      yaxis=dict(title=dict(text="Generated Frequency", font=dict(size=20)),
                                 tickfont=dict(size=18)),
                      template="plotly_white",
                      colorway=px.colors.qualitative.Safe,
                      legend=dict(font=dict(size=16)))
    fig.write_image(output_path + ".png", scale=3)
    print(f"Plot saved as png at: {output_path}.png")
