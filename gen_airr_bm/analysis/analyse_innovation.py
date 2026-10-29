import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dataclasses import dataclass, field

from gen_airr_bm.analysis.analyse_innovation_umi import preprocess_gen_for_innovation_precision, \
    plot_innovation_scores_by_n_gen_novel, plot_innovation_precision_sensitivity
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_sequence_files, get_subset_index
from gen_airr_bm.utils.compairr_utils import run_compairr_existence
from gen_airr_bm.utils.plotting_utils import get_collection_specification_for_title, get_donor_symbols, wrap_title, \
    plot_repertoire_statistics
from gen_airr_bm.utils.statistics_utils import aggregate_repertoire_scores, check_one_repertoire_per_donor, \
    describe_error_bars, run_repertoire_statistics, summarise_repertoire_scores

# The innovation score of the unique clones scenario, the proportion of unique test sequences found in the generated
# sequences. The innovation precision is only reported for the clonally expanded scenario (analyse_innovation_umi).
METRIC_NAME = "innovation_sensitivity"
METRIC_DESCRIPTION = "Innovation Sensitivity"
SCORE_COLUMNS = ["metric", "model", "reference", "dataset", "subset", "score"]


@dataclass
class InnovationScores:
    """ Class to store precision and sensitivity scores for different models and datasets. """
    innovation_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["dataset", "model", "precision_innovation", "sensitivity_innovation", "n_gen_novel", "n_test_only"]
    ))
    scores: list = field(default_factory=list)


def run_innovation_analysis(analysis_config: AnalysisConfig) -> None:
    """ Runs precision sensitivity analysis on the generated and reference sequences.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
    Returns:
        None
    """
    print("Running innovation analysis")

    output_dir = analysis_config.analysis_output_dir
    compairr_output_dir = f"{output_dir}/compairr_output"

    for directory in [output_dir, compairr_output_dir]:
        os.makedirs(directory, exist_ok=True)

    compute_and_plot_innovation_scores(analysis_config, compairr_output_dir)


def compute_and_plot_innovation_scores(analysis_config: AnalysisConfig, compairr_output_dir: str) -> None:
    """ Compute precision and sensitivity scores and plot them. The scores of every generated subset are averaged
    within its repertoire before the models are summarised over repertoires (see statistics_utils). The models are also
    compared at the repertoire level: with each other and in the contrasts given in the config. These comparisons are
    descriptive, or include confidence intervals and tests if statistical_tests is set, which requires a donor_pattern
    and one repertoire per donor.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        compairr_output_dir (str): Directory to store CompAIRR output files.
    Returns:
        None
    """
    test_reference = 'test'

    scores = InnovationScores()

    for model in analysis_config.model_names:
        collect_innovation_scores(analysis_config, model, test_reference, compairr_output_dir, scores)

    scores_df = pd.DataFrame(scores.scores, columns=SCORE_COLUMNS)
    scores_df.to_csv(f"{analysis_config.analysis_output_dir}/innovation_repertoire_scores.tsv", sep="\t", index=False)
    if analysis_config.statistical_tests:
        check_one_repertoire_per_donor(aggregate_repertoire_scores(scores_df, "score", analysis_config.donor_pattern),
                                       analysis_config.donor_pattern)

    plot_innovation_scores_by_n_gen_novel(analysis_config, scores)
    plot_innovation_precision_sensitivity(analysis_config, scores)
    if analysis_config.receptor_type != "UMI":
        plot_innovation_sensitivity_by_model(analysis_config, scores_df)
    scores.innovation_df.to_csv(f"{analysis_config.analysis_output_dir}/innovation_scores.csv", index=False)
    run_innovation_statistics(analysis_config, scores_df)


def collect_innovation_scores(analysis_config: AnalysisConfig, model: str, test_reference: str,
                              compairr_output_dir: str,
                              scores: InnovationScores) -> None:
    """ Collect precision and sensitivity scores for a given model.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        model (str): Name of the model to analyze.
        test_reference (str): Reference dataset for testing.
        compairr_output_dir (str): Directory to store CompAIRR output files.
        scores (InnovationScores): Storage for precision and sensitivity scores.
    Returns:
        None
    """
    comparison_files_dir = get_sequence_files(analysis_config, model, test_reference)

    for ref_file, gen_files in comparison_files_dir.items():
        dataset_name = os.path.splitext(os.path.basename(ref_file))[0]
        for gen_file in gen_files:
            sensitivity = compute_compairr_overlap_ratio(analysis_config, ref_file, gen_file, compairr_output_dir,
                                                         model, scores)
            scores.scores.append({"metric": METRIC_NAME, "model": model, "reference": test_reference,
                                  "dataset": dataset_name, "subset": get_subset_index(gen_file), "score": sensitivity})


def compute_compairr_overlap_ratio(analysis_config: AnalysisConfig, ref_file: str, gen_file: str,
                                   compairr_output_dir: str, name: str, scores: InnovationScores) \
        -> float:
    """ Compute the overlap ratio between two sequence sets using CompAIRR for precision or sensitivity.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        search_for_file (str): Path to the file of sequences for which to search for existence in another sequence set.
        search_in_file (str): Path to the file to search for existence in.
        compairr_output_dir (str): Directory to store CompAIRR output files.
        name (str): Name of the model used for generation, or "upper_reference" for the upper reference.
        metric (str): Metric type, either "precision" or "sensitivity".
    Returns:
        float: The innovation sensitivity, the proportion of test sequences found in the generated sequences. The
        innovation precision is only stored in scores.innovation_df.
    """
    file_name = f"{os.path.splitext(os.path.basename(gen_file))[0]}_{name}_innovation"
    run_compairr_existence(compairr_output_dir, ref_file, gen_file, file_name,
                           allowed_mismatches=analysis_config.allowed_mismatches, indels=analysis_config.indels)
    compairr_result = pd.read_csv(f"{compairr_output_dir}/{file_name}_overlap.tsv", sep='\t',
                                  names=['sequence_id', 'overlap_count'], header=0)
    n_nonzero_rows = compairr_result[(compairr_result['overlap_count'] != 0)].shape[0]
    gen_file_df = pd.read_csv(gen_file, sep='\t')
    ref_file_df = pd.read_csv(ref_file, sep='\t')
    innovation_precision = n_nonzero_rows / len(gen_file_df)
    innovation_sensitivity = n_nonzero_rows / len(ref_file_df)

    dataset_name = os.path.splitext(os.path.basename(gen_file))[0]
    scores.innovation_df.loc[len(scores.innovation_df)] = [
        dataset_name, name, innovation_precision, innovation_sensitivity, len(gen_file_df),
        len(ref_file_df)
    ]

    return innovation_sensitivity


def plot_innovation_sensitivity_by_model(analysis_config: AnalysisConfig, scores_df: pd.DataFrame) -> None:
    """ Plot innovation sensitivity per model, with one dot per generated subset and a marker for the mean over
    repertoires, with the SD across repertoires as error bars. The subsets are first averaged within each repertoire
    for the mean and SD. If donors have several repertoires, the dots of each donor get their own marker symbol.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores_df (pd.DataFrame): One row per metric, model, dataset and generated subset, with the score in score.
    Returns:
        None
    """
    df = scores_df[scores_df["metric"] == METRIC_NAME]
    repertoire_scores = aggregate_repertoire_scores(df, "score", analysis_config.donor_pattern)
    summary = summarise_repertoire_scores(repertoire_scores, statistical_tests=False)
    donors = repertoire_scores.drop_duplicates("repertoire").set_index("repertoire")["donor"]
    df = df.assign(donor=df["dataset"].map(donors))
    donor_symbols = get_donor_symbols(donors)

    colors = px.colors.qualitative.Dark24
    model_names_sorted = sorted(analysis_config.model_names)
    color_map = {model: colors[i % len(colors)] for i, model in enumerate(model_names_sorted)}

    # Box traces with only their points shown, as in px.strip, which cannot set marker symbols
    fig = go.Figure()
    for model in model_names_sorted:
        for donor in sorted(donor_symbols) if donor_symbols else [None]:
            points = df[(df["model"] == model) & ((df["donor"] == donor) if donor is not None else True)]
            if points.empty:
                continue
            fig.add_trace(go.Box(
                x=[model] * len(points), y=points["score"], name=model, boxpoints="all", jitter=0.3, pointpos=0,
                fillcolor="rgba(255,255,255,0)", line=dict(color="rgba(255,255,255,0)"), hoveron="points",
                text=[f"{dataset}, subset {subset}" for dataset, subset in zip(points["dataset"], points["subset"])],
                hovertemplate="%{text}: %{y:.3g}<extra></extra>", showlegend=False,
                marker=dict(color=color_map[model], size=11, opacity=0.6,
                            symbol=donor_symbols[donor] if donor is not None else "circle")))
    # Legend entries for the donor symbols, in grey since the colours are the models
    for donor, symbol in sorted((donor_symbols or {}).items()):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Donor {donor}",
                                 marker=dict(color="grey", size=11, symbol=symbol)))

    # Mean per model, drawn as a horizontal tick over each column
    mean_per_model = summary.set_index("source").reindex(model_names_sorted)

    fig.add_trace(
        go.Scatter(
            x=mean_per_model.index,
            y=mean_per_model["mean"],
            error_y=dict(type="data", array=mean_per_model["sd"], visible=True, color="rgba(50,50,50,0.9)"),
            mode="markers",
            marker=dict(symbol="line-ew", size=26, line=dict(width=2.0, color="rgba(50,50,50,0.9)")),
            name="Mean",
            hovertemplate="Mean: %{y:.3g}<extra></extra>",
            showlegend=False,
        )
    )

    collection_specification = get_collection_specification_for_title(analysis_config.receptor_type,
                                                                     analysis_config.collection)
    fig.update_layout(
        title={'text': wrap_title(f"Innovation Scores for Generated {collection_specification} Repertoires", width=50),
               'font': {'size': 20},
               'y': 0.95,
               'yanchor': 'top'
               },
        margin=dict(t=130),
        template="plotly_white",
        colorway=px.colors.qualitative.Dark24,
        xaxis_title={'text': "Model", 'font': {'size': 20}},
        yaxis_title={'text': "Innovation Score", 'font': {'size': 20}},
        xaxis=dict(tickfont=dict(size=18), categoryorder="array", categoryarray=model_names_sorted),
        yaxis=dict(tickfont=dict(size=18)),
        boxmode="overlay",
        showlegend=donor_symbols is not None,
        legend=dict(tracegroupgap=0),
        # The figure grows with the donor legend, which is only shown if donors have several repertoires
        height=max(500, 300 + 30 * len(donor_symbols or {})),
    )

    fig.add_annotation(text=describe_error_bars(summary["n_repertoires"].max(), summary["n_donors"].max(),
                                                dots="generated subsets"),
                       xref="paper", yref="paper", x=0, y=1.02, showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=14))

    output_path = (
        f"{analysis_config.analysis_output_dir}/"
        "innovation_sensitivity_by_model.png"
    )

    fig.write_image(output_path, scale=3)


def run_innovation_statistics(analysis_config: AnalysisConfig, scores_df: pd.DataFrame) -> None:
    """ Runs the repertoire-level comparisons of the innovation sensitivity and plots the contrasts from the config.
    There is no train vs. test reference, since train and test contain different unique clones.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores_df (pd.DataFrame): One row per metric, model, dataset and generated subset, with the score in score.
    Returns:
        None
    """
    collection_specification = get_collection_specification_for_title(analysis_config.receptor_type,
                                                                     analysis_config.collection)
    output_prefix = f"{analysis_config.analysis_output_dir}/statistics/{METRIC_NAME}"
    repertoire_scores = aggregate_repertoire_scores(scores_df[scores_df["metric"] == METRIC_NAME], "score",
                                                    analysis_config.donor_pattern)
    results = run_repertoire_statistics(repertoire_scores, analysis_config.model_names, output_prefix,
                                        analysis_config.statistical_tests, None, analysis_config.contrasts,
                                        analysis_config.donor_pattern)
    plot_repertoire_statistics(results, output_prefix, METRIC_NAME, collection_specification,
                               score_description=METRIC_DESCRIPTION, x_label=f"Difference in {METRIC_DESCRIPTION}")
