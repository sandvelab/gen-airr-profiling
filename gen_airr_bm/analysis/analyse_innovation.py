import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dataclasses import dataclass, field

from gen_airr_bm.analysis.analyse_innovation_umi import preprocess_gen_for_innovation_precision, \
    plot_innovation_scores_by_n_gen_novel, plot_innovation_precision_sensitivity
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.file_utils import get_sequence_files
from gen_airr_bm.utils.compairr_utils import run_compairr_existence


@dataclass
class InnovationScores:
    """ Class to store precision and sensitivity scores for different models and datasets. """
    innovation_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["dataset", "model", "precision_innovation", "sensitivity_innovation", "n_gen_novel", "n_test_only"]
    ))


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
    """ Compute precision and sensitivity scores and plot them.
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

    plot_innovation_scores_by_n_gen_novel(analysis_config, scores)
    plot_innovation_precision_sensitivity(analysis_config, scores)
    if analysis_config.receptor_type != "UMI":
        plot_innovation_sensitivity_by_model(analysis_config, scores)
    scores.innovation_df.to_csv(f"{analysis_config.analysis_output_dir}/innovation_scores.csv", index=False)


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
        for gen_file in gen_files:
            compute_compairr_overlap_ratio(analysis_config, ref_file, gen_file, compairr_output_dir, model, scores)


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
        float: Ratio of non-zero overlap counts to total counts.
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


def plot_innovation_sensitivity_by_model(analysis_config: AnalysisConfig, scores: InnovationScores) -> None:
    """ Plot innovation sensitivity per model, with one dot per dataset and a mean marker.
    Args:
        analysis_config (AnalysisConfig): Configuration for the analysis, including paths and model names.
        scores (InnovationScores): Storage class for innovation scores.
    Returns:
        None
    """
    df = scores.innovation_df.copy()

    colors = px.colors.qualitative.Dark24
    model_names_sorted = sorted(analysis_config.model_names)
    color_map = {model: colors[i % len(colors)] for i, model in enumerate(model_names_sorted)}

    fig = px.strip(
        df,
        x="model",
        y="sensitivity_innovation",
        color="model",
        hover_data=["dataset"],
        category_orders={"model": model_names_sorted},
        color_discrete_map=color_map,
        stripmode="overlay",
    )

    fig.update_traces(marker=dict(size=11, opacity=0.6), jitter=0.3)

    # Mean per model, drawn as a horizontal tick over each column
    mean_per_model = (
        df.groupby("model", as_index=False)["sensitivity_innovation"]
        .mean()
        .set_index("model")
        .loc[model_names_sorted]
        .reset_index()
    )

    fig.add_trace(
        go.Scatter(
            x=mean_per_model["model"],
            y=mean_per_model["sensitivity_innovation"],
            mode="markers",
            marker=dict(symbol="line-ew", size=26, line=dict(width=2.0, color="rgba(50,50,50,0.9)")),
            name="Mean",
            hovertemplate="Mean: %{y:.3g}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title={'text': f"Innovation Scores for {analysis_config.receptor_type}",
               'font': {'size': 28}},
        template="plotly_white",
        colorway=px.colors.qualitative.Dark24,
        xaxis_title={'text': "Model", 'font': {'size': 24}},
        yaxis_title={'text': "Innovation score", 'font': {'size': 24}},
        xaxis=dict(tickfont=dict(size=18)),
        yaxis=dict(tickfont=dict(size=18)),
        showlegend=False
    )

    output_path = (
        f"{analysis_config.analysis_output_dir}/"
        "innovation_sensitivity_by_model.png"
    )

    fig.write_image(output_path, width=900, height=600, scale=2)
