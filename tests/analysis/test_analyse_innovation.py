import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from gen_airr_bm.analysis.analyse_innovation import (
    InnovationScores,
    collect_innovation_scores,
    compute_and_plot_innovation_scores,
    compute_compairr_overlap_ratio,
    plot_innovation_sensitivity_by_model,
    run_innovation_statistics,
)
from gen_airr_bm.core.analysis_config import AnalysisConfig

MODULE = "gen_airr_bm.analysis.analyse_innovation"


@pytest.fixture
def sample_analysis_config(tmp_path):
    return AnalysisConfig(
        analysis="innovation",
        model_names=["modelA", "modelB"],
        analysis_output_dir=str(tmp_path),
        root_output_dir=str(tmp_path),
        default_model_name="humanTRB",
        reference_data=["train", "test"],
        n_subsets=2,
        subfolder_name="all models",
        allowed_mismatches=0,
        indels=False,
        receptor_type="TCR"
    )


def make_scores_df(scores: dict) -> pd.DataFrame:
    """ Scores as {(metric, repertoire, model): [score per subset]}, in the format of the innovation scores. """
    return pd.DataFrame([{"metric": metric, "model": model, "reference": "test", "dataset": repertoire,
                          "subset": subset, "score": score}
                         for (metric, repertoire, model), subset_scores in scores.items()
                         for subset, score in enumerate(subset_scores)])


def test_compute_compairr_overlap_ratio(mocker, tmp_path, sample_analysis_config):
    mocker.patch(f"{MODULE}.run_compairr_existence")
    compairr_dir = tmp_path / "compairr"
    compairr_dir.mkdir()
    # 2 of the 4 test sequences are found among the 5 generated sequences
    pd.DataFrame({"sequence_id": list("abcd"), "overlap_count": [1, 0, 2, 0]}).to_csv(
        compairr_dir / "rep_1_0_modelA_innovation_overlap.tsv", sep="\t", index=False)
    pd.DataFrame({"junction_aa": list("ABCD")}).to_csv(tmp_path / "rep_1.tsv", sep="\t", index=False)
    pd.DataFrame({"junction_aa": list("ABCDE")}).to_csv(tmp_path / "rep_1_0.tsv", sep="\t", index=False)
    scores = InnovationScores()

    sensitivity = compute_compairr_overlap_ratio(sample_analysis_config, str(tmp_path / "rep_1.tsv"),
                                                            str(tmp_path / "rep_1_0.tsv"), str(compairr_dir),
                                                            "modelA", scores)

    assert sensitivity == pytest.approx(2 / 4)
    # The precision is only kept in innovation_df
    assert scores.innovation_df.iloc[0].tolist() == ["rep_1_0", "modelA", pytest.approx(0.4), pytest.approx(0.5), 5, 4]


def test_collect_innovation_scores(mocker, sample_analysis_config):
    mocker.patch(f"{MODULE}.get_sequence_files", return_value={
        "/out/test_compairr_sequences/rep_1.tsv": ["/out/gen/modelA/rep_1_0.tsv", "/out/gen/modelA/rep_1_1.tsv"]})
    mocker.patch(f"{MODULE}.compute_compairr_overlap_ratio", side_effect=[0.1, 0.3])
    scores = InnovationScores()

    collect_innovation_scores(sample_analysis_config, "modelA", "test", "/out/compairr", scores)

    # One sensitivity row per generated subset, with the repertoire and the subset index from the file names
    assert [(row["metric"], row["dataset"], row["subset"], row["score"]) for row in scores.scores] == [
        ("innovation_sensitivity", "rep_1", 0, 0.1), ("innovation_sensitivity", "rep_1", 1, 0.3)]


def test_plot_innovation_sensitivity_by_model(mocker, sample_analysis_config):
    mock_write = mocker.patch.object(go.Figure, "write_image", autospec=True)
    scores_df = make_scores_df({
        ("innovation_sensitivity", "rep_1", "modelA"): [0.1, 0.3],
        ("innovation_sensitivity", "rep_2", "modelA"): [0.5, 0.7],
        ("innovation_sensitivity", "rep_1", "modelB"): [0.2, 0.2],
        ("innovation_sensitivity", "rep_2", "modelB"): [0.4, 0.4],
    })

    plot_innovation_sensitivity_by_model(sample_analysis_config, scores_df)

    fig = mock_write.call_args.args[0]
    # One dot per generated subset, in circles since every repertoire is its own donor
    model_a_dots = [trace for trace in fig.data if trace.name == "modelA"]
    assert len(model_a_dots) == 1
    assert sorted(model_a_dots[0].y) == pytest.approx([0.1, 0.3, 0.5, 0.7])
    assert model_a_dots[0].marker.symbol == "circle"
    assert not fig.layout.showlegend
    # The mean over repertoires, with the SD across the repertoire means as error bars
    mean_trace = next(trace for trace in fig.data if trace.name == "Mean")
    assert list(mean_trace.x) == ["modelA", "modelB"]
    assert list(mean_trace.y) == pytest.approx([0.4, 0.3])
    assert list(mean_trace.error_y.array) == pytest.approx([np.std([0.2, 0.6], ddof=1), np.std([0.2, 0.4], ddof=1)])
    assert fig.layout.annotations[0].text == ("Error bars: SD across repertoires (n=2); dots: generated subsets")


def test_plot_innovation_sensitivity_by_model_donor_symbols(mocker, sample_analysis_config):
    mock_write = mocker.patch.object(go.Figure, "write_image", autospec=True)
    sample_analysis_config.donor_pattern = "^([^_]+)_"
    scores_df = make_scores_df({("innovation_sensitivity", repertoire, "modelA"): [0.1, 0.3]
                                for repertoire in ["d1_pLN", "d1_spleen", "d2_pLN"]})

    plot_innovation_sensitivity_by_model(sample_analysis_config, scores_df)

    fig = mock_write.call_args.args[0]
    # The subsets of each donor get their own symbol, with a legend entry per donor
    dots = {trace.marker.symbol: len(trace.y) for trace in fig.data if trace.name == "modelA"}
    assert dots == {"circle": 4, "square": 2}
    assert [trace.name for trace in fig.data if trace.showlegend is not False] == ["Donor d1", "Donor d2"]
    assert fig.layout.showlegend
    assert fig.layout.annotations[0].text.startswith("Error bars: SD across repertoires (n=3 from 2 donors)")


@pytest.mark.parametrize("statistical_tests", [False, True])
def test_run_innovation_statistics(sample_analysis_config, statistical_tests):
    sample_analysis_config.statistical_tests = statistical_tests
    sample_analysis_config.donor_pattern = "^(rep_\\d+)$"
    sample_analysis_config.contrasts = [["modelB", "modelA"]]
    scores_df = make_scores_df({("innovation_sensitivity", f"rep_{i}", model): [base + 0.01 * i, base + 0.02 * i]
                                for i in range(1, 4) for model, base in [("modelA", 0.2), ("modelB", 0.3)]})

    run_innovation_statistics(sample_analysis_config, scores_df)

    prefix = f"{sample_analysis_config.analysis_output_dir}/statistics/innovation_sensitivity"
    contrasts = pd.read_csv(f"{prefix}_contrasts.tsv", sep="\t")
    assert contrasts[["first", "second", "n_repertoires"]].values.tolist() == [["modelB", "modelA", 3]]
    assert ("p_adjusted" in contrasts.columns) == statistical_tests
    assert os.path.exists(f"{prefix}_contrasts.png")
    # There is no train vs. test reference to compare with, and no statistics for the innovation precision
    assert not os.path.exists(f"{prefix}_vs_train_test.tsv")
    statistics_dir = f"{sample_analysis_config.analysis_output_dir}/statistics"
    assert not os.path.exists(f"{statistics_dir}/innovation_precision_summary.tsv")


def test_compute_and_plot_with_tests_requires_one_repertoire_per_donor(mocker, sample_analysis_config):
    sample_analysis_config.statistical_tests = True
    sample_analysis_config.donor_pattern = "^(rep)_"

    def collect(analysis_config, model, test_reference, compairr_output_dir, scores):
        scores.scores.extend(make_scores_df({("innovation_sensitivity", f"rep_{i}", model): [0.5]
                                             for i in range(1, 3)}).to_dict("records"))

    mocker.patch(f"{MODULE}.collect_innovation_scores", side_effect=collect)

    with pytest.raises(ValueError, match="one repertoire per donor"):
        compute_and_plot_innovation_scores(sample_analysis_config, "/out/compairr")
