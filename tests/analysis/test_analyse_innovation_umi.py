import os

import numpy as np
import pandas as pd
import pytest

from gen_airr_bm.analysis.analyse_innovation_umi import (
    InnovationScores,
    collect_model_scores,
    compute_and_plot_innovation_scores,
    plot_innovation_scores,
    run_innovation_statistics,
)
from gen_airr_bm.core.analysis_config import AnalysisConfig

MODULE = "gen_airr_bm.analysis.analyse_innovation_umi"


@pytest.fixture
def sample_analysis_config(tmp_path):
    return AnalysisConfig(
        analysis="innovation_umi",
        model_names=["modelA", "modelB"],
        analysis_output_dir=str(tmp_path),
        root_output_dir=str(tmp_path),
        default_model_name="humanIGH",
        reference_data=["train", "test"],
        n_subsets=2,
        subfolder_name="all models excluding train",
        allowed_mismatches=0,
        indels=False,
        receptor_type="BCR UMI"
    )


def make_scores_df(scores: dict) -> pd.DataFrame:
    """ Scores as {(metric, repertoire, model): [score per subset]}, in the format of the innovation scores. """
    return pd.DataFrame([{"metric": metric, "model": model, "reference": "test_only", "dataset": repertoire,
                          "subset": subset, "score": score}
                         for (metric, repertoire, model), subset_scores in scores.items()
                         for subset, score in enumerate(subset_scores)])


def test_collect_model_scores(mocker, sample_analysis_config):
    mocker.patch(f"{MODULE}.get_sequence_files", return_value={
        "/out/test_only_compairr_sequences/rep_1.tsv": ["/out/gen/modelA/rep_1_0.tsv", "/out/gen/modelA/rep_1_1.tsv"]})
    mocker.patch(f"{MODULE}.get_innovation_scores", return_value=([0.1, 0.3], [0.01, 0.03]))
    scores = InnovationScores()

    collect_model_scores(sample_analysis_config, "modelA", "test_only", "/out/compairr", scores)

    # One row per metric and generated subset, with the subset index from the file name
    assert [(row["metric"], row["dataset"], row["subset"], row["score"]) for row in scores.scores] == [
        ("innovation_sensitivity", "rep_1", 0, 0.1), ("innovation_precision", "rep_1", 0, 0.01),
        ("innovation_sensitivity", "rep_1", 1, 0.3), ("innovation_precision", "rep_1", 1, 0.03)]


def test_plot_innovation_scores_over_repertoires(mocker, sample_analysis_config):
    mock_plot_avg = mocker.patch(f"{MODULE}.plot_avg_innovation_scores")
    for plot in ["plot_innovation_scores_by_n_gen_novel", "plot_innovation_scores_by_n_gen_novel_pseudo_log",
                 "plot_innovation_precision_sensitivity"]:
        mocker.patch(f"{MODULE}.{plot}")
    scores_df = make_scores_df({
        ("innovation_sensitivity", "rep_1", "modelA"): [0.1, 0.3], ("innovation_sensitivity", "rep_2", "modelA"): [0.5, 0.7],
        ("innovation_precision", "rep_1", "modelA"): [0.01, 0.01], ("innovation_precision", "rep_2", "modelA"): [0.03, 0.03],
    })

    plot_innovation_scores(sample_analysis_config, InnovationScores(), scores_df)

    overview_calls = {call.args[5]: call for call in mock_plot_avg.call_args_list}
    sensitivity = overview_calls["innovation_BCR_UMI"]
    # Repertoire means 0.2 and 0.6: the error bar is their SD, not the mean of the within-repertoire SDs
    assert sensitivity.args[1]["modelA"] == pytest.approx(0.4)
    assert sensitivity.args[2]["modelA"] == pytest.approx(np.std([0.2, 0.6], ddof=1))
    assert sensitivity.kwargs["repertoire_scores"]["modelA"] == [("rep_1", pytest.approx(0.2)),
                                                                 ("rep_2", pytest.approx(0.6))]
    assert sensitivity.kwargs["error_bar_caption"].startswith("Error bars: SD across repertoires (n=2)")
    assert overview_calls["innovation_BCR_UMI_normalized"].args[1]["modelA"] == pytest.approx(0.02)


@pytest.mark.parametrize("statistical_tests", [False, True])
def test_run_innovation_statistics(sample_analysis_config, statistical_tests):
    sample_analysis_config.statistical_tests = statistical_tests
    sample_analysis_config.donor_pattern = "^(rep_\\d+)$"
    sample_analysis_config.contrasts = [["modelB", "modelA"]]
    scores_df = make_scores_df({(metric, f"rep_{i}", model): [base + 0.01 * i, base + 0.02 * i]
                                for metric in ["innovation_sensitivity", "innovation_precision"]
                                for i in range(1, 4) for model, base in [("modelA", 0.2), ("modelB", 0.3)]})

    run_innovation_statistics(sample_analysis_config, scores_df)

    for metric in ["innovation_sensitivity", "innovation_precision"]:
        prefix = f"{sample_analysis_config.analysis_output_dir}/statistics/{metric}"
        contrasts = pd.read_csv(f"{prefix}_contrasts.tsv", sep="\t")
        assert contrasts[["first", "second", "n_repertoires"]].values.tolist() == [["modelB", "modelA", 3]]
        assert ("p_adjusted" in contrasts.columns) == statistical_tests
        assert os.path.exists(f"{prefix}_contrasts.png")


def test_compute_and_plot_with_tests_requires_one_repertoire_per_donor(mocker, sample_analysis_config):
    sample_analysis_config.statistical_tests = True
    sample_analysis_config.donor_pattern = "^(rep)_"
    mocker.patch(f"{MODULE}.preprocess_test_for_innovation_sensitivity")
    mocker.patch(f"{MODULE}.preprocess_gen_for_innovation_precision")

    def collect(analysis_config, model, test_reference, compairr_output_dir, scores):
        scores.scores.extend(make_scores_df({("innovation_sensitivity", f"rep_{i}", model): [0.5] for i in range(1, 3)})
                             .to_dict("records"))

    mocker.patch(f"{MODULE}.collect_model_scores", side_effect=collect)

    with pytest.raises(ValueError, match="one repertoire per donor"):
        compute_and_plot_innovation_scores(sample_analysis_config, "/out/compairr")
