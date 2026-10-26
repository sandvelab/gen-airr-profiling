import os
import numpy as np
import pandas as pd
import pytest

from gen_airr_bm.analysis.analyse_memorization import (
    run_memorization_analysis,
    get_model_memorization_scores,
    get_reference_memorization_scores,
    get_memorization_scores,
    plot_results,
    run_memorization_statistics,
)
from gen_airr_bm.core.analysis_config import AnalysisConfig


@pytest.fixture
def sample_analysis_config():
    return AnalysisConfig(
        analysis="memorization",
        model_names=["model1", "model2"],
        analysis_output_dir="/tmp/test_output/analysis_mem",
        root_output_dir="/tmp/test_output",
        default_model_name="humanTRB",
        reference_data=["train", "test"],
        n_subsets=5,
        subfolder_name="analysis_subfolder",
        allowed_mismatches=0,
        indels=False,
        receptor_type="TCR"
    )


def make_scores_df(scores: dict) -> pd.DataFrame:
    """ Scores as {(model, repertoire): [score per subset]}, in the format of get_model_memorization_scores. """
    return pd.DataFrame([{"model": model, "file": repertoire, "subset": subset, "memorization_score": score,
                          "novel_sequences": 100}
                         for (model, repertoire), subset_scores in scores.items()
                         for subset, score in enumerate(subset_scores)])


def test_run_memorization_analysis(mocker, sample_analysis_config):
    # Mocks
    mock_makedirs = mocker.patch("os.makedirs")
    mock_get_model_scores = mocker.patch(
        "gen_airr_bm.analysis.analyse_memorization.get_model_memorization_scores",
        return_value=make_scores_df({("model1", "rep_1"): [0.1, 0.2], ("model1", "rep_2"): [0.5, 0.5],
                                     ("model2", "rep_1"): [0.3, 0.3]})
    )
    mock_get_reference_scores = mocker.patch(
        "gen_airr_bm.analysis.analyse_memorization.get_reference_memorization_scores"
    )
    mock_plot_results = mocker.patch("gen_airr_bm.analysis.analyse_memorization.plot_results")
    mock_statistics = mocker.patch("gen_airr_bm.analysis.analyse_memorization.run_memorization_statistics")

    run_memorization_analysis(sample_analysis_config)

    mock_makedirs.assert_called_once_with("/tmp/test_output/analysis_mem", exist_ok=True)
    mock_get_model_scores.assert_called_once_with(
        sample_analysis_config, "/tmp/test_output/analysis_mem", "train"
    )
    # No train vs. test reference for non-UMI TCR, since train and test contain different unique clones
    mock_get_reference_scores.assert_not_called()

    # The subsets are averaged within each repertoire before plotting
    repertoire_scores, reference_scores, fig_dir, file_name, receptor_type, collection = \
        mock_plot_results.call_args.args
    assert repertoire_scores[["source", "repertoire", "score"]].values.tolist() == [
        ["model1", "rep_1", pytest.approx(0.15)], ["model1", "rep_2", pytest.approx(0.5)],
        ["model2", "rep_1", pytest.approx(0.3)]]
    assert (reference_scores, fig_dir, file_name, receptor_type, collection) == (
        None, "/tmp/test_output/analysis_mem", "memorization", "TCR", None)
    mock_statistics.assert_called_once_with(sample_analysis_config, repertoire_scores, None)


def test_run_memorization_analysis_umi_reference(mocker, sample_analysis_config):
    sample_analysis_config.receptor_type = "BCR UMI"
    mocker.patch("os.makedirs")
    mocker.patch("gen_airr_bm.analysis.analyse_memorization.get_model_memorization_scores",
                 return_value=make_scores_df({("model1", "rep_1"): [0.1, 0.2]}))
    reference_scores = pd.DataFrame({"repertoire": ["rep_1"], "score": [0.05]})
    mocker.patch("gen_airr_bm.analysis.analyse_memorization.get_reference_memorization_scores",
                 return_value=reference_scores)
    mock_plot_results = mocker.patch("gen_airr_bm.analysis.analyse_memorization.plot_results")
    mock_statistics = mocker.patch("gen_airr_bm.analysis.analyse_memorization.run_memorization_statistics")

    run_memorization_analysis(sample_analysis_config)

    assert mock_plot_results.call_args.args[1] is reference_scores
    assert mock_statistics.call_args.args[2] is reference_scores


def test_run_memorization_analysis_with_tests_requires_one_repertoire_per_donor(mocker, sample_analysis_config):
    sample_analysis_config.statistical_tests = True
    sample_analysis_config.donor_pattern = "^(rep)_"
    mocker.patch("os.makedirs")
    mocker.patch("gen_airr_bm.analysis.analyse_memorization.get_model_memorization_scores",
                 return_value=make_scores_df({("model1", "rep_1"): [0.1], ("model1", "rep_2"): [0.2]}))

    with pytest.raises(ValueError, match="one repertoire per donor"):
        run_memorization_analysis(sample_analysis_config)


def test_run_memorization_analysis_empty_reference(mocker, sample_analysis_config):
    mocker.patch("os.makedirs")

    # Case 1: Missing 'train'
    sample_analysis_config.reference_data = ["test"]
    with pytest.raises(ValueError):
        run_memorization_analysis(sample_analysis_config)


def test_get_model_memorization_scores(mocker, sample_analysis_config):
    # Prepare a deterministic mapping of reference->generated files
    comparison_mapping = {
        "/ref/ref1.tsv": ["/gen/ref1_0.tsv", "/gen/ref1_1.tsv"],
        "/ref/ref2.tsv": ["/gen/ref2_0.tsv"]
    }

    mock_get_sequence_files = mocker.patch(
        "gen_airr_bm.analysis.analyse_memorization.get_sequence_files",
        return_value=comparison_mapping
    )

    # Side effect returns one value per generated file to match lengths
    def mem_scores_side_effect(analysis_config, ref_file, gen_files, output_dir, model_name):
        return [0.1 * (i + 1) for i in range(len(gen_files))], [10] * len(gen_files)  # scores, n sequences

    mock_get_mem_scores = mocker.patch(
        "gen_airr_bm.analysis.analyse_memorization.get_memorization_scores",
        side_effect=mem_scores_side_effect
    )

    mock_to_csv = mocker.patch("pandas.DataFrame.to_csv")

    result = get_model_memorization_scores(
        analysis_config=sample_analysis_config,
        output_dir="/tmp/test_output/analysis_mem",
        train_reference="train"
    )

    # Sequence files should be requested for each model with the proper train reference
    assert mock_get_sequence_files.call_count == len(sample_analysis_config.model_names)
    for i, model_name in enumerate(sample_analysis_config.model_names):
        assert mock_get_sequence_files.call_args_list[i].args == (
            sample_analysis_config, model_name, "train"
        )

    # Memorization scores computed for each ref/gen pair per model
    assert mock_get_mem_scores.call_count == len(comparison_mapping) * len(sample_analysis_config.model_names)
    mock_to_csv.assert_called_once()

    # One row per model and generated subset, with the repertoire and the subset index from the file names
    assert result.columns.tolist() == ["model", "file", "subset", "memorization_score", "novel_sequences"]
    model1 = result[result["model"] == "model1"]
    assert model1[["file", "subset", "memorization_score"]].values.tolist() == [
        ["ref1", 0, pytest.approx(0.1)], ["ref1", 1, pytest.approx(0.2)], ["ref2", 0, pytest.approx(0.1)]]


def test_get_reference_memorization_scores(mocker, sample_analysis_config):
    # Prepare train/test file pairs, unsorted
    file_pairs = [
        ("/ref/train/rep_2.tsv", "/ref/test/rep_2.tsv"),
        ("/ref/train/rep_1.tsv", "/ref/test/rep_1.tsv"),
    ]

    mocker.patch(
        "gen_airr_bm.analysis.analyse_memorization.get_reference_files",
        return_value=file_pairs
    )

    # get_memorization_scores returns a list, we'll return one score to match usage
    mock_get_mem_scores = mocker.patch(
        "gen_airr_bm.analysis.analyse_memorization.get_memorization_scores",
        side_effect=[([0.1], [100]), ([0.3], [100])]
    )

    reference_scores = get_reference_memorization_scores(
        analysis_config=sample_analysis_config,
        output_dir="/tmp/test_output/analysis_mem"
    )

    # One score per repertoire, in sorted order
    assert mock_get_mem_scores.call_args_list[0].args == (
        sample_analysis_config, "/ref/train/rep_1.tsv", ["/ref/test/rep_1.tsv"], "/tmp/test_output/analysis_mem",
        "reference"
    )
    assert reference_scores["repertoire"].tolist() == ["rep_1", "rep_2"]
    assert reference_scores["score"].tolist() == pytest.approx([0.1, 0.3])


def test_get_memorization_scores_calls_compute(mocker, sample_analysis_config):
    ref_file = "/ref/train.tsv"
    gen_files = ["/gen/a.tsv", "/gen/b.tsv", "/gen/c.tsv"]

    # Avoid real directory creation
    mock_makedirs = mocker.patch("os.makedirs")

    # Return increasing scores per call
    mock_compute = mocker.patch(
        "gen_airr_bm.analysis.analyse_memorization.compute_overlap_score",
        side_effect=[(0.11, 100), (0.22, 200), (0.33, 300)]
    )

    scores, ns = get_memorization_scores(
        analysis_config=sample_analysis_config,
        train_file=ref_file,
        test_or_gen_files=gen_files,
        output_dir="/tmp/test_output/analysis_mem",
        name="modelX"
    )

    # compute_overlap_score called for each gen file
    assert mock_compute.call_count == len(gen_files)
    for i, gen in enumerate(gen_files):
        assert mock_compute.call_args_list[i].args == (
            sample_analysis_config, ref_file, gen, "/tmp/test_output/analysis_mem/compairr_output", "modelX"
        )

    # Expected output
    assert scores == [0.11, 0.22, 0.33]
    assert ns == [100, 200, 300]


def make_repertoire_scores(scores: dict, donors: dict = None) -> pd.DataFrame:
    """ Scores as {model: [score per repertoire]}, repertoires rep_1, rep_2, ...; every repertoire is its own donor
    unless donors maps repertoires to donors. """
    return pd.DataFrame([{"source": model, "repertoire": f"rep_{i + 1}",
                          "donor": (donors or {}).get(f"rep_{i + 1}", f"rep_{i + 1}"), "score": score,
                          "n_subsets": 2}
                         for model, model_scores in scores.items() for i, score in enumerate(model_scores)])


def test_plot_results(tmp_path):
    repertoire_scores = make_repertoire_scores({"model1": [0.1, 0.2, 0.3], "model2": [0.5, 0.7, 0.6]},
                                               donors={"rep_1": "d1", "rep_2": "d1", "rep_3": "d2"})
    reference_scores = pd.DataFrame({"repertoire": ["rep_1", "rep_2", "rep_3"], "score": [0.4, 0.42, 0.44]})

    plot_results(repertoire_scores, reference_scores, str(tmp_path), "memorization", "BCR UMI")

    assert os.path.exists(tmp_path / "memorization.png")
    # Sorted by mean, descending, with the SD across repertoires
    summary = pd.read_csv(tmp_path / "memorization.tsv", sep="\t")
    assert summary.columns.tolist() == ["model", "mean_overlap_score", "std"]
    assert summary["model"].tolist() == ["model2", "model1"]
    assert summary["mean_overlap_score"].tolist() == pytest.approx([0.6, 0.2])
    assert summary["std"].tolist() == pytest.approx([np.std([0.5, 0.7, 0.6], ddof=1), np.std([0.1, 0.2, 0.3], ddof=1)])
    with open(tmp_path / "memorization_mean_ref.tsv") as f:
        assert float(f.readline()) == pytest.approx(0.42)


def test_plot_results_without_reference(tmp_path):
    plot_results(make_repertoire_scores({"model1": [0.1, 0.2]}), None, str(tmp_path), "memorization", "TCR")

    # Tuning reads "None" as no reference
    with open(tmp_path / "memorization_mean_ref.tsv") as f:
        assert f.readline().strip() == "None"


@pytest.mark.parametrize("statistical_tests", [False, True])
def test_run_memorization_statistics(tmp_path, sample_analysis_config, statistical_tests):
    sample_analysis_config.analysis_output_dir = str(tmp_path)
    sample_analysis_config.statistical_tests = statistical_tests
    sample_analysis_config.donor_pattern = "^(rep_\\d+)$"
    sample_analysis_config.contrasts = [["model2", "model1"]]
    repertoire_scores = make_repertoire_scores({"model1": [0.1, 0.2, 0.25], "model2": [0.5, 0.7, 0.6]})

    run_memorization_statistics(sample_analysis_config, repertoire_scores, None)

    prefix = f"{tmp_path}/statistics/memorization"
    contrasts = pd.read_csv(f"{prefix}_contrasts.tsv", sep="\t")
    assert contrasts[["first", "second", "n_repertoires"]].values.tolist() == [["model2", "model1", 3]]
    assert ("p_adjusted" in contrasts.columns) == statistical_tests
    assert os.path.exists(f"{prefix}_contrasts.png")
    # Without a train vs. test reference there is no comparison with it
    assert not os.path.exists(f"{prefix}_vs_train_test.tsv")


def test_run_memorization_statistics_with_reference(tmp_path, sample_analysis_config):
    sample_analysis_config.analysis_output_dir = str(tmp_path)
    repertoire_scores = make_repertoire_scores({"model1": [0.1, 0.2], "model2": [0.5, 0.7]})
    reference_scores = pd.DataFrame({"repertoire": ["rep_1", "rep_2"], "score": [0.05, 0.06]})

    run_memorization_statistics(sample_analysis_config, repertoire_scores, reference_scores)

    vs_train_test = pd.read_csv(f"{tmp_path}/statistics/memorization_vs_train_test.tsv", sep="\t")
    assert vs_train_test["first"].tolist() == ["model1", "model2"]
    assert os.path.exists(f"{tmp_path}/statistics/memorization_vs_train_test.png")
