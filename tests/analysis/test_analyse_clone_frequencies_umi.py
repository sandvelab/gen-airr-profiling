import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from gen_airr_bm.analysis.analyse_clone_frequencies_umi import (
    get_frequencies_df,
    pseudo_log_transform,
    create_scatter_plot,
    plot_frequencies_by_dataset,
    plot_frequencies_combined,
    collect_and_plot_clone_frequencies,
    split_subset_name,
    summarize_and_plot_all,
    run_clone_frequency_statistics,
)
from gen_airr_bm.core.analysis_config import AnalysisConfig

MODULE = "gen_airr_bm.analysis.analyse_clone_frequencies_umi"


@pytest.fixture
def sample_analysis_config(tmp_path):
    return AnalysisConfig(
        analysis="clone_frequencies",
        model_names=["model1", "model2"],
        analysis_output_dir=str(tmp_path),
        root_output_dir=str(tmp_path),
        default_model_name="humanIGH",
        reference_data=["train", "test"],
        n_subsets=2,
        subfolder_name="all models",
        receptor_type="BCR UMI"
    )


def make_scores_df(scores: dict) -> pd.DataFrame:
    """ Scores as {(reference, repertoire, model): [jsd per subset]}, in the format of the clone frequency scores. """
    return pd.DataFrame([{"metric": "clone_frequency", "model": model, "reference": reference, "dataset": repertoire,
                          "subset": subset, "jsd": jsd}
                         for (reference, repertoire, model), jsds in scores.items()
                         for subset, jsd in enumerate(jsds)])


def test_get_frequencies_df():
    sample_1 = ["AAA", "BBB", "AAA", "CCC"]  # counts: AAA=2, BBB=1, CCC=1
    sample_2 = ["AAA", "AAA", "DDD"]          # counts: AAA=2, DDD=1

    df = get_frequencies_df(sample_1, sample_2, "train", "test")

    # Index contains union of sequences
    expected_index = set(["AAA", "BBB", "CCC", "DDD"])
    assert set(df.index) == expected_index

    # Counts should be integers and correct
    assert df.loc["AAA", "count_train"] == 2
    assert df.loc["BBB", "count_train"] == 1
    assert df.loc["CCC", "count_train"] == 1
    assert df.loc["DDD", "count_train"] == 0

    assert df.loc["AAA", "count_test"] == 2
    assert df.loc["DDD", "count_test"] == 1
    assert df.loc["BBB", "count_test"] == 0

    # Frequencies should be counts / sample size
    assert np.isclose(df.loc["AAA", "freq_train"], 2/4)
    assert np.isclose(df.loc["AAA", "freq_test"], 2/3)
    assert np.isclose(df.loc["DDD", "freq_test"], 1/3)


def test_pseudo_log_transform():
    # Includes zeros and small/large positives to ensure it doesn't error and is monotonic non-decreasing
    x = np.array([0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0])
    y = pseudo_log_transform(x)

    # Should preserve shape and be finite
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))

    # Monotonic non-decreasing
    assert np.all(np.diff(y) >= -1e-12)

    # Zero maps to zero in typical symlog
    assert np.isclose(y[0], 0.0)


def test_create_scatter_plot():
    # Minimal combined df with required columns
    df = pd.DataFrame({
        "pseudo_freq_model": [0.0, 0.5, 1.0],
        "pseudo_freq_reference": [0.0, 0.4, 1.2],
        "sequence": ["AAA", "BBB", "CCC"],
    })

    fig = create_scatter_plot(df, name1="reference", name2="model", title_text="Test Title")

    assert isinstance(fig, go.Figure)
    # Scatter trace present
    assert any(trace.type == "scatter" for trace in fig.data)


def test_plot_frequencies_by_dataset(tmp_path, monkeypatch):
    # Prepare tiny frequency frames
    df = pd.DataFrame({
        "count_ref": [1, 1],
        "count_mod": [1, 0],
        "freq_ref": [0.5, 0.5],
        "freq_mod": [0.5, 0.0],
    }, index=["AAA", "BBB"])  # index acts as sequence ids

    freqs = {"set1": df}

    # Speed up: patch write_image to avoid heavy image engine; instead, create a small placeholder file
    created_files = []

    def fake_write_image(self, path, **kwargs):
        # just create an empty file to simulate output
        open(path, "wb").close()
        created_files.append(path)

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image, raising=False)

    outdir = tmp_path / "plots"
    outdir.mkdir()

    jsds = plot_frequencies_by_dataset(freqs, str(outdir), name1="ref", name2="mod")

    # Should have returned one JSD (since dataset name has no _all), keyed by dataset name
    assert list(jsds) == ["set1"]
    # File created with expected naming
    expected_png = outdir / "set1_mod_ref_symlog.png"
    assert expected_png.exists()


def test_plot_frequencies_combined(tmp_path, monkeypatch):
    # Build two datasets, one with _all and one without
    df_all = pd.DataFrame({
        "count_ref": [1, 1],
        "count_mod": [1, 0],
        "freq_ref": [0.5, 0.5],
        "freq_mod": [0.5, 0.0],
    }, index=["AAA", "BBB"])  

    df_part = pd.DataFrame({
        "count_ref": [1, 1],
        "count_mod": [1, 1],
        "freq_ref": [1.0, 1.0],
        "freq_mod": [1.0, 1.0],
    }, index=["CCC", "DDD"])

    freqs = {
        "dataset_all": df_all,
        "dataset_part": df_part,
    }

    created_files = []

    def fake_write_image(self, path, **kwargs):
        open(path, "wb").close()
        created_files.append(path)

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image, raising=False)

    outdir = tmp_path / "plots2"
    outdir.mkdir()

    # With filter_combined_rep=True -> only uses dataset_all
    plot_frequencies_combined(freqs, str(outdir), "ref", "mod", 0, filter_combined_rep=True)
    expected_png = outdir / "combined_repertoires_mod_ref_symlog_jsd.png"
    assert expected_png.exists()

    # Ensure figure creation worked even when including all datasets
    # # Overwrite file again should still succeed
    plot_frequencies_combined(freqs, str(outdir), "ref", "mod", 0, filter_combined_rep=False)
    assert expected_png.exists()  # still present


def test_split_subset_name():
    assert split_subset_name("IGK_scov11_m6_0_3") == ("IGK_scov11_m6_0", 3)


def test_collect_and_plot_clone_frequencies(mocker, sample_analysis_config):
    mocker.patch(f"{MODULE}.collect_clone_frequencies_models", return_value={})
    mocker.patch(f"{MODULE}.collect_clone_frequencies_reference", return_value={})
    mocker.patch(f"{MODULE}.plot_frequencies_combined")
    # Two subsets of two repertoires per model and reference, then the train vs. test reference per repertoire
    model_jsds = {"rep_1_0": 0.2, "rep_1_1": 0.4, "rep_2_0": 0.5, "rep_2_1": 0.7}
    reference_jsds = {"reference_rep_2": 0.12, "reference_rep_1": 0.1}
    mocker.patch(f"{MODULE}.plot_frequencies_by_dataset", side_effect=[model_jsds] * 4 + [reference_jsds])
    mock_summarize = mocker.patch(f"{MODULE}.summarize_and_plot_all")
    mock_statistics = mocker.patch(f"{MODULE}.run_clone_frequency_statistics")

    collect_and_plot_clone_frequencies(sample_analysis_config, sample_analysis_config.analysis_output_dir)

    scores_df = pd.read_csv(f"{sample_analysis_config.analysis_output_dir}/clone_frequency_jsd_scores.tsv", sep="\t")
    assert len(scores_df) == 16
    assert scores_df[["model", "reference", "dataset", "subset"]].iloc[1].tolist() == ["model1", "train", "rep_1", 1]

    _, _, reference_score = mock_summarize.call_args.args
    assert reference_score == pytest.approx(0.11)
    reference_scores = mock_statistics.call_args.args[2]
    assert reference_scores["repertoire"].tolist() == ["rep_1", "rep_2"]
    assert reference_scores["score"].tolist() == pytest.approx([0.1, 0.12])


def test_collect_and_plot_clone_frequencies_with_tests_requires_one_repertoire_per_donor(mocker,
                                                                                         sample_analysis_config):
    sample_analysis_config.statistical_tests = True
    sample_analysis_config.donor_pattern = "^(rep)_"
    mocker.patch(f"{MODULE}.collect_clone_frequencies_models", return_value={})
    mocker.patch(f"{MODULE}.collect_clone_frequencies_reference", return_value={})
    mocker.patch(f"{MODULE}.plot_frequencies_combined")
    mocker.patch(f"{MODULE}.plot_frequencies_by_dataset", side_effect=[{"rep_1_0": 0.2, "rep_2_0": 0.3}] * 4 + [{}])

    with pytest.raises(ValueError, match="one repertoire per donor"):
        collect_and_plot_clone_frequencies(sample_analysis_config, sample_analysis_config.analysis_output_dir)


def test_summarize_and_plot_all(mocker, sample_analysis_config):
    mock_plot_grouped = mocker.patch(f"{MODULE}.plot_grouped_avg_scores")
    scores_df = make_scores_df({
        ("test", "rep_1", "model1"): [0.2, 0.4], ("test", "rep_2", "model1"): [0.5, 0.7],
        ("train", "rep_1", "model1"): [0.5, 0.7], ("train", "rep_2", "model1"): [0.8, 0.6],
    })

    summarize_and_plot_all(sample_analysis_config, scores_df, 0.21)

    kwargs = mock_plot_grouped.call_args.kwargs
    # Repertoire means: test 0.3 and 0.6, train 0.6 and 0.7
    assert kwargs["mean_scores_by_ref"]["test"]["model1"] == pytest.approx(0.45)
    assert kwargs["mean_scores_by_ref"]["train"]["model1"] == pytest.approx(0.65)
    # The error bars are the SD across the repertoire means, not across all subsets
    assert kwargs["std_scores_by_ref"]["test"]["model1"] == pytest.approx(np.std([0.3, 0.6], ddof=1))
    assert kwargs["repertoire_scores_by_ref"]["test"]["model1"] == [("rep_1", pytest.approx(0.3)),
                                                                    ("rep_2", pytest.approx(0.6))]
    assert kwargs["error_bar_caption"].startswith("Error bars: SD across repertoires (n=2)")
    assert kwargs["file_name"] == "clone_frequency_jsd_comparison"
    assert kwargs["reference_score"] == pytest.approx(0.21)


@pytest.mark.parametrize("statistical_tests", [False, True])
def test_run_clone_frequency_statistics(sample_analysis_config, statistical_tests):
    sample_analysis_config.statistical_tests = statistical_tests
    sample_analysis_config.donor_pattern = "^(rep_\\d+)$"
    sample_analysis_config.contrasts = [["model2", "model1"]]
    scores_df = make_scores_df({
        ("test", f"rep_{i}", model): [base + 0.01 * i, base + 0.02 * i]
        for i in range(1, 4) for model, base in [("model1", 0.2), ("model2", 0.3)]
    } | {("train", "rep_1", "model1"): [0.9, 0.9]})
    reference_scores = pd.DataFrame({"repertoire": ["rep_1", "rep_2", "rep_3"], "score": [0.1, 0.12, 0.11]})

    run_clone_frequency_statistics(sample_analysis_config, scores_df, reference_scores)

    prefix = f"{sample_analysis_config.analysis_output_dir}/statistics/clone_frequency"
    vs_train_test = pd.read_csv(f"{prefix}_vs_train_test.tsv", sep="\t")
    # Only the test scores are compared, three repertoires per model
    assert vs_train_test["first"].tolist() == ["model1", "model2"]
    assert vs_train_test["n_repertoires"].tolist() == [3, 3]
    assert ("p_adjusted" in vs_train_test.columns) == statistical_tests
    contrasts = pd.read_csv(f"{prefix}_contrasts.tsv", sep="\t")
    assert contrasts[["first", "second"]].values.tolist() == [["model2", "model1"]]
    for plot in ["vs_train_test", "contrasts"]:
        assert os.path.exists(f"{prefix}_{plot}.png")


def test_run_clone_frequency_statistics_skips_without_test(sample_analysis_config):
    sample_analysis_config.reference_data = ["train"]
    scores_df = make_scores_df({("train", "rep_1", "model1"): [0.2, 0.3]})

    run_clone_frequency_statistics(sample_analysis_config, scores_df, pd.DataFrame(columns=["repertoire", "score"]))

    assert not os.path.exists(f"{sample_analysis_config.analysis_output_dir}/statistics")
