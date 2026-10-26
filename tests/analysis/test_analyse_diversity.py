import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from gen_airr_bm.analysis.analyse_diversity import run_diversity_analysis, compute_and_plot_diversity_scores, \
    compute_diversities_for_models, compute_diversity, shannon_entropy, gini_simpson_index, pielou_evenness, \
    gini_coefficient, plot_diversity_scatter_plotly, \
    run_diversity_statistics
from gen_airr_bm.core.analysis_config import AnalysisConfig


@pytest.fixture
def sample_analysis_config():
    return AnalysisConfig(
        analysis="network",
        model_names=["model1", "model2"],
        analysis_output_dir="/tmp/test_output/analysis",
        root_output_dir="/tmp/test_output",
        default_model_name="humanTRB",
        reference_data=["test"],
        n_subsets=5,
        subfolder_name="analysis_subfolder",
        receptor_type="TCR"
    )


def dummy_diversity_func(data):
    return len(data)


def test_run_diversity_analysis(mocker, sample_analysis_config):
    mock_makedirs = mocker.patch('os.makedirs')
    mock_compute_and_plot_diversity_scores = mocker.patch(
        'gen_airr_bm.analysis.analyse_diversity.compute_and_plot_diversity_scores')

    run_diversity_analysis(sample_analysis_config)

    # Check function calls
    mock_makedirs.assert_called_once_with("/tmp/test_output/analysis", exist_ok=True)
    assert mock_compute_and_plot_diversity_scores.call_count == 4  # Four diversity metrics

    # Check that the data path is passed correctly
    for call in mock_compute_and_plot_diversity_scores.call_args_list:
        assert call[0][1] == {"test": "/tmp/test_output/test_compairr_sequences"}

    # Check that the expected metrics are processed
    expected_metrics = ["Shannon Entropy", "Gini Simpson Index", "Pielou Evenness", "Gini Coefficient"]
    actual_metrics = [call[0][4] for call in mock_compute_and_plot_diversity_scores.call_args_list]
    assert set(actual_metrics) == set(expected_metrics)


def test_run_diversity_analysis_empty_reference(mocker, sample_analysis_config):
    mocker.patch('os.makedirs')
    mock_compute_and_plot_diversity_scores = mocker.patch(
        'gen_airr_bm.analysis.analyse_diversity.compute_and_plot_diversity_scores')

    sample_analysis_config.reference_data = []  # Set empty reference data

    run_diversity_analysis(sample_analysis_config)

    # Check that empty reference_dirs was passed
    assert mock_compute_and_plot_diversity_scores.call_count == 4
    for call in mock_compute_and_plot_diversity_scores.call_args_list:
        assert call[0][1] == {}


def test_compute_and_plot_diversity_scores(mocker, sample_analysis_config):
    mock_compute_diversity = mocker.patch(
        'gen_airr_bm.analysis.analyse_diversity.compute_diversity',
        side_effect=lambda ref_dir, func: {"6174_pLN_CD8_0": 0.5}
    )
    mock_compute_diversities_for_models = mocker.patch(
        'gen_airr_bm.analysis.analyse_diversity.compute_diversities_for_models',
        return_value={
            "model1": {"6174_pLN_CD8_0_0": 0.1, "6174_pLN_CD8_0_1": 0.3},
            "model2": {"6174_pLN_CD8_0_0": 0.2, "6174_pLN_CD8_0_1": 0.4},
        }
    )
    mock_plot = mocker.patch('gen_airr_bm.analysis.analyse_diversity.plot_diversity_scatter_plotly')
    mock_statistics = mocker.patch('gen_airr_bm.analysis.analyse_diversity.run_diversity_statistics')

    compute_and_plot_diversity_scores(
        analysis_config=sample_analysis_config,
        reference_dirs={"test": "/some/fake/path"},
        output_path="/fake/output/plot",
        diversity_function=dummy_diversity_func,
        metric_name="Dummy Metric"
    )

    mock_compute_diversity.assert_called_once_with("/some/fake/path", dummy_diversity_func)
    expected_model_dir = "/tmp/test_output/novel_generated_compairr_sequences_split"
    mock_compute_diversities_for_models.assert_called_once_with(
        ["model1", "model2"], expected_model_dir, dummy_diversity_func
    )

    # One row per model and generated subset, with the repertoire (ending with the experiment index) and the subset
    reference_diversities, scores_df, output_path, metric_name = mock_plot.call_args.args[:4]
    assert reference_diversities == {"test": {"6174_pLN_CD8_0": 0.5}}
    assert scores_df.values.tolist() == [["model1", "6174_pLN_CD8_0", 0, 0.1], ["model1", "6174_pLN_CD8_0", 1, 0.3],
                                         ["model2", "6174_pLN_CD8_0", 0, 0.2], ["model2", "6174_pLN_CD8_0", 1, 0.4]]
    assert (output_path, metric_name) == ("/fake/output/plot", "Dummy Metric")
    mock_statistics.assert_called_once_with(sample_analysis_config, scores_df, "Dummy Metric")


def test_compute_and_plot_diversity_scores_with_tests_requires_one_repertoire_per_donor(mocker,
                                                                                        sample_analysis_config):
    sample_analysis_config.statistical_tests = True
    sample_analysis_config.donor_pattern = "^([^_]+)_"
    mocker.patch('gen_airr_bm.analysis.analyse_diversity.compute_diversity', return_value={})
    mocker.patch('gen_airr_bm.analysis.analyse_diversity.compute_diversities_for_models',
                 return_value={"model1": {"6174_pLN_CD8_0_0": 0.1, "6174_pLN_Treg_1_0": 0.3}, "model2": {}})

    with pytest.raises(ValueError, match="one repertoire per donor"):
        compute_and_plot_diversity_scores(sample_analysis_config, {"test": "/some/fake/path"}, "/fake/output/plot",
                                          dummy_diversity_func, "Dummy Metric")


def test_compute_diversities_for_models_with_config(mocker, sample_analysis_config):
    models = sample_analysis_config.model_names
    gen_dir = f"{sample_analysis_config.root_output_dir}/generated_compairr_sequences_split"

    mock_compute_diversity = mocker.patch(
        "gen_airr_bm.analysis.analyse_diversity.compute_diversity",
        side_effect=lambda path, func: f"mocked_result_for_{path}"
    )

    result = compute_diversities_for_models(models, gen_dir, dummy_diversity_func)

    expected_calls = [
        (f"{gen_dir}/model1", dummy_diversity_func),
        (f"{gen_dir}/model2", dummy_diversity_func)
    ]
    actual_calls = [call.args for call in mock_compute_diversity.call_args_list]
    assert actual_calls == expected_calls

    expected_result = {
        "model1": f"mocked_result_for_{gen_dir}/model1",
        "model2": f"mocked_result_for_{gen_dir}/model2"
    }
    assert result == expected_result


def test_compute_diversity(mocker):
    fake_dir = "/fake/data"
    fake_files = ["dataset1.tsv", "dataset2.tsv"]
    fake_file_paths = [os.path.join(fake_dir, f) for f in fake_files]

    mocker.patch("os.listdir", return_value=fake_files)

    fake_df = pd.DataFrame({"junction_aa": ["AAA", "BBB", "CCC"]})
    mock_read_csv = mocker.patch("pandas.read_csv", return_value=fake_df)

    result = compute_diversity(fake_dir, dummy_diversity_func)

    os.listdir.assert_called_once_with(fake_dir)

    # Assert read_csv called for each dataset
    assert mock_read_csv.call_count == 2
    for path in fake_file_paths:
        mock_read_csv.assert_any_call(path, sep="\t", usecols=["junction_aa"])

    expected_result = {
        "dataset1": 3,
        "dataset2": 3
    }
    assert result == expected_result


@pytest.mark.parametrize(
    "sequences, expected_entropy",
    [
        (["AAA", "AAA", "AAA", "AAA"], 0.0),  # No diversity
        (["AAA", "BBB", "CCC", "DDD"], 2.0),  # Max entropy: 4 items, equally likely
        (["AAA", "AAA", "BBB", "CCC"], -((2/4)*math.log2(2/4) + (1/4)*math.log2(1/4) + (1/4)*math.log2(1/4))),
        ([], 0.0),  # Handle empty list gracefully
    ]
)
def test_shannon_entropy_variations(sequences, expected_entropy):
    result = shannon_entropy(sequences)
    assert math.isclose(result, expected_entropy, rel_tol=1e-9)


@pytest.mark.parametrize(
    "sequences, expected_index",
    [
        (["AAA", "AAA", "AAA", "AAA"], 0.0),             # No diversity
        (["AAA", "BBB", "CCC", "DDD"], 0.75),            # Equal proportions (max diversity for 4 items)
        (["AAA", "AAA", "BBB", "CCC"], 1 - (0.5**2 + 0.25**2 + 0.25**2)),  # Uneven distribution
        ([], 0.0),                                       # Empty list
    ]
)
def test_gini_simpson_index_variations(sequences, expected_index):
    result = gini_simpson_index(sequences)
    assert pytest.approx(result, rel=1e-9) == expected_index


@pytest.mark.parametrize(
    "sequences, expected_evenness",
    [
        (["AAA", "AAA", "AAA", "AAA"], 0.0),  # Only one unique sequence
        (["AAA", "BBB", "CCC", "DDD"], 1.0),  # Perfect evenness
        (["AAA", "AAA", "BBB", "CCC"],
         (-((0.5 * math.log2(0.5)) + 0.25 * math.log2(0.25) + 0.25 * math.log2(0.25)) / math.log2(3)),
        ),
        ([], 0.0),  # Empty list
        (["AAA"], 0.0),  # Only one item
    ]
)
def test_pielou_evenness_variations(sequences, expected_evenness):
    result = pielou_evenness(sequences)
    assert pytest.approx(result, rel=1e-9) == expected_evenness



@pytest.mark.parametrize(
    "sequences, expected_gini",
    [
        ([], 0.0),                          # Empty list → 0
        (["AAA"], 0.0),                     # One item → 0 (perfect equality)
        (["AAA", "AAA", "AAA", "AAA"], 0.0),  # All same → 0
        (["AAA", "BBB", "CCC", "DDD"], 0.0),  # All equally frequent → 0
        (["AAA", "AAA", "AAA", "BBB"], 0.25),  # Unequal frequencies → Gini > 0
    ]
)
def test_gini_coefficient_variations(sequences, expected_gini):
    result = gini_coefficient(sequences)
    assert pytest.approx(result, rel=1e-9) == expected_gini


def make_scores_df(scores: dict) -> pd.DataFrame:
    """ Scores as {(repertoire, model): [score per subset]}, in the format of compute_and_plot_diversity_scores. """
    return pd.DataFrame([{"model": model, "dataset": repertoire, "subset": subset, "score": score}
                         for (repertoire, model), subset_scores in scores.items()
                         for subset, score in enumerate(subset_scores)])


def test_plot_diversity_scatter_plotly(mocker, tmp_path):
    mock_write = mocker.patch.object(go.Figure, "write_image", autospec=True)
    reference_diversities = {"train": {"6174_pLN_CD8_0": 0.9, "6174_pLN_Treg_1": 0.9}}
    scores_df = make_scores_df({("6174_pLN_CD8_0", "modelA"): [0.1, 0.3], ("6174_pLN_Treg_1", "modelA"): [0.5, 0.7]})
    output_path = str(tmp_path / "path")

    plot_diversity_scatter_plotly(reference_diversities, scores_df, output_path, "Gini Coefficient", "TCR",
                                  donor_pattern="^([^_]+)_")

    # One row per generated subset, labelled by repertoire without the experiment index added in training
    plotting_df = pd.read_csv(output_path + ".tsv", sep="\t")
    assert plotting_df.columns.tolist() == ["dataset", "subset", "gini coefficient", "source"]
    assert plotting_df["dataset"].tolist() == ["6174_pLN_CD8", "6174_pLN_CD8", "6174_pLN_Treg", "6174_pLN_Treg"]

    fig = mock_write.call_args.args[0]
    # One dot per subset, coloured by repertoire
    assert {trace.name: len(trace.y) for trace in fig.data if trace.name != "Mean"} == {"6174_pLN_CD8": 2,
                                                                                         "6174_pLN_Treg": 2}
    # The mean over repertoires, with the SD across the repertoire means as error bars
    mean_trace = next(trace for trace in fig.data if trace.name == "Mean")
    assert list(mean_trace.y) == pytest.approx([0.4])
    assert list(mean_trace.error_y.array) == pytest.approx([np.std([0.2, 0.6], ddof=1)])
    assert fig.layout.annotations[0].text == ("Error bars: SD across repertoires (n=2 from 1 donors); "
                                              "dots: generated subsets")
    # The train sequences are all unique, so their diversity is a reference line
    assert fig.layout.shapes[0].y0 == pytest.approx(0.9)
    mock_write.assert_called_once_with(fig, output_path + ".png", scale=3)


def test_plot_diversity_scatter_plotly_umi_references(mocker, tmp_path):
    mock_write = mocker.patch.object(go.Figure, "write_image", autospec=True)
    reference_diversities = {"train": {"IGK_s1_m6_0": 0.8, "IGK_s2_m6_1": 0.6},
                             "test": {"IGK_s1_m6_0": 0.7, "IGK_s2_m6_1": 0.5}}
    scores_df = make_scores_df({("IGK_s1_m6_0", "modelA"): [0.1, 0.3], ("IGK_s2_m6_1", "modelA"): [0.5, 0.7]})

    plot_diversity_scatter_plotly(reference_diversities, scores_df, str(tmp_path / "path"), "Shannon Entropy",
                                  "BCR UMI", donor_pattern="^[^_]+_([^_]+)_")

    fig = mock_write.call_args.args[0]
    # Train and test are sources with one dot per repertoire, with their own mean
    mean_trace = next(trace for trace in fig.data if trace.name == "Mean")
    assert list(mean_trace.x) == ["train", "test", "modelA"]
    assert list(mean_trace.y) == pytest.approx([0.7, 0.6, 0.4])
    assert fig.layout.annotations[0].text.endswith("dots: generated subsets<br>Train and test dots: repertoires")
    assert not fig.layout.shapes


@pytest.mark.parametrize("statistical_tests", [False, True])
def test_run_diversity_statistics(tmp_path, sample_analysis_config, statistical_tests):
    sample_analysis_config.analysis_output_dir = str(tmp_path)
    sample_analysis_config.statistical_tests = statistical_tests
    sample_analysis_config.donor_pattern = "^(rep_\\d+)$"
    sample_analysis_config.contrasts = [["model2", "model1"]]
    scores_df = make_scores_df({(f"rep_{i}", model): [base + 0.01 * i, base + 0.02 * i]
                                for i in range(1, 4) for model, base in [("model1", 0.2), ("model2", 0.3)]})

    run_diversity_statistics(sample_analysis_config, scores_df, "Shannon Entropy")

    prefix = f"{tmp_path}/statistics/diversity_shannon_entropy"
    contrasts = pd.read_csv(f"{prefix}_contrasts.tsv", sep="\t")
    assert contrasts[["first", "second", "n_repertoires"]].values.tolist() == [["model2", "model1", 3]]
    assert ("p_adjusted" in contrasts.columns) == statistical_tests
    assert os.path.exists(f"{prefix}_contrasts.png")
    assert not os.path.exists(f"{prefix}_vs_train_test.tsv")


def test_plot_diversity_scatter_plotly_keeps_names_without_experiment_index(mocker, tmp_path):
    mocker.patch.object(go.Figure, "write_image", autospec=True)
    scores_df = make_scores_df({("6174_pLN_CD8", "modelA"): [0.1, 0.3], ("6174_pLN_Treg", "modelA"): [0.5, 0.7]})

    plot_diversity_scatter_plotly({"train": {"6174_pLN_CD8": 0.9}}, scores_df, str(tmp_path / "path"),
                                  "Gini Coefficient", "TCR")

    plotting_df = pd.read_csv(tmp_path / "path.tsv", sep="\t")
    assert plotting_df["dataset"].unique().tolist() == ["6174_pLN_CD8", "6174_pLN_Treg"]


def test_plot_diversity_scatter_plotly_many_repertoires(mocker, tmp_path):
    mock_write = mocker.patch.object(go.Figure, "write_image", autospec=True)
    mocker.patch("gen_airr_bm.analysis.analyse_diversity.MAX_COLOURED_REPERTOIRES", 2)
    scores_df = make_scores_df({(f"P0000{i}_neg_0", "modelA"): [0.1 * i, 0.1 * i + 0.2] for i in range(1, 4)})

    plot_diversity_scatter_plotly({"train": {"P00001_neg_0": 0.9}}, scores_df, str(tmp_path / "path"),
                                  "Shannon Entropy", "TCR", donor_pattern="^(P\\d+)_")

    fig = mock_write.call_args.args[0]
    # One uncoloured dot per repertoire: the mean over its subsets, without a legend
    dots = [trace for trace in fig.data if trace.name != "Mean"]
    assert len(dots) == 1
    assert sorted(dots[0].y) == pytest.approx([0.2, 0.3, 0.4])
    assert dots[0].marker.color == "grey"
    assert not fig.layout.showlegend
    assert fig.layout.annotations[0].text == "Error bars: SD across repertoires (n=3); dots: repertoire means"
    # The tsv still has one row per generated subset
    assert len(pd.read_csv(tmp_path / "path.tsv", sep="\t")) == 6
