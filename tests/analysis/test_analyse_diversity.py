import math
import os

import numpy as np
import pandas as pd
import pytest

from gen_airr_bm.analysis.analyse_diversity import run_diversity_analysis, compute_and_plot_diversity_scores, \
    compute_diversities_for_models, compute_diversity, shannon_entropy, gini_simpson_index, pielou_evenness, \
    gini_coefficient, plot_diversity_scatter_plotly
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
        n_subsets=5
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
        side_effect=lambda ref_dir, func: 0.5
    )
    mock_compute_diversities_for_models = mocker.patch(
        'gen_airr_bm.analysis.analyse_diversity.compute_diversities_for_models',
        return_value={
            "model1": {"dataset1_0": 0.1, "dataset1_1": 0.3},
            "model2": {"dataset1_0": 0.2, "dataset1_1": 0.4},
        }
    )
    mock_plot = mocker.patch(
        'gen_airr_bm.analysis.analyse_diversity.plot_diversity_scatter_plotly'
    )

    # Prepare inputs
    reference_dirs = {"test": "/some/fake/path"}
    output_path = "/fake/output/plot.png"
    metric_name = "Dummy Metric"

    compute_and_plot_diversity_scores(
        analysis_config=sample_analysis_config,
        reference_dirs=reference_dirs,
        output_path=output_path,
        diversity_function=dummy_diversity_func,
        metric_name=metric_name
    )

    # Check compute_diversity called correctly
    mock_compute_diversity.assert_called_once_with("/some/fake/path", dummy_diversity_func)

    # Check compute_diversities_for_models called with correct args
    expected_model_dir = "/tmp/test_output/generated_compairr_sequences_split"
    mock_compute_diversities_for_models.assert_called_once_with(
        ["model1", "model2"], expected_model_dir, dummy_diversity_func
    )

    # Check plot function call
    expected_reference_diversities = {"test": 0.5}
    expected_models_diversities_grouped = {
        "model1": {"dataset1": np.mean([0.1, 0.3])},
        "model2": {"dataset1": np.mean([0.2, 0.4])},
    }
    mock_plot.assert_called_once_with(
        expected_reference_diversities,
        expected_models_diversities_grouped,
        output_path,
        metric_name
    )


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


def test_plot_diversity_scatter_plotly(mocker, tmp_path):
    mock_fig = mocker.Mock()
    mock_px_scatter = mocker.patch("gen_airr_bm.analysis.analyse_diversity.px.scatter", return_value=mock_fig)

    reference_diversities = {
        "ref1": {"ds1": 0.1, "ds2": 0.2}
    }
    models_diversities = {
        "modelA": {"ds1": 0.15, "ds2": 0.25}
    }
    output_path = str(tmp_path / "path")
    metric_name = "Gini Coefficient"

    plot_diversity_scatter_plotly(reference_diversities, models_diversities, output_path, metric_name)

    expected_df = pd.DataFrame([
        {"dataset": "ds1", "gini coefficient": 0.1, "source": "ref1"},
        {"dataset": "ds2", "gini coefficient": 0.2, "source": "ref1"},
        {"dataset": "ds1", "gini coefficient": 0.15, "source": "modelA"},
        {"dataset": "ds2", "gini coefficient": 0.25, "source": "modelA"},
    ])
    actual_df = mock_px_scatter.call_args[0][0]

    pd.testing.assert_frame_equal(
        actual_df.reset_index(drop=True).sort_values(by=["source", "dataset"]).reset_index(drop=True),
        expected_df.sort_values(by=["source", "dataset"]).reset_index(drop=True)
    )

    mock_fig.write_image.assert_called_once_with(output_path + ".png")
