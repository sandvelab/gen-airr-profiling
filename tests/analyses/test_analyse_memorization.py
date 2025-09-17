import os
import numpy as np
import pytest

from gen_airr_bm.analysis.analyse_memorization import (
    run_memorization_analysis,
    get_model_memorization_scores,
    get_reference_memorization_score,
    get_memorization_scores,
    plot_results,
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
        n_unique_samples=10
    )


def test_run_memorization_analysis(mocker, sample_analysis_config):
    # Mocks
    mock_makedirs = mocker.patch("os.makedirs")
    mock_get_model_scores = mocker.patch(
        "gen_airr_bm.analysis.analyse_memorization.get_model_memorization_scores",
        return_value={"model1": [0.1, 0.2], "model2": [0.3]}
    )
    mock_get_reference_score = mocker.patch(
        "gen_airr_bm.analysis.analyse_memorization.get_reference_memorization_score",
        return_value=0.123
    )
    mock_plot_results = mocker.patch(
        "gen_airr_bm.analysis.analyse_memorization.plot_results"
    )

    run_memorization_analysis(sample_analysis_config)

    # Directories creation
    mock_makedirs.assert_called_once_with("/tmp/test_output/analysis_mem", exist_ok=True)

    # Functions called with expected arguments
    mock_get_model_scores.assert_called_once_with(
        sample_analysis_config, "/tmp/test_output/analysis_mem", "train"
    )
    mock_get_reference_score.assert_called_once_with(
        sample_analysis_config, "/tmp/test_output/analysis_mem"
    )

    # Plot called with expected data
    mock_plot_results.assert_called_once_with(
        {"model1": [0.1, 0.2], "model2": [0.3]},
        0.123,
        "/tmp/test_output/analysis_mem",
        "memorization.png"
    )


def test_run_memorization_analysis_empty_reference(mocker, sample_analysis_config):
    mocker.patch("os.makedirs")

    # Case 1: Missing 'test'
    sample_analysis_config.reference_data = ["train"]
    with pytest.raises(ValueError):
        run_memorization_analysis(sample_analysis_config)

    # Case 2: Missing 'train'
    sample_analysis_config.reference_data = ["test"]
    with pytest.raises(ValueError):
        run_memorization_analysis(sample_analysis_config)


def test_get_model_memorization_scores(mocker, sample_analysis_config):
    # Prepare a deterministic mapping of reference->generated files
    comparison_mapping = {
        "/ref/ref1.tsv": ["/gen/g1.tsv", "/gen/g2.tsv"],
        "/ref/ref2.tsv": ["/gen/g3.tsv"]
    }

    mock_get_sequence_files = mocker.patch(
        "gen_airr_bm.analysis.analyse_memorization.get_sequence_files",
        return_value=comparison_mapping
    )

    # Side effect returns one value per generated file to match lengths
    def mem_scores_side_effect(ref_file, gen_files, output_dir, model_name):
        return [0.0] * len(gen_files)

    mock_get_mem_scores = mocker.patch(
        "gen_airr_bm.analysis.analyse_memorization.get_memorization_scores",
        side_effect=mem_scores_side_effect
    )

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

    # Each model accumulates one score per generated file
    expected_len = sum(len(v) for v in comparison_mapping.values())
    assert set(result.keys()) == set(sample_analysis_config.model_names)
    for model in sample_analysis_config.model_names:
        assert len(result[model]) == expected_len


def test_get_reference_memorization_score(mocker, sample_analysis_config):
    # Prepare train/test file pairs
    file_pairs = [
        ("/ref/train_1.tsv", "/ref/test_1.tsv"),
        ("/ref/train_2.tsv", "/ref/test_2.tsv"),
        ("/ref/train_3.tsv", "/ref/test_3.tsv"),
    ]

    mocker.patch(
        "gen_airr_bm.analysis.analyse_memorization.get_reference_files",
        return_value=file_pairs
    )

    # get_memorization_scores returns a list, we'll return one score to match usage
    seq = [[0.1], [0.3], [0.5]]
    mock_get_mem_scores = mocker.patch(
        "gen_airr_bm.analysis.analyse_memorization.get_memorization_scores",
        side_effect=seq
    )

    mean_score = get_reference_memorization_score(
        analysis_config=sample_analysis_config,
        output_dir="/tmp/test_output/analysis_mem"
    )

    # Assert calls
    assert mock_get_mem_scores.call_count == len(file_pairs)
    for i, (train_f, test_f) in enumerate(file_pairs):
        assert mock_get_mem_scores.call_args_list[i].args == (
            train_f, test_f, "/tmp/test_output/analysis_mem", "reference"
        )

    # Expected mean of the first (and only) elements per pair
    expected_mean = np.mean([s[0] for s in seq])
    assert mean_score == expected_mean


def test_get_memorization_scores_calls_compute(mocker):
    ref_file = "/ref/train.tsv"
    gen_files = ["/gen/a.tsv", "/gen/b.tsv", "/gen/c.tsv"]

    # Avoid real directory creation
    mock_makedirs = mocker.patch("os.makedirs")

    # Return increasing scores per call
    mock_compute = mocker.patch(
        "gen_airr_bm.analysis.analyse_memorization.compute_jaccard_similarity",
        side_effect=[0.11, 0.22, 0.33]
    )

    out = get_memorization_scores(
        ref_file=ref_file,
        gen_files=gen_files,
        output_dir="/tmp/test_output/analysis_mem",
        model_name="modelX"
    )

    # Helper dir created
    mock_makedirs.assert_called_once()
    helper_dir = f"/tmp/test_output/analysis_mem/compairr_helper_files"
    assert mock_makedirs.call_args.kwargs.get("exist_ok") is True

    # compute_jaccard_similarity called for each gen file
    assert mock_compute.call_count == len(gen_files)
    for i, gen in enumerate(gen_files):
        assert mock_compute.call_args_list[i].args == (
            helper_dir, ref_file, gen, "/tmp/test_output/analysis_mem", "modelX"
        )

    # Expected output
    assert out == [0.11, 0.22, 0.33]


def test_plot_results(mocker):
    # Patch Plotly Figure and Bar
    mock_fig = mocker.Mock()
    mock_Figure = mocker.patch("gen_airr_bm.analysis.analyse_memorization.go.Figure", return_value=mock_fig)
    mock_Bar = mocker.patch("gen_airr_bm.analysis.analyse_memorization.go.Bar")

    # Inputs
    model_scores = {
        "model2": [0.5, 0.7],    # mean=0.6, std=something
        "model1": [0.1, 0.2, 0.3]  # mean=0.2
    }
    mean_reference_score = 0.42
    fig_dir = "/tmp/out/mem"
    file_name = "mem_plot.png"

    # Execute
    plot_results(model_scores, mean_reference_score, fig_dir, file_name)

    # Figure constructed
    mock_Figure.assert_called_once()

    # Expected sorting by mean descending -> ["model2", "model1"]
    expected_models = ("model2", "model1")
    expected_scores = (np.mean([0.5, 0.7]), np.mean([0.1, 0.2, 0.3]))
    expected_errors = [np.std([0.5, 0.7]), np.std([0.1, 0.2, 0.3])]

    # Bar called with expected args
    assert mock_Bar.call_count == 1
    bar_kwargs = mock_Bar.call_args.kwargs
    assert tuple(bar_kwargs["x"]) == expected_models
    assert tuple(bar_kwargs["y"]) == expected_scores
    assert bar_kwargs["error_y"]["type"] == "data"
    assert bar_kwargs["error_y"]["visible"] is True
    assert pytest.approx(bar_kwargs["error_y"]["array"], rel=1e-9) == expected_errors
    assert bar_kwargs["marker"]["color"] == "skyblue"

    # Layout updated with expected labels and title
    assert mock_fig.update_layout.call_count == 1
    layout_kwargs = mock_fig.update_layout.call_args.kwargs
    assert "Average Memorization Scores Across Models" in layout_kwargs["title"]
    assert layout_kwargs["xaxis_title"] == "Models"
    assert layout_kwargs["yaxis_title"] == "Mean Jaccard Similarity"
    assert layout_kwargs["xaxis_tickangle"] == -45
    assert layout_kwargs["template"] == "plotly_white"

    # Reference line added
    assert mock_fig.add_hline.call_count == 1
    hline_kwargs = mock_fig.add_hline.call_args.kwargs
    assert hline_kwargs["y"] == mean_reference_score
    assert "reference=" in hline_kwargs["annotation_text"]
    assert f"{mean_reference_score:.3f}" in hline_kwargs["annotation_text"]

    # Saved to correct path
    mock_fig.write_image.assert_called_once_with(os.path.join(fig_dir, file_name))