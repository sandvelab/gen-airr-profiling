import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from gen_airr_bm.analysis.analyse_network import (
    run_network_analysis,
    compute_and_plot_connectivity,
    get_connectivity_distributions_by_dataset,
    get_node_degree_distributions,
    get_mean_reference_divergence_score,
    compute_connectivity_with_compairr,
    get_degrees_from_overlap,
    calculate_jsd,
    summarize_and_plot_dataset_connectivity,
    summarize_and_plot_all, calculate_divergence_scores,
)
from gen_airr_bm.core.analysis_config import AnalysisConfig


@pytest.fixture
def sample_analysis_config():
    """Create a sample AnalysisConfig for testing."""
    return AnalysisConfig(
        analysis="network",
        model_names=["model1", "model2"],
        analysis_output_dir="/tmp/test_output",
        root_output_dir="/tmp/test_root",
        default_model_name="humanTRB",
        reference_data=["train", "test"],
        n_subsets=5,
        n_unique_samples=10
    )


@pytest.fixture
def sample_compairr_result():
    """Create a sample Compairr result DataFrame."""
    return pd.DataFrame({
        'sequence_id': ['seq1', 'seq2', 'seq3', 'seq4'],
        'overlap_count': [3, 2, 4, 1]
    })


@pytest.fixture
def sample_degree_distribution():
    """Create a sample degree distribution Series."""
    return pd.Series([2, 1, 1], index=[0, 1, 2], name='count')


def test_run_network_analysis(mocker, sample_analysis_config):
    """Test that run_network_analysis creates necessary directories and calls compute_and_plot_connectivity."""
    mock_makedirs = mocker.patch('os.makedirs')
    mock_compute = mocker.patch('gen_airr_bm.analysis.analyse_network.compute_and_plot_connectivity')

    run_network_analysis(sample_analysis_config)

    # Check that makedirs was called for each directory
    expected_calls = [
        (("/tmp/test_output",), {"exist_ok": True}),
        (("/tmp/test_output/compairr_helper_files",), {"exist_ok": True}),
        (("/tmp/test_output/compairr_output",), {"exist_ok": True})
    ]

    # Verify that makedirs was called with the expected arguments
    assert mock_makedirs.call_count == 3
    for call in expected_calls:
        assert call in [(args, kwargs) for args, kwargs in mock_makedirs.call_args_list]

    # Check that compute_and_plot_connectivity was called with correct arguments
    mock_compute.assert_called_once_with(
        sample_analysis_config,
        "/tmp/test_output/compairr_output",
        "/tmp/test_output/compairr_helper_files"
    )


def test_compute_and_plot_connectivity(mocker, sample_analysis_config):
    """Test compute_and_plot_connectivity processes files correctly."""
    mock_get_files = mocker.patch("gen_airr_bm.analysis.analyse_network.get_sequence_files")
    mock_gcd = mocker.patch("gen_airr_bm.analysis.analyse_network.get_connectivity_distributions_by_dataset")
    mock_calc_div = mocker.patch("gen_airr_bm.analysis.analyse_network.calculate_divergence_scores")
    mock_plot_dataset = mocker.patch("gen_airr_bm.analysis.analyse_network.summarize_and_plot_dataset_connectivity")
    mock_get_ref_score = mocker.patch(
        "gen_airr_bm.analysis.analyse_network.get_mean_reference_divergence_score", return_value=0.42
    )
    mock_plot_all = mocker.patch("gen_airr_bm.analysis.analyse_network.summarize_and_plot_all")

    # Mock the file structure returned for each model/reference
    mock_get_files.return_value = {
        "/path/to/ref1.tsv": ["/path/to/gen1_1.tsv", "/path/to/gen1_2.tsv"],
        "/path/to/ref2.tsv": ["/path/to/gen2_1.tsv", "/path/to/gen2_2.tsv"],
    }

    # Fake degree distributions returned by get_connectivity_distributions_by_dataset
    ref_degree_dist = pd.Series([1, 2], index=[0, 1], name="ref")
    gen_degree_dists = [pd.Series([2, 1], index=[0, 1], name="genA"),
                        pd.Series([0, 1], index=[1, 2], name="genB")]

    def gcd_side_effect(ref_file, gen_files, helper_dir, output_dir, model_name, reference, analysis_output_dir):
        # Mimic dataset_name derivation in implementation: basename without extension
        dataset_name = os.path.splitext(os.path.basename(ref_file))[0]
        # Return the proper 3-tuple
        return dataset_name, ref_degree_dist, gen_degree_dists

    mock_gcd.side_effect = gcd_side_effect

    # calculate_divergence_scores_per_dataset must accept (pd.Series, list[pd.Series]) and return list[float]
    def calc_div_side_effect(ref_dd, gen_dds):
        assert ref_dd is ref_degree_dist
        assert gen_dds == gen_degree_dists
        return [0.1, 0.2]

    mock_calc_div.side_effect = calc_div_side_effect

    # Act
    compute_and_plot_connectivity(
        sample_analysis_config, "/tmp/compairr_output", "/tmp/compairr_helper_dir"
    )

    # Assert calls
    # Called once per (reference, model)
    assert mock_get_files.call_count == len(sample_analysis_config.reference_data) * len(
        sample_analysis_config.model_names)

    # For each (reference, model), two datasets, so calc_div called 2 * refs * models
    expected_calc_div_calls = 2 * len(sample_analysis_config.reference_data) * len(sample_analysis_config.model_names)
    assert mock_calc_div.call_count == expected_calc_div_calls

    # summarize_and_plot_dataset_connectivity called once per dataset per reference (after aggregating across models)
    expected_plot_dataset_calls = 2 * len(sample_analysis_config.reference_data)  # two datasets: ref1, ref2
    assert mock_plot_dataset.call_count == expected_plot_dataset_calls

    # Verify per-call arguments for dataset plotting
    called_refs = set()
    called_datasets = []
    for call in mock_plot_dataset.call_args_list:
        ds_name, divergence_scores, out_dir, reference = call.args
        called_refs.add(reference)
        called_datasets.append(ds_name)

        # divergence_scores is a dict: {model_name: list[float]}
        # Each model should have collected [0.1, 0.2]
        assert set(divergence_scores.keys()) == set(sample_analysis_config.model_names)
        for model in sample_analysis_config.model_names:
            assert divergence_scores[model] == [0.1, 0.2]

        # Output dir matches config
        assert out_dir == sample_analysis_config.analysis_output_dir

        # Dataset names should be derived from the ref file names: "ref1" and "ref2"
        assert ds_name in {"ref1", "ref2"}

    # Both references should have been processed
    assert called_refs == set(sample_analysis_config.reference_data)
    # Both datasets should have been plotted for each reference (order not guaranteed)
    assert {"ref1", "ref2"}.issubset(set(called_datasets))

    # Overall summary plotted once
    assert mock_get_ref_score.call_count == 1
    mock_plot_all.assert_called_once()
    all_args = mock_plot_all.call_args[0]
    divergence_scores_all, analysis_output_dir, references, mean_ref_score = all_args

    assert analysis_output_dir == sample_analysis_config.analysis_output_dir
    assert references == sample_analysis_config.reference_data
    assert mean_ref_score == 0.42

    # Light structure checks on divergence_scores_all
    # Expect entries for both references and both datasets with both models
    for reference in sample_analysis_config.reference_data:
        assert set(divergence_scores_all[reference].keys()) == {"ref1", "ref2"}
        for ds in ["ref1", "ref2"]:
            for model in sample_analysis_config.model_names:
                assert divergence_scores_all[reference][ds][model] == [0.1, 0.2]


def test_get_connectivity_distributions_by_dataset(mocker):
    """Test get_connectivity_distributions_by_dataset computes distributions and plots them."""
    mock_get_dists = mocker.patch('gen_airr_bm.analysis.analyse_network.get_node_degree_distributions')
    mock_plot = mocker.patch('gen_airr_bm.analysis.analyse_network.plot_degree_distribution')

    # Mock degree distributions
    ref_dist = pd.Series([2, 1], index=[0, 1])
    gen_dists = [pd.Series([1, 2], index=[0, 1]), pd.Series([3, 1], index=[0, 1])]
    mock_get_dists.return_value = (ref_dist, gen_dists)

    dataset_name, ref1_degree_dist, gen_degree_dists = get_connectivity_distributions_by_dataset(
        "/path/to/dataset1.tsv",
        ["/path/to/gen1.tsv", "/path/to/gen2.tsv"],
        "/tmp/helper",
        "/tmp/output",
        "model1",
        "test",
        "/tmp/analysis"
    )

    mock_get_dists.assert_called_once()
    mock_plot.assert_called_once_with(ref_dist, gen_dists, "/tmp/analysis", "model1", "test", "dataset1")
    assert dataset_name == "dataset1"


def test_calculate_divergence_scores(mocker):
    """Test calculate_divergence_scores_per_dataset computes JSD scores correctly."""
    mock_calc_jsd = mocker.patch('gen_airr_bm.analysis.analyse_network.calculate_jsd')

    mock_calc_jsd.side_effect = [[0.1], [0.3], [0.2]]

    ref_dist = pd.Series([2, 1], index=[0, 1])
    gen_dists = [
        pd.Series([1, 2], index=[0, 1]),
        pd.Series([3, 1], index=[0, 1]),
        pd.Series([2, 2], index=[0, 1])
    ]

    scores = calculate_divergence_scores(ref_dist, gen_dists)

    assert scores == [0.1, 0.3, 0.2]
    assert mock_calc_jsd.call_count == 3


def test_get_node_degree_distributions(mocker):
    """Test get_node_degree_distributions processes files correctly."""
    mock_compute_conn = mocker.patch('gen_airr_bm.analysis.analyse_network.compute_connectivity_with_compairr')
    mock_get_degrees = mocker.patch('gen_airr_bm.analysis.analyse_network.get_degrees_from_overlap')

    # Mock connectivity results
    mock_compute_conn.side_effect = [
        pd.DataFrame({'sequence_id': ['seq1'], 'overlap_count': [2]}),  # gen1
        pd.DataFrame({'sequence_id': ['seq2'], 'overlap_count': [3]}),  # gen2
        pd.DataFrame({'sequence_id': ['seq3'], 'overlap_count': [1]})   # ref
    ]

    # Mock degree distributions
    mock_get_degrees.side_effect = [
        pd.Series([1], index=[1]),  # gen1
        pd.Series([1], index=[2]),  # gen2
        pd.Series([1], index=[0])   # ref
    ]

    ref_dist, gen_dists = get_node_degree_distributions(
        "/path/to/ref.tsv",
        ["/path/to/gen1.tsv", "/path/to/gen2.tsv"],
        "/tmp/helper",
        "/tmp/output",
        "model1",
        "test"
    )

    # Check that compute_connectivity_with_compairr was called 3 times
    assert mock_compute_conn.call_count == 3

    # Check that get_degrees_from_overlap was called 3 times
    assert mock_get_degrees.call_count == 3

    # Check the results
    assert len(gen_dists) == 2
    pd.testing.assert_series_equal(ref_dist, pd.Series([1], index=[0]))


def test_compute_connectivity_with_compairr_valid_new_file(mocker):
    """Test compute_connectivity_with_compairr with a new file."""
    mock_exists = mocker.patch('os.path.exists')
    mock_read_csv = mocker.patch('pandas.read_csv')
    mock_dedupe = mocker.patch('gen_airr_bm.analysis.analyse_network.deduplicate_single_dataset')
    mock_compairr = mocker.patch('gen_airr_bm.analysis.analyse_network.run_compairr_existence')

    # Mock file existence checks
    mock_exists.side_effect = lambda path: path == "/input/sequences.tsv"

    # Mock CSV reading
    expected_df = pd.DataFrame({
        'sequence_id': ['seq1', 'seq2'],
        'overlap_count': [2, 3]
    })
    mock_read_csv.return_value = expected_df

    result = compute_connectivity_with_compairr(
        "/input/sequences.tsv",
        "/tmp/helper",
        "/tmp/output",
        "test",
    )

    # Check that deduplicate_single_dataset was called
    mock_dedupe.assert_called_once_with(
        "/input/sequences.tsv",
        "/tmp/helper/sequences_test_unique.tsv",
    )

    # Check that run_compairr_existence was called
    mock_compairr.assert_called_once()

    # Check that pandas.read_csv was called
    mock_read_csv.assert_called_once_with(
        "/tmp/output/sequences_test_overlap.tsv",
        sep='\t',
        names=['sequence_id', 'overlap_count'],
        header=0
    )

    # Check the result
    pd.testing.assert_frame_equal(result, expected_df)


def test_compute_connectivity_with_compairr_invalid_file_not_found(mocker):
    """Test compute_connectivity_with_compairr raises error for missing file."""
    mock_exists = mocker.patch('os.path.exists')
    mock_exists.return_value = False

    with pytest.raises(FileNotFoundError, match="Input sequences file not found"):
        compute_connectivity_with_compairr(
            "/nonexistent/file.tsv",
            "/tmp/helper",
            "/tmp/output",
            "test",
        )


def test_compute_connectivity_with_compairr_valid_existing_file(mocker):
    """Test compute_connectivity_with_compairr skips deduplication for existing unique file."""
    mock_exists = mocker.patch('os.path.exists')
    mock_read_csv = mocker.patch('pandas.read_csv')
    mock_compairr = mocker.patch('gen_airr_bm.analysis.analyse_network.run_compairr_existence')
    mock_dedupe = mocker.patch('gen_airr_bm.analysis.analyse_network.deduplicate_single_dataset')

    # Mock file existence - input exists, unique file exists
    def exists_side_effect(path):
        return path in ["/input/sequences.tsv", "/tmp/helper/sequences_test_unique.tsv"]
    mock_exists.side_effect = exists_side_effect

    expected_df = pd.DataFrame({
        'sequence_id': ['seq1'],
        'overlap_count': [1]
    })
    mock_read_csv.return_value = expected_df

    result = compute_connectivity_with_compairr(
        "/input/sequences.tsv",
        "/tmp/helper",
        "/tmp/output",
        "test",
    )

    # Check that deduplicate_single_dataset was NOT called
    mock_dedupe.assert_not_called()

    # Check that run_compairr_existence was still called
    mock_compairr.assert_called_once()


def test_get_degrees_from_overlap_valid(sample_compairr_result):
    """Test get_degrees_from_overlap with normal input."""
    result = get_degrees_from_overlap(sample_compairr_result)

    # Expected: overlap_count - 1 = [2, 1, 3, 0]
    # value_counts: {0: 1, 1: 1, 2: 1, 3: 1}
    expected = pd.Series([1, 1, 1, 1], index=[0, 1, 2, 3])

    # Sort both series by index for comparison
    result_sorted = result.sort_index()
    expected_sorted = expected.sort_index()

    pd.testing.assert_series_equal(result_sorted, expected_sorted, check_names=False)


def test_get_degrees_from_overlap_invalid_missing_column():
    """Test get_degrees_from_overlap raises error for missing overlap_count column."""
    df = pd.DataFrame({'sequence_id': ['seq1', 'seq2']})

    with pytest.raises(ValueError, match="Compairr result DataFrame must contain 'overlap_count' column"):
        get_degrees_from_overlap(df)


def test_get_degrees_from_overlap_valid_empty_dataframe():
    """Test get_degrees_from_overlap with empty DataFrame."""
    df = pd.DataFrame({'sequence_id': [], 'overlap_count': []})
    result = get_degrees_from_overlap(df)

    assert len(result) == 0
    assert isinstance(result, pd.Series)


def test_calculate_jsd_identical_distributions():
    """Test calculate_jsd with identical distributions."""
    dist1 = pd.Series([2, 1], index=[0, 1], name='count')
    dist2 = pd.Series([2, 1], index=[0, 1], name='count')

    result = calculate_jsd(dist1, dist2)

    assert len(result) == 1
    assert result[0] == pytest.approx(0.0, abs=1e-10)


def test_calculate_jsd_different_distributions():
    """Test calculate_jsd with different distributions."""
    dist1 = pd.Series([3, 0], index=[0, 1], name='count')
    dist2 = pd.Series([0, 3], index=[0, 1], name='count')

    result = calculate_jsd(dist1, dist2)

    assert len(result) == 1
    assert 0 < result[0] <= 1  # JSD should be between 0 and 1


def test_calculate_jsd_different_indices():
    """Test calculate_jsd with distributions having different indices."""
    dist1 = pd.Series([2], index=[0], name='count')
    dist2 = pd.Series([2], index=[1], name='count')

    result = calculate_jsd(dist1, dist2)

    assert len(result) == 1
    assert 0 < result[0] <= 1


def test_calculate_jsd_empty_distributions():
    """Test calculate_jsd with empty distributions."""
    dist1 = pd.Series([], dtype=int, name='count')
    dist2 = pd.Series([], dtype=int, name='count')

    result = calculate_jsd(dist1, dist2)

    assert len(result) == 1
    # JSD of empty distributions should be NaN or 0
    assert np.isnan(result[0]) or result[0] == 0


def test_summarize_and_plot_dataset_connectivity(mocker):
    """Test summarize_and_plot_dataset_connectivity calls plot_avg_scores with computed stats."""
    mock_plot_avg_scores = mocker.patch('gen_airr_bm.analysis.analyse_network.plot_avg_scores')

    # Provide raw divergence scores so function computes mean/std
    divergence_scores = {"model1": [0.5, 0.5], "model2": [0.3, 0.3]}

    summarize_and_plot_dataset_connectivity(
        "test_dataset",
        divergence_scores,
        "/tmp/output",
        "test",
    )

    expected_mean = {"model1": 0.5, "model2": 0.3}
    expected_std = {"model1": 0.0, "model2": 0.0}

    mock_plot_avg_scores.assert_called_once_with(
        mean_scores_dict=expected_mean,
        std_scores_dict=expected_std,
        output_dir="/tmp/output",
        reference_data="test",
        distribution_type="connectivity",
        file_name="test_dataset_connectivity.png",
    )


def test_summarize_and_plot_all(mocker):
    """Test summarize_and_plot_all computes mean/std per reference and calls plot_grouped_avg_scores."""
    mock_plot_grouped = mocker.patch('gen_airr_bm.analysis.analyse_network.plot_grouped_avg_scores')

    divergence_scores_all = {
        "test": {
            "dataset1": {"model1": [0.2, 0.4], "model2": [0.1, 0.3]},
            "dataset2": {"model1": [0.4, 0.2], "model2": [0.3, 0.1]}
        },
        "train": {
            "dataset1": {"model1": [0.5, 0.7], "model2": [0.6, 0.8]},
            "dataset2": {"model1": [0.7, 0.5], "model2": [0.8, 0.6]}
        }
    }
    output_dir = "/tmp/out"
    reference_datasets = ["train", "test"]
    mean_reference_score = 0.21

    summarize_and_plot_all(
        divergence_scores_all=divergence_scores_all,
        output_dir=output_dir,
        reference_datasets=reference_datasets,
        mean_reference_score=mean_reference_score
    )

    assert mock_plot_grouped.call_count == 1
    kwargs = mock_plot_grouped.call_args.kwargs

    expected_mean = {
        "test": {"model1": 0.3, "model2": 0.2},
        "train": {"model1": 0.6, "model2": 0.7}
    }
    expected_std = {
        "test": {"model1": 0.1, "model2": 0.1},
        "train": {"model1": 0.1, "model2": 0.1}
    }

    # Compare nested float dicts using approx to avoid rounding issues
    mean_scores = kwargs["mean_scores_by_ref"]
    std_scores = kwargs["std_scores_by_ref"]
    for ref, models in expected_mean.items():
        for model, exp_val in models.items():
            assert mean_scores[ref][model] == pytest.approx(exp_val, rel=1e-12, abs=1e-12)

    for ref, models in expected_std.items():
        for model, exp_val in models.items():
            assert std_scores[ref][model] == pytest.approx(exp_val, rel=1e-12, abs=1e-12)

    # Non-float kwargs can be compared directly
    assert kwargs["output_dir"] == output_dir
    assert kwargs["reference_data"] == reference_datasets
    assert kwargs["distribution_type"] == "connectivity"
    assert kwargs["file_name"] == "all_datasets_connectivity.png"
    assert kwargs["scoring_method"] == "JSD"
    assert kwargs["reference_score"] == mean_reference_score


def test_get_mean_reference_divergence_score(mocker, sample_analysis_config):
    """Test get_reference_divergence_score collects all ref1 scores and returns their mean."""
    mocker.patch(
        'gen_airr_bm.analysis.analyse_network.get_reference_files',
        return_value=[("trainA.tsv", "testA.tsv"), ("trainB.tsv", "testB.tsv")]
    )

    side_effect_gcd = [
        ("datasetA", pd.Series([1, 2]), [pd.Series([2, 1])]),
        ("datasetB", pd.Series([3, 4]), [pd.Series([4, 3])]),
    ]
    mocker.patch(
        "gen_airr_bm.analysis.analyse_network.get_connectivity_distributions_by_dataset",
        side_effect=side_effect_gcd,
    )

    # Mock calculate_divergence_scores_per_dataset with the CORRECT signature:
    # (ref1_degree_dist: pd.Series, ref2_or_gen_degree_dists: list[pd.Series]) -> list[float]
    mocker.patch(
        "gen_airr_bm.analysis.analyse_network.calculate_divergence_scores",
        side_effect=[[0.1, 0.3], [0.2, 0.4]],  # one list per dataset
    )

    mean = get_mean_reference_divergence_score(
        analysis_config=sample_analysis_config,
        compairr_output_helper_dir="/tmp/helper",
        compairr_output_dir="/tmp/output",
    )

    # Mean of [0.1, 0.3, 0.2, 0.4] = 0.25
    assert mean == pytest.approx(0.25)


def test_end_to_end_workflow_with_mocks(mocker, sample_analysis_config):
    """Test the complete workflow with mocked external dependencies."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Update config to use temp directory
        sample_analysis_config.analysis_output_dir = temp_dir

        mock_get_files = mocker.patch('gen_airr_bm.analysis.analyse_network.get_sequence_files')
        mocker.patch('gen_airr_bm.analysis.analyse_network.deduplicate_single_dataset')
        mocker.patch('gen_airr_bm.analysis.analyse_network.run_compairr_existence')
        mock_read_csv = mocker.patch('pandas.read_csv')
        mocker.patch('gen_airr_bm.analysis.analyse_network.plot_degree_distribution')
        mocker.patch('gen_airr_bm.analysis.analyse_network.plot_avg_scores')
        mocker.patch('gen_airr_bm.analysis.analyse_network.calculate_jsd', return_value=[0.1])
        mocker.patch('gen_airr_bm.analysis.analyse_network.get_mean_reference_divergence_score', return_value=0.25)
        mocker.patch('gen_airr_bm.analysis.analyse_network.summarize_and_plot_all')
        mocker.patch('os.path.exists', return_value=True)

        # Mock file structure
        mock_get_files.return_value = {
            "/ref/dataset1.tsv": ["/gen/dataset1_gen1.tsv"]
        }

        # Mock Compairr results
        mock_read_csv.return_value = pd.DataFrame({
            'sequence_id': ['seq1', 'seq2', 'seq3'],
            'overlap_count': [2, 3, 1]
        })

        # This should run without errors
        run_network_analysis(sample_analysis_config)

        # Verify that the main functions were called
        assert mock_get_files.call_count == 2 * 2  # Once per model and reference dataset
        assert mock_read_csv.call_count >= 2   # At least once per model
