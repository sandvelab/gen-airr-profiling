import tempfile

import numpy as np
import pandas as pd
import pytest

from gen_airr_bm.analysis.analyse_network import (
    run_network_analysis,
    compute_and_plot_connectivity,
    calculate_degree_divergence_scores,
    get_node_degree_distributions,
    compute_connectivity_with_compairr,
    get_degrees_from_overlap,
    calculate_jsd,
    plot_connectivity_scores
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
        reference_data="test",
        n_subsets=5
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
    mock_get_files = mocker.patch('gen_airr_bm.analysis.analyse_network.get_sequence_files')
    mock_calc_scores = mocker.patch('gen_airr_bm.analysis.analyse_network.calculate_degree_divergence_scores')
    mock_plot = mocker.patch('gen_airr_bm.analysis.analyse_network.plot_connectivity_scores')

    # Mock the file structure
    mock_get_files.return_value = {
        "/path/to/ref1.tsv": ["/path/to/gen1_1.tsv", "/path/to/gen1_2.tsv"],
        "/path/to/ref2.tsv": ["/path/to/gen2_1.tsv", "/path/to/gen2_2.tsv"]
    }

    # Mock divergence scores
    mock_calc_scores.side_effect = [[0.1, 0.2], [0.3, 0.1], [0.5, 0.4], [0.2, 0.3]]

    compute_and_plot_connectivity(sample_analysis_config, "/tmp/compairr_output", "/tmp/compairr_helper")

    # Check that get_sequence_files was called for each model
    assert mock_get_files.call_count == 2

    # Check that calculate_degree_divergence_scores was called for each ref file and model
    assert mock_calc_scores.call_count == 4  # 2 models * 2 ref files

    # Check that plot_connectivity_scores was called for each dataset
    assert mock_plot.call_count == 2  # 2 datasets (ref1, ref2)


def test_calculate_degree_divergence_scores(mocker):
    """Test calculate_degree_divergence_scores computes scores correctly."""
    mock_get_dists = mocker.patch('gen_airr_bm.analysis.analyse_network.get_node_degree_distributions')
    mock_plot = mocker.patch('gen_airr_bm.analysis.analyse_network.plot_degree_distribution')
    mock_calc_jsd = mocker.patch('gen_airr_bm.analysis.analyse_network.calculate_jsd')

    # Mock degree distributions
    ref_dist = pd.Series([2, 1], index=[0, 1])
    gen_dists = [pd.Series([1, 2], index=[0, 1]), pd.Series([3, 1], index=[0, 1])]
    mock_get_dists.return_value = (ref_dist, gen_dists)

    # Mock JSD calculations
    mock_calc_jsd.side_effect = [[0.1], [0.2]]

    result = calculate_degree_divergence_scores(
        "/path/to/ref.tsv",
        ["/path/to/gen1.tsv", "/path/to/gen2.tsv"],
        "/tmp/helper",
        "/tmp/output",
        "model1",
        "test",
        "/tmp/analysis",
        "dataset1"
    )

    # Check that get_node_degree_distributions was called
    mock_get_dists.assert_called_once()

    # Check that plot_degree_distribution was called
    mock_plot.assert_called_once_with(
        ref_dist, gen_dists, "/tmp/analysis", "model1", "test", "dataset1"
    )

    # Check that calculate_jsd was called for each generated distribution
    assert mock_calc_jsd.call_count == 2

    # Check the result
    assert result == [0.1, 0.2]


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
        "test"
    )

    # Check that deduplicate_single_dataset was called
    mock_dedupe.assert_called_once_with(
        "/input/sequences.tsv",
        "/tmp/helper/sequences_test_unique.tsv"
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
            "test"
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
        "test"
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


def test_plot_connectivity_scores(mocker):
    """Test plot_connectivity_scores calls plot_avg_scores with correct arguments."""
    mock_plot_avg_scores = mocker.patch('gen_airr_bm.analysis.analyse_network.plot_avg_scores')

    mean_scores = {"model1": 0.5, "model2": 0.3}
    std_scores = {"model1": 0.1, "model2": 0.05}

    plot_connectivity_scores(
        mean_scores,
        std_scores,
        "/tmp/output",
        "test",
        "connectivity",
        "test_connectivity.png"
    )

    mock_plot_avg_scores.assert_called_once_with(
        mean_scores,
        std_scores,
        "/tmp/output",
        "test",
        "test_connectivity.png",
        "connectivity",
        scoring_method="JSD"
    )


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
        assert mock_get_files.call_count == 2  # Once per model
        assert mock_read_csv.call_count >= 2   # At least once per model
