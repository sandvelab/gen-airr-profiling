import tempfile

import numpy as np
import pandas as pd
import pytest

from gen_airr_bm.analysis.analyse_network import (
    run_network_analysis,
    compute_and_plot_connectivity,
    process_dataset,
    get_node_degree_distributions,
    get_reference_divergence_score,
    compute_connectivity_with_compairr,
    get_degrees_from_overlap,
    calculate_jsd,
    summarize_and_plot_dataset_connectivity,
    summarize_and_plot_all,
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
    mock_get_files = mocker.patch('gen_airr_bm.analysis.analyse_network.get_sequence_files')
    mock_process = mocker.patch('gen_airr_bm.analysis.analyse_network.process_dataset')
    mock_plot_dataset = mocker.patch('gen_airr_bm.analysis.analyse_network.summarize_and_plot_dataset_connectivity')
    mock_get_ref_score = mocker.patch('gen_airr_bm.analysis.analyse_network.get_reference_divergence_score',
                                      return_value=0.42)
    mock_plot_all = mocker.patch('gen_airr_bm.analysis.analyse_network.summarize_and_plot_all')

    # Mock the file structure
    mock_get_files.return_value = {
        "/path/to/ref1.tsv": ["/path/to/gen1_1.tsv", "/path/to/gen1_2.tsv"],
        "/path/to/ref2.tsv": ["/path/to/gen2_1.tsv", "/path/to/gen2_2.tsv"]
    }

    # Prepare process_dataset returns: one per (model, ref_file)
    def process_side_effect(ref_file, gen_files, helper_dir, output_dir, model_name, reference, analysis_output_dir):
        dataset_name = ref_file.split('/')[-1].replace('.tsv', '')
        # Return divergence_scores dict keyed by the current model
        return dataset_name, {model_name: [[0.1], [0.2]]}

    mock_process.side_effect = process_side_effect

    compute_and_plot_connectivity(sample_analysis_config, "/tmp/compairr_output",
                                  "/tmp/compairr_helper_dir")

    # get_sequence_files called once per model
    assert mock_get_files.call_count == len(sample_analysis_config.model_names) * len(sample_analysis_config.reference_data)
    # process_dataset called for each model x each ref dataset
    assert mock_process.call_count == len(sample_analysis_config.model_names) * len(sample_analysis_config.reference_data) * len(mock_get_files.return_value)

    # summarize_and_plot_dataset_connectivity called once per dataset (ref1, ref2) for the reference "test"
    assert mock_plot_dataset.call_count == len(mock_get_files.return_value) * len(sample_analysis_config.reference_data)

    # reference JSD computed and overall plot created
    mock_get_ref_score.assert_called_once()
    mock_plot_all.assert_called_once()
    # Ensure the reference score is passed through
    assert mock_plot_all.call_args.kwargs.get('mean_reference_score') or mock_plot_all.call_args.args[-1] == 0.42


def test_process_dataset(mocker):
    """Test process_dataset computes distributions, plots them, and returns divergence scores."""
    mock_get_dists = mocker.patch('gen_airr_bm.analysis.analyse_network.get_node_degree_distributions')
    mock_plot = mocker.patch('gen_airr_bm.analysis.analyse_network.plot_degree_distribution')
    mock_calc_jsd = mocker.patch('gen_airr_bm.analysis.analyse_network.calculate_jsd')

    # Mock degree distributions
    ref_dist = pd.Series([2, 1], index=[0, 1])
    gen_dists = [pd.Series([1, 2], index=[0, 1]), pd.Series([3, 1], index=[0, 1])]
    mock_get_dists.return_value = (ref_dist, gen_dists)

    # Mock JSD calculations (each returns a single-element list as implemented)
    mock_calc_jsd.side_effect = [[0.1], [0.2]]

    dataset_name, divergence_scores = process_dataset(
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
    # divergence_scores are lists of single-element lists per current calculate_jsd implementation
    assert divergence_scores == [0.1, 0.2]


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

    divergence_scores_by_reference = {
        "test": {
            "model1": [0.2, 0.4],
            "model2": [0.1, 0.3]
        },
        "train": {
            "model1": [0.5, 0.7],
            "model2": [0.6, 0.8]
        }
    }
    output_dir = "/tmp/out"
    reference_datasets = ["train", "test"]
    mean_reference_score = 0.21

    summarize_and_plot_all(
        divergence_scores_by_reference=divergence_scores_by_reference,
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


def test_get_reference_divergence_score(mocker, sample_analysis_config):
    """Test get_reference_divergence_score collects all ref1 scores and returns their mean."""
    mocker.patch(
        'gen_airr_bm.analysis.analyse_network.get_reference_files',
        return_value=[("trainA.tsv", ["testA.tsv"]), ("trainB.tsv", ["testB.tsv"])]
    )

    # process_dataset returns (dataset_name, divergence_scores)
    # divergence_scores must be list of floats
    def process_side_effect(train_file, test_file, helper_dir, output_dir, ref1, ref2, out_dir):
        ds_name = train_file.replace(".tsv", "")
        if train_file == "trainA.tsv":
            # First dataset contributes [0.1, 0.3]
            scores = [0.1, 0.3]
        elif train_file == "trainB.tsv":
            # Second dataset contributes [0.2, 0.4]
            scores = [0.2, 0.4]
        else:
            scores = []

        # divergence_scores -> list of floats
        return ds_name, scores

    mocker.patch(
        'gen_airr_bm.analysis.analyse_network.process_dataset',
        side_effect=process_side_effect
    )

    mean = get_reference_divergence_score(
        analysis_config=sample_analysis_config,
        compairr_output_helper_dir="/tmp/helper",
        compairr_output_dir="/tmp/output",
        train_ref="train",
        test_ref="test"
    )

    # Scores were [0.1, 0.3, 0.2, 0.4] => mean 0.25
    assert mean == pytest.approx(0.25, rel=1e-6)


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
        mocker.patch('gen_airr_bm.analysis.analyse_network.get_reference_divergence_score', return_value=0.25)
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
