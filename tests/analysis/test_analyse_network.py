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
    get_reference_divergence_scores,
    compute_connectivity_with_compairr,
    get_node_degree_from_compairr_output,
    calculate_jsd,
    summarize_and_plot_dataset_connectivity,
    summarize_and_plot_all, calculate_divergence_scores,
    get_subset_index,
    run_network_statistics,
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
        subfolder_name="analysis_subfolder",
        receptor_type="TCR",
        indels=True,
        allowed_mismatches=1
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
    reference_scores = pd.DataFrame({"repertoire": ["ref1", "ref2"], "score": [0.4, 0.44]})
    mock_get_ref_score = mocker.patch(
        "gen_airr_bm.analysis.analyse_network.get_reference_divergence_scores", return_value=reference_scores
    )
    mock_plot_all = mocker.patch("gen_airr_bm.analysis.analyse_network.summarize_and_plot_all")
    mock_statistics = mocker.patch("gen_airr_bm.analysis.analyse_network.run_network_statistics")
    mocker.patch("pandas.DataFrame.to_csv")

    # Mock the file structure returned for each model/reference
    mock_get_files.return_value = {
        "/path/to/ref1.tsv": ["/path/to/gen1_1.tsv", "/path/to/gen1_2.tsv"],
        "/path/to/ref2.tsv": ["/path/to/gen2_1.tsv", "/path/to/gen2_2.tsv"],
    }

    # Fake degree distributions returned by get_connectivity_distributions_by_dataset
    ref_degree_dist = pd.Series([1, 2], index=[0, 1], name="ref")
    gen_degree_dists = [pd.Series([2, 1], index=[0, 1], name="genA"),
                        pd.Series([0, 1], index=[1, 2], name="genB")]

    def gcd_side_effect(ref_file, gen_files, helper_dir, output_dir, model_name, reference, allowed_mismatches, indels):
        # Mimic dataset_name derivation in implementation: basename without extension
        dataset_name = os.path.splitext(os.path.basename(ref_file))[0]
        # Return the proper 3-tuple
        return dataset_name, ref_degree_dist, gen_degree_dists

    mock_gcd.side_effect = gcd_side_effect

    # calculate_divergence_scores_per_dataset must accept (pd.Series, list[pd.Series]) and return list[float]
    def calc_div_side_effect(ref_dd, gen_dds, ref_name="ref", gen_name="gen"):
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

    # # summarize_and_plot_dataset_connectivity called once per dataset per reference (after aggregating across models)
    expected_plot_dataset_calls = 2 * len(sample_analysis_config.reference_data)  # two datasets: ref1, ref2
    assert mock_plot_dataset.call_count == expected_plot_dataset_calls

    # Verify per-call arguments for dataset plotting
    called_refs = set()
    called_datasets = []
    for call in mock_plot_dataset.call_args_list:
        # summarize_and_plot_dataset_connectivity(dataset_name, divergence_scores, analysis_config.analysis_output_dir, reference)
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

    # Overall summary plotted once, from one row per subset, with the per-repertoire reference scores
    assert mock_get_ref_score.call_count == 1
    mock_plot_all.assert_called_once()
    _, scores_df, plotted_reference_scores = mock_plot_all.call_args[0]
    assert plotted_reference_scores is reference_scores
    mock_statistics.assert_called_once()

    # 2 references x 2 models x 2 datasets x 2 subsets, with the subset index from the generated file name
    assert len(scores_df) == 16
    assert set(scores_df["dataset"]) == {"ref1", "ref2"}
    assert set(scores_df["subset"]) == {1, 2}
    assert scores_df.groupby("subset")["jsd"].first().to_dict() == {1: 0.1, 2: 0.2}


def test_get_connectivity_distributions_by_dataset(mocker, sample_analysis_config):
    """Test get_connectivity_distributions_by_dataset computes distributions and plots them."""
    mock_get_dists = mocker.patch('gen_airr_bm.analysis.analyse_network.get_node_degree_distributions')

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
        sample_analysis_config.allowed_mismatches,
        sample_analysis_config.indels
    )

    mock_get_dists.assert_called_once()
    assert dataset_name == "dataset1"

    # get_node_degree_distributions should have been called with the exact positional args
    mock_get_dists.assert_called_once_with(
        "/path/to/dataset1.tsv",
        ["/path/to/gen1.tsv", "/path/to/gen2.tsv"],
        "/tmp/helper",
        "/tmp/output",
        "model1",
        "test",
        sample_analysis_config.allowed_mismatches,
        sample_analysis_config.indels
    )

    # The returned reference distribution should match the mocked one
    pd.testing.assert_series_equal(ref1_degree_dist, ref_dist)

    # The returned generated distributions should be a list of Series matching the mocked ones
    assert isinstance(gen_degree_dists, list)
    assert len(gen_degree_dists) == len(gen_dists)
    for returned, expected in zip(gen_degree_dists, gen_dists):
        pd.testing.assert_series_equal(returned, expected)


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

    scores = calculate_divergence_scores(ref_dist, gen_dists, dist1_name="ref", dist2_name="gen")

    assert scores == [0.1, 0.3, 0.2]
    assert mock_calc_jsd.call_count == 3


def test_get_node_degree_distributions(mocker):
    """Test get_node_degree_distributions processes files correctly."""
    mock_compute_conn = mocker.patch('gen_airr_bm.analysis.analyse_network.compute_connectivity_with_compairr')
    mock_get_degrees = mocker.patch('gen_airr_bm.analysis.analyse_network.get_node_degree_from_compairr_output')

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
        "test",
        1,
        True
    )

    # Check that compute_connectivity_with_compairr was called 3 times
    assert mock_compute_conn.call_count == 3

    # Check that get_node_degree_from_compairr_output was called 3 times
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
        1,
        True
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
            1,
            True
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
        1,
        True
    )

    # Check that deduplicate_single_dataset was NOT called
    mock_dedupe.assert_not_called()

    # Check that run_compairr_existence was still called
    mock_compairr.assert_called_once()


def test_get_node_degree_from_compairr_output_valid(sample_compairr_result):
    """Test get_node_degree_from_compairr_output with normal input."""
    result = get_node_degree_from_compairr_output(sample_compairr_result)

    # Expected: overlap_count - 1 = [2, 1, 3, 0]
    # value_counts: {0: 1, 1: 1, 2: 1, 3: 1}
    expected = pd.Series([1, 1, 1, 1], index=[0, 1, 2, 3])

    # Sort both series by index for comparison
    result_sorted = result.sort_index()
    expected_sorted = expected.sort_index()

    pd.testing.assert_series_equal(result_sorted, expected_sorted, check_names=False)


def test_get_node_degree_from_compairr_output_invalid_missing_column():
    """Test get_node_degree_from_compairr_output raises error for missing overlap_count column."""
    df = pd.DataFrame({'sequence_id': ['seq1', 'seq2']})

    with pytest.raises(ValueError, match="Compairr result DataFrame must contain 'overlap_count' column"):
        get_node_degree_from_compairr_output(df)


def test_get_node_degree_from_compairr_output_valid_empty_dataframe():
    """Test get_node_degree_from_compairr_output with empty DataFrame."""
    df = pd.DataFrame({'sequence_id': [], 'overlap_count': []})
    result = get_node_degree_from_compairr_output(df)

    assert len(result) == 0
    assert isinstance(result, pd.Series)


def test_calculate_jsd_identical_distributions():
    """Test calculate_jsd with identical distributions."""
    dist1 = pd.Series([2, 1], index=[0, 1], name='count')
    dist2 = pd.Series([2, 1], index=[0, 1], name='count')

    result = calculate_jsd(dist1, dist2, dist1_name="dist1", dist2_name="dist2")

    assert len(result) == 1
    assert result[0] == pytest.approx(0.0, abs=1e-10)


def test_calculate_jsd_different_distributions():
    """Test calculate_jsd with different distributions."""
    dist1 = pd.Series([3, 0], index=[0, 1], name='count')
    dist2 = pd.Series([0, 3], index=[0, 1], name='count')

    result = calculate_jsd(dist1, dist2, dist1_name="dist1", dist2_name="dist2")

    assert len(result) == 1
    assert 0 < result[0] <= 1  # JSD should be between 0 and 1


def test_calculate_jsd_different_indices():
    """Test calculate_jsd with distributions having different indices."""
    dist1 = pd.Series([2], index=[0], name='count')
    dist2 = pd.Series([2], index=[1], name='count')

    result = calculate_jsd(dist1, dist2, dist1_name="dist1", dist2_name="dist2")

    assert len(result) == 1
    assert 0 < result[0] <= 1


def test_calculate_jsd_empty_distributions():
    """Test calculate_jsd with empty distributions."""
    dist1 = pd.Series([], dtype=int, name='count')
    dist2 = pd.Series([], dtype=int, name='count')

    result = calculate_jsd(dist1, dist2, dist1_name="dist1", dist2_name="dist2")

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
        file_name="test_dataset_connectivity",
    )


def make_scores_df(scores: dict) -> pd.DataFrame:
    """ Scores as {(reference, dataset, model): [score per subset]}. """
    return pd.DataFrame([{"metric": "connectivity", "model": model, "reference": reference, "dataset": dataset,
                          "subset": subset, "jsd": score}
                         for (reference, dataset, model), subset_scores in scores.items()
                         for subset, score in enumerate(subset_scores)])


def test_summarize_and_plot_all(mocker, sample_analysis_config):
    """Test summarize_and_plot_all averages the subsets within each repertoire before summarising over repertoires."""
    mock_plot_grouped = mocker.patch('gen_airr_bm.analysis.analyse_network.plot_grouped_avg_scores')
    scores_df = make_scores_df({
        ("test", "dataset1", "model1"): [0.2, 0.4], ("test", "dataset2", "model1"): [0.5, 0.7],
        ("train", "dataset1", "model1"): [0.5, 0.7], ("train", "dataset2", "model1"): [0.8, 0.6],
    })
    reference_scores = pd.DataFrame({"repertoire": ["dataset1", "dataset2"], "score": [0.2, 0.22]})

    summarize_and_plot_all(sample_analysis_config, scores_df, reference_scores)

    assert mock_plot_grouped.call_count == 1
    kwargs = mock_plot_grouped.call_args.kwargs
    # Repertoire means: test 0.3 and 0.6, train 0.6 and 0.7
    assert kwargs["mean_scores_by_ref"]["test"]["model1"] == pytest.approx(0.45)
    assert kwargs["mean_scores_by_ref"]["train"]["model1"] == pytest.approx(0.65)
    # The error bars are the SD across the repertoire means, not across all subsets
    assert kwargs["std_scores_by_ref"]["test"]["model1"] == pytest.approx(np.std([0.3, 0.6], ddof=1))
    assert kwargs["repertoire_scores_by_ref"]["test"]["model1"] == [("dataset1", pytest.approx(0.3)),
                                                                    ("dataset2", pytest.approx(0.6))]
    assert kwargs["error_bar_caption"].startswith("Error bars: SD across repertoires (n=2)")
    assert kwargs["distribution_type"] == "connectivity"
    assert kwargs["file_name"] == "all_datasets_connectivity"
    assert kwargs["scoring_method"] == "JSD"
    assert kwargs["reference_score"] == pytest.approx(0.21)


def test_get_reference_divergence_scores(mocker, sample_analysis_config):
    """Test get_reference_divergence_scores returns one train vs. test score per repertoire."""
    mocker.patch(
        'gen_airr_bm.analysis.analyse_network.get_reference_files',
        return_value=[("trainB.tsv", "testB.tsv"), ("trainA.tsv", "testA.tsv")]
    )
    mocker.patch(
        "gen_airr_bm.analysis.analyse_network.get_connectivity_distributions_by_dataset",
        side_effect=[("datasetA", pd.Series([1, 2]), [pd.Series([2, 1])]),
                     ("datasetB", pd.Series([3, 4]), [pd.Series([4, 3])])],
    )
    mocker.patch(
        "gen_airr_bm.analysis.analyse_network.calculate_divergence_scores",
        side_effect=[[0.1], [0.2]],
    )

    reference_scores = get_reference_divergence_scores(
        analysis_config=sample_analysis_config,
        compairr_output_helper_dir="/tmp/helper",
        compairr_output_dir="/tmp/output",
    )

    # The reference files are processed in sorted order
    assert reference_scores["repertoire"].tolist() == ["datasetA", "datasetB"]
    assert reference_scores["score"].tolist() == pytest.approx([0.1, 0.2])


@pytest.mark.parametrize("gen_file, expected", [
    ("/gen/model/6271_pLN_CD8_3.tsv", 3),
    ("rep_1_0.tsv", 0),
])
def test_get_subset_index(gen_file, expected):
    assert get_subset_index(gen_file) == expected


def test_get_subset_index_without_index():
    assert np.isnan(get_subset_index("/gen/dataset1_gen1.tsv"))


@pytest.mark.parametrize("statistical_tests", [False, True])
def test_run_network_statistics(tmp_path, sample_analysis_config, statistical_tests):
    sample_analysis_config.analysis_output_dir = str(tmp_path)
    sample_analysis_config.statistical_tests = statistical_tests
    sample_analysis_config.donor_pattern = "^(dataset\\d+)$"
    sample_analysis_config.contrasts = [["model2", "model1"]]
    scores_df = make_scores_df({
        ("test", f"dataset{i}", model): [base + 0.01 * i, base + 0.02 * i]
        for i in range(1, 4) for model, base in [("model1", 0.2), ("model2", 0.3)]
    } | {("train", "dataset1", "model1"): [0.9, 0.9]})
    reference_scores = pd.DataFrame({"repertoire": ["dataset1", "dataset2", "dataset3"], "score": [0.1, 0.12, 0.11]})

    run_network_statistics(sample_analysis_config, scores_df, reference_scores)

    prefix = f"{tmp_path}/statistics/connectivity"
    vs_train_test = pd.read_csv(f"{prefix}_vs_train_test.tsv", sep="\t")
    # Only the test scores are compared, three repertoires per model
    assert vs_train_test["first"].tolist() == ["model1", "model2"]
    assert vs_train_test["n_repertoires"].tolist() == [3, 3]
    assert ("p_adjusted" in vs_train_test.columns) == statistical_tests
    contrasts = pd.read_csv(f"{prefix}_contrasts.tsv", sep="\t")
    assert contrasts[["first", "second"]].values.tolist() == [["model2", "model1"]]
    for plot in ["vs_train_test", "contrasts"]:
        assert os.path.exists(f"{prefix}_{plot}.png")


def test_run_network_statistics_with_tests_requires_one_repertoire_per_donor(tmp_path, sample_analysis_config):
    sample_analysis_config.analysis_output_dir = str(tmp_path)
    sample_analysis_config.statistical_tests = True
    sample_analysis_config.donor_pattern = "^(dataset)"
    scores_df = make_scores_df({("test", f"dataset{i}", "model1"): [0.2, 0.3] for i in range(1, 3)})
    reference_scores = pd.DataFrame({"repertoire": ["dataset1", "dataset2"], "score": [0.1, 0.1]})

    with pytest.raises(ValueError, match="one repertoire per donor"):
        run_network_statistics(sample_analysis_config, scores_df, reference_scores)


def test_end_to_end_workflow_with_mocks(mocker, sample_analysis_config):
    """Test the complete workflow with mocked external dependencies."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Update config to use temp directory
        sample_analysis_config.analysis_output_dir = temp_dir

        mock_get_files = mocker.patch('gen_airr_bm.analysis.analyse_network.get_sequence_files')
        mocker.patch('gen_airr_bm.analysis.analyse_network.deduplicate_single_dataset')
        mocker.patch('gen_airr_bm.analysis.analyse_network.run_compairr_existence')
        mock_read_csv = mocker.patch('pandas.read_csv')
        mocker.patch('gen_airr_bm.analysis.analyse_network.plot_degree_distribution_by_dataset')
        mocker.patch('gen_airr_bm.analysis.analyse_network.plot_avg_scores')
        mocker.patch('gen_airr_bm.analysis.analyse_network.calculate_jsd', return_value=[0.1])
        mocker.patch('gen_airr_bm.analysis.analyse_network.get_reference_divergence_scores',
                     return_value=pd.DataFrame({"repertoire": ["dataset1"], "score": [0.25]}))
        mocker.patch('gen_airr_bm.analysis.analyse_network.summarize_and_plot_all')
        mocker.patch('gen_airr_bm.analysis.analyse_network.run_network_statistics')
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
