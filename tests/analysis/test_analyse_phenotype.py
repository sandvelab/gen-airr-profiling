import os
import pandas as pd
import pytest

from gen_airr_bm.analysis.analyse_phenotype import calculate_similarities_matrix
from gen_airr_bm.core.analysis_config import AnalysisConfig


@pytest.fixture
def sample_analysis_config():
    """Create a sample AnalysisConfig for testing."""
    return AnalysisConfig(
        analysis="phenotype",
        model_names=["model1", "model2"],
        analysis_output_dir="/tmp/test_output",
        root_output_dir="/tmp/test_root",
        default_model_name="humanTRB",
        reference_data=["train", "test"],
        n_subsets=5,
        subfolder_name="analysis_subfolder",
        receptor_type="TCR",
        allowed_mismatches=0,
        indels=False,
    )


def _make_overlap_df(n_rows):
    # Create a dataframe with n_rows where both dataset_1 and dataset_2 are non-zero
    return pd.DataFrame({"dataset_1": [1] * n_rows, "dataset_2": [1] * n_rows})


def _make_union_df(n_rows):
    # Create a dataframe with n_rows to emulate the unique sequences file
    return pd.DataFrame({"junction_aa": [f"seq{i}" for i in range(n_rows)]})


def test_calculate_similarities_matrix(mocker, tmp_path, sample_analysis_config):
    # Prepare fake directory and files
    sequences_dir = "/fake/generated_compairr_sequences/modelX"
    output_dir = str(tmp_path / "analysis_out")
    model_name = "modelX"

    # Three datasets
    fake_files = ["a.tsv", "b.tsv", "c.tsv"]
    mocker.patch("os.listdir", return_value=fake_files)

    # We'll mock dedupe and compairr runner to be no-ops
    mocker.patch("gen_airr_bm.analysis.analyse_phenotype.deduplicate_and_merge_two_datasets")
    mocker.patch("gen_airr_bm.analysis.analyse_phenotype.run_compairr_existence")

    # Define expected overlap counts and unions per pair
    # Key uses file_name pattern used in code: "{dataset1}_{dataset2}"
    overlap_counts = {
        "a_b": 2,
        "a_c": 1,
        "b_c": 3,
    }
    union_counts = {
        "a_b": 4,
        "a_c": 2,
        "b_c": 6,
    }

    # pandas.read_csv is used for two different purposes: reading the compairr overlap file
    # and reading the unique sequences file. We'll inspect the path passed to decide which
    # dataframe to return.
    def fake_read_csv(path, sep=None, usecols=None):
        path = str(path)
        if "compairr_output" in path and path.endswith("_overlap.tsv"):
            # extract file_name between compairr_output/ and _overlap.tsv
            base = os.path.basename(path)
            file_name = base.replace("_overlap.tsv", "")
            # produce dataframe with overlap_counts rows
            n = overlap_counts.get(file_name, 0)
            return _make_overlap_df(n)
        elif path.endswith("_unique.tsv"):
            base = os.path.basename(path)
            file_name = base.replace("_unique.tsv", "")
            n = union_counts.get(file_name, 0)
            return _make_union_df(n)
        else:
            # Fallback, return empty
            return pd.DataFrame()

    mocker.patch("pandas.read_csv", side_effect=fake_read_csv)

    # Make sure os.path.exists behaves normally for directories, but for helper files return False
    real_exists = os.path.exists

    def fake_exists(p):
        # If path points to a helper file (endswith _unique.tsv or _concat.tsv), pretend it doesn't exist
        s = str(p)
        if s.endswith("_unique.tsv") or s.endswith("_concat.tsv"):
            return False
        return real_exists(p)

    mocker.patch("os.path.exists", side_effect=fake_exists)

    matrix, dataset_names = calculate_similarities_matrix(sample_analysis_config, sequences_dir)

    # dataset_names should be the files without suffix
    assert dataset_names == ["a", "b", "c"]

    # Convert to numeric and verify diagonal is 1.0
    n = len(dataset_names)
    for i in range(n):
        assert matrix[i][i] == 1.0

    # Check pairwise values match overlap/union expectations and are symmetric
    def expected_jacc(pair_key):
        return overlap_counts[pair_key] / union_counts[pair_key]

    # mapping indices
    idx = {name: i for i, name in enumerate(dataset_names)}

    assert pytest.approx(matrix[idx["a"]][idx["b"]], rel=1e-9) == expected_jacc("a_b")
    assert pytest.approx(matrix[idx["b"]][idx["a"]], rel=1e-9) == expected_jacc("a_b")

    assert pytest.approx(matrix[idx["a"]][idx["c"]], rel=1e-9) == expected_jacc("a_c")
    assert pytest.approx(matrix[idx["c"]][idx["a"]], rel=1e-9) == expected_jacc("a_c")

    assert pytest.approx(matrix[idx["b"]][idx["c"]], rel=1e-9) == expected_jacc("b_c")
    assert pytest.approx(matrix[idx["c"]][idx["b"]], rel=1e-9) == expected_jacc("b_c")


def test_calculate_similarities_matrix_skips_existing_helpers(mocker, tmp_path, sample_analysis_config):
    # Test that if helper files exist, deduplication is skipped
    sequences_dir = "/fake/generated_compairr_sequences/modelY"
    output_dir = str(tmp_path / "analysis_out")
    model_name = "modelY"

    fake_files = ["x.tsv", "y.tsv"]
    mocker.patch("os.listdir", return_value=fake_files)

    dedupe_mock = mocker.patch("gen_airr_bm.analysis.analyse_phenotype.deduplicate_and_merge_two_datasets")
    compairr_mock = mocker.patch("gen_airr_bm.analysis.analyse_phenotype.run_compairr_existence")

    # Simulate that helper files already exist for x_y
    def fake_exists(p):
        s = str(p)
        if s.endswith("x_y_unique.tsv") or s.endswith("x_y_concat.tsv"):
            return True
        return False

    mocker.patch("os.path.exists", side_effect=fake_exists)

    # For the overlap and unique reads return simple dataframes
    mocker.patch("pandas.read_csv", side_effect=lambda p, sep=None, usecols=None: _make_overlap_df(1) if "_overlap.tsv" in str(p) else _make_union_df(1))

    matrix, dataset_names = calculate_similarities_matrix(sample_analysis_config, sequences_dir)

    # deduplicate should not be called because helper files exist
    dedupe_mock.assert_not_called()
    # compairr should still be called because the code always runs run_compairr_existence after checking helpers
    assert compairr_mock.call_count == 1

    # matrix should be 2x2 with diagonals 1.0 and off-diagonals 1/1 == 1.0
    assert matrix[0][0] == 1.0
    assert matrix[1][1] == 1.0
    assert matrix[0][1] == 1.0
    assert matrix[1][0] == 1.0

