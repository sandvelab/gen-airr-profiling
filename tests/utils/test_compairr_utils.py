import pytest
from gen_airr_bm.utils.compairr_utils import run_compairr_existence


def _exists_side_effect_factory(binary_exists: bool, overlap_exists: bool):
    """Factory to create a side_effect for os.path.exists that distinguishes between
    the binary path and the expected overlap output path."""
    def _side_effect(path: str) -> bool:
        if path == "compairr-1.13.0-linux-x86_64":
            return binary_exists
        if path.endswith("_overlap.tsv"):
            return overlap_exists
        # Default to False for all other paths
        return False
    return _side_effect


@pytest.mark.parametrize(
    "binary_exists,allowed_mismatches,indels,expected_suffix",
    [
        (False, 0, False, ""),                 # default: no -d, no --indels
        (False, 1, False, " -d 1"),            # d=1, no indels
        (False, 1, True, " -d 1 --indels"),    # d=1, with indels
        (True,  2, True, " -d 2"),             # d=2, --indels should NOT be added
    ],
)
def test_run_compairr_existence(mocker, binary_exists, allowed_mismatches, indels, expected_suffix):
    compairr_output_dir = "/tmp/compairr/out"
    search_for_file = "/data/unique.tsv"
    search_in_file = "/data/concat.tsv"
    file_identifier = "dataset_model"

    # Patch makedirs and run_command
    mock_makedirs = mocker.patch("gen_airr_bm.utils.compairr_utils.os.makedirs")
    mock_run_command = mocker.patch("gen_airr_bm.utils.compairr_utils.run_command")

    # Ensure that output file does NOT exist to trigger execution
    mocker.patch(
        "gen_airr_bm.utils.compairr_utils.os.path.exists",
        side_effect=_exists_side_effect_factory(binary_exists=binary_exists, overlap_exists=False)
    )

    run_compairr_existence(
        compairr_output_dir=compairr_output_dir,
        search_for_file=search_for_file,
        search_in_file=search_in_file,
        file_identifier=file_identifier,
        allowed_mismatches=allowed_mismatches,
        indels=indels,
    )

    # Directory ensured
    mock_makedirs.assert_called_once_with(compairr_output_dir, exist_ok=True)

    # Validate constructed command
    compairr_call = "./compairr-1.13.0-linux-x86_64" if binary_exists else "compairr"
    expected_command = (
        f"{compairr_call} -x {search_for_file} {search_in_file}"
        f" -f -t 8 -u -g -o {compairr_output_dir}/{file_identifier}_overlap.tsv "
        f"--log {compairr_output_dir}/{file_identifier}_log.txt"
        f"{expected_suffix}"
    )

    mock_run_command.assert_called_once()
    called_command = mock_run_command.call_args.args[0]
    assert called_command == expected_command


def test_run_compairr_existence_skips_if_output_exists(mocker):
    compairr_output_dir = "/tmp/compairr/out"
    search_for_file = "/data/unique.tsv"
    search_in_file = "/data/concat.tsv"
    file_identifier = "dataset_model"

    # Patch makedirs and run_command
    mock_makedirs = mocker.patch("gen_airr_bm.utils.compairr_utils.os.makedirs")
    mock_run_command = mocker.patch("gen_airr_bm.utils.compairr_utils.run_command")

    mock_exists = mocker.patch("gen_airr_bm.utils.compairr_utils.os.path.exists",
                               side_effect=_exists_side_effect_factory(binary_exists=True, overlap_exists=True))

    run_compairr_existence(compairr_output_dir=compairr_output_dir,
                           search_for_file=search_for_file,
                           search_in_file=search_in_file,
                           file_identifier=file_identifier,)

    # Directory ensured
    mock_makedirs.assert_called_once_with(compairr_output_dir, exist_ok=True)

    # It should check binary and overlap file existence
    assert mock_exists.call_count >= 1

    # Since overlap exists, we must not call run_command
    mock_run_command.assert_not_called()
