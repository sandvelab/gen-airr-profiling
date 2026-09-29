import subprocess

import pytest

from gen_airr_bm.training.immuneml_runner import run_immuneml_command


def test_run_immuneml_command_raises_on_failure(mocker):
    process = mocker.Mock(pid=1, wait=mocker.Mock(return_value=1))
    mocker.patch("gen_airr_bm.training.immuneml_runner.subprocess.Popen", return_value=process)

    with pytest.raises(subprocess.CalledProcessError):
        run_immuneml_command("config.yaml", "out")


def test_run_immuneml_command_succeeds(mocker):
    process = mocker.Mock(pid=1, wait=mocker.Mock(return_value=0))
    mocker.patch("gen_airr_bm.training.immuneml_runner.subprocess.Popen", return_value=process)

    run_immuneml_command("config.yaml", "out")
