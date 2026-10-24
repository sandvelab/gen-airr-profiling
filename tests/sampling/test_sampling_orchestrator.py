import os
import zipfile
from pathlib import Path

import pytest
import yaml

from gen_airr_bm.core.sampling_config import SamplingConfig
from gen_airr_bm.sampling.sampling_orchestrator import SamplingOrchestrator


def make_sampling_config(tmp_path, **overrides):
    root_dir = tmp_path / "root_out"
    root_dir.mkdir()
    base = dict(
        model_name="modelA",
        experiment_name="exp_3",
        immuneml_config=str(tmp_path / "base.yaml"),
        n_samples=50,
        root_output_dir=str(root_dir),
    )
    base.update(overrides)
    return SamplingConfig(**base)


def test_find_paths(tmp_path):
    config = make_sampling_config(tmp_path)
    model_dir = Path(config.root_output_dir) / config.experiment_name / config.model_name
    dataset_dir = model_dir / "dataset_one" / "immuneml" / "gen_model" / "trained_model_model"
    dataset_dir.mkdir(parents=True)
    zip_path = dataset_dir / "trained.zip"
    zip_path.write_text("zip")

    found_path, dataset_name = SamplingOrchestrator.find_paths(config)

    assert found_path == zip_path
    assert dataset_name == "dataset_one"


def test_prepare_immuneml_sampling_config(tmp_path):
    base_cfg = tmp_path / "base.yaml"
    yaml.safe_dump(
        {
            "instructions": {
                "gen_model": {
                    "ml_config_path": "old.zip",
                    "gen_examples_count": 5,
                }
            }
        },
        base_cfg.open("w", encoding="utf-8"),
        sort_keys=False,
    )
    config = make_sampling_config(tmp_path, immuneml_config=str(base_cfg))
    model_zip = Path("/path/to/model.zip")

    cfg_path = SamplingOrchestrator.prepare_immuneml_sampling_config(
        sampling_config=config,
        dataset_name="datasetA",
        model_zip_path=model_zip,
    )

    assert cfg_path.name == "exp_3_modelA_datasetA.yaml"
    saved = yaml.safe_load(cfg_path.read_text())
    assert saved["instructions"]["gen_model"]["ml_config_path"] == str(model_zip)
    assert saved["instructions"]["gen_model"]["gen_examples_count"] == config.n_samples


def test_run_immuneml_sampling(tmp_path, mocker):
    config = make_sampling_config(tmp_path)
    mock_run = mocker.patch(
        "gen_airr_bm.sampling.sampling_orchestrator.run_immuneml_command"
    )
    cfg_path = tmp_path / "config.yaml"

    out_dir = SamplingOrchestrator.run_immuneml_sampling(config, "datasetB", cfg_path)

    expected = Path(config.root_output_dir) / "sampling" / config.experiment_name / config.model_name / "datasetB"
    assert Path(out_dir) == expected
    mock_run.assert_called_once_with(cfg_path, str(expected))


def test_copy_generated_sequences(tmp_path, mocker):
    config = make_sampling_config(tmp_path, experiment_name="exp_12")
    results_dir = tmp_path / "results"
    gen_dir = results_dir / "gen_model" / "generated_sequences"
    gen_dir.mkdir(parents=True)
    seq_file = gen_dir / "seqs.tsv"
    seq_file.write_text("locus\tcdr3\n")

    mock_cp = mocker.patch.object(os, "system", return_value=0)

    SamplingOrchestrator.copy_generated_sequences(config, str(results_dir), "datasetC")

    dest_dir = Path(config.root_output_dir) / "resampled_sequences_raw" / config.model_name
    out_file = dest_dir / "datasetC_12.tsv"
    mock_cp.assert_called_once_with(f"cp {seq_file} {out_file}")


def test_run_sampling(mocker):
    config = SamplingConfig(
        model_name="modelB",
        experiment_name="exp_7",
        immuneml_config="config.yaml",
        n_samples=100,
        root_output_dir="/tmp/root",
    )
    m_find = mocker.patch.object(
        SamplingOrchestrator,
        "find_paths",
        return_value=(Path("model.zip"), "datasetZ"),
    )
    m_prepare = mocker.patch.object(
        SamplingOrchestrator,
        "prepare_immuneml_sampling_config",
        return_value=Path("out.yaml"),
    )
    m_run = mocker.patch.object(
        SamplingOrchestrator,
        "run_immuneml_sampling",
        return_value="/tmp/res",
    )
    m_copy = mocker.patch.object(SamplingOrchestrator, "copy_generated_sequences")
    m_copy_p_gens = mocker.patch.object(SamplingOrchestrator, "copy_p_gens")
    m_set_device = mocker.patch.object(SamplingOrchestrator, "set_model_device")

    SamplingOrchestrator.run_sampling(config)

    m_find.assert_called_once_with(config)
    m_set_device.assert_not_called()
    m_prepare.assert_called_once_with(config, "datasetZ", Path("model.zip"))
    m_run.assert_called_once_with(config, "datasetZ", Path("out.yaml"))
    m_copy.assert_called_once_with(config, "/tmp/res", "datasetZ")
    m_copy_p_gens.assert_not_called()


def write_ref_sequences(config, dataset_name):
    experiment_n = config.experiment_name.split("_")[-1]
    for split in ["train", "test"]:
        ref_dir = Path(config.root_output_dir) / f"{split}_sequences"
        ref_dir.mkdir(parents=True)
        (ref_dir / f"{dataset_name}_{experiment_n}.tsv").write_text("junction_aa\nCASSF\n")


def test_prepare_immuneml_sampling_config_with_p_gens(tmp_path):
    base_cfg = tmp_path / "base.yaml"
    yaml.safe_dump(
        {
            "definitions": [],
            "instructions": {
                "gen_model": {
                    "type": "ApplyGenModel",
                    "ml_config_path": "old.zip",
                    "gen_examples_count": 0,
                    "reports": [],
                }
            }
        },
        base_cfg.open("w", encoding="utf-8"),
        sort_keys=False,
    )
    config = make_sampling_config(tmp_path, immuneml_config=str(base_cfg), compute_p_gens=True)
    write_ref_sequences(config, "datasetA")

    cfg_path = SamplingOrchestrator.prepare_immuneml_sampling_config(config, "datasetA", Path("/path/to/model.zip"))

    saved = yaml.safe_load(cfg_path.read_text())
    assert saved["instructions"]["gen_model"]["p_gen_datasets"] == ["train", "test"]
    for split in ["train", "test"]:
        dataset = saved["definitions"]["datasets"][split]
        assert dataset["format"] == "AIRR"
        assert dataset["params"]["path"] == str(Path(config.root_output_dir) / f"{split}_sequences" / "datasetA_3.tsv")
        assert dataset["params"]["is_repertoire"] is False
        assert dataset["params"]["separator"] == "\t"


def test_prepare_immuneml_sampling_config_without_p_gens_adds_no_datasets(tmp_path):
    base_cfg = tmp_path / "base.yaml"
    yaml.safe_dump({"definitions": [], "instructions": {"gen_model": {"ml_config_path": "old.zip"}}},
                   base_cfg.open("w", encoding="utf-8"))
    config = make_sampling_config(tmp_path, immuneml_config=str(base_cfg))

    cfg_path = SamplingOrchestrator.prepare_immuneml_sampling_config(config, "datasetA", Path("model.zip"))

    saved = yaml.safe_load(cfg_path.read_text())
    assert saved["definitions"] == []
    assert "p_gen_datasets" not in saved["instructions"]["gen_model"]


def test_add_p_gen_datasets_missing_ref_sequences(tmp_path):
    config = make_sampling_config(tmp_path, compute_p_gens=True)
    data = {"definitions": [], "instructions": {"gen_model": {}}}

    with pytest.raises(FileNotFoundError):
        SamplingOrchestrator.add_p_gen_datasets(data, config, "datasetA")


def test_copy_p_gens(tmp_path):
    config = make_sampling_config(tmp_path, experiment_name="exp_12", compute_p_gens=True)
    results_dir = tmp_path / "results"
    p_gens_dir = results_dir / "gen_model" / "datasets_with_p_gens"
    p_gens_dir.mkdir(parents=True)
    for split in ["train", "test"]:
        (p_gens_dir / f"{split}_p_gens.tsv").write_text(
            f"sequence_id\tjunction_aa\tv_call\tj_call\tlocus\tp_gen_model\n\t{split}\tTRBV1\tTRBJ1\tTRB\t0.1\n")

    SamplingOrchestrator.copy_p_gens(config, str(results_dir), "datasetC")

    for split in ["train", "test"]:
        out_file = Path(config.root_output_dir) / f"{split}_p_gens" / config.model_name / "datasetC_12.tsv"
        assert out_file.read_text() == f"junction_aa\tv_call\tj_call\tp_gen_model\n{split}\tTRBV1\tTRBJ1\t0.1\n"


def test_run_sampling_with_p_gens(mocker):
    config = SamplingConfig(model_name="modelB", experiment_name="exp_7", immuneml_config="config.yaml",
                            n_samples=100, root_output_dir="/tmp/root", compute_p_gens=True)
    mocker.patch.object(SamplingOrchestrator, "find_paths", return_value=(Path("model.zip"), "datasetZ"))
    mocker.patch.object(SamplingOrchestrator, "prepare_immuneml_sampling_config", return_value=Path("out.yaml"))
    mocker.patch.object(SamplingOrchestrator, "run_immuneml_sampling", return_value="/tmp/res")
    mocker.patch.object(SamplingOrchestrator, "copy_generated_sequences")
    m_copy_p_gens = mocker.patch.object(SamplingOrchestrator, "copy_p_gens")

    SamplingOrchestrator.run_sampling(config)

    m_copy_p_gens.assert_called_once_with(config, "/tmp/res", "datasetZ")


def test_run_p_gens_only_keeps_sampled_sequences(mocker):
    config = SamplingConfig(model_name="modelB", experiment_name="exp_7", immuneml_config="config.yaml",
                            n_samples=0, root_output_dir="/tmp/root", compute_p_gens=True)
    mocker.patch.object(SamplingOrchestrator, "find_paths", return_value=(Path("model.zip"), "datasetZ"))
    mocker.patch.object(SamplingOrchestrator, "prepare_immuneml_sampling_config", return_value=Path("out.yaml"))
    mocker.patch.object(SamplingOrchestrator, "run_immuneml_sampling", return_value="/tmp/res")
    m_copy = mocker.patch.object(SamplingOrchestrator, "copy_generated_sequences")
    m_copy_p_gens = mocker.patch.object(SamplingOrchestrator, "copy_p_gens")

    SamplingOrchestrator.run_sampling(config)

    m_copy.assert_not_called()
    m_copy_p_gens.assert_called_once_with(config, "/tmp/res", "datasetZ")


def test_p_gens_only_writes_to_separate_dirs(tmp_path, mocker):
    base_cfg = tmp_path / "base.yaml"
    yaml.safe_dump({"definitions": [], "instructions": {"gen_model": {"ml_config_path": "old.zip"}}},
                   base_cfg.open("w", encoding="utf-8"))
    config = make_sampling_config(tmp_path, immuneml_config=str(base_cfg), n_samples=0, compute_p_gens=True)
    write_ref_sequences(config, "datasetA")
    mock_run = mocker.patch("gen_airr_bm.sampling.sampling_orchestrator.run_immuneml_command")

    cfg_path = SamplingOrchestrator.prepare_immuneml_sampling_config(config, "datasetA", Path("model.zip"))
    out_dir = SamplingOrchestrator.run_immuneml_sampling(config, "datasetA", cfg_path)

    root_dir = Path(config.root_output_dir)
    assert cfg_path.parent == root_dir / "p_gen_runs_configs"
    assert yaml.safe_load(cfg_path.read_text())["instructions"]["gen_model"]["gen_examples_count"] == 0
    assert Path(out_dir) == root_dir / "p_gen_runs" / config.experiment_name / config.model_name / "datasetA"
    mock_run.assert_called_once_with(cfg_path, out_dir)
    assert not (root_dir / "sampling").exists() and not (root_dir / "sampling_configs").exists()


def write_model_zip(zip_path, model_overview):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        zip_file.writestr("model_overview.yaml", yaml.safe_dump(model_overview, sort_keys=False))
        zip_file.writestr("state_dict.pt", b"weights")


def read_model_overview(zip_path):
    with zipfile.ZipFile(zip_path) as zip_file:
        return yaml.safe_load(zip_file.read("model_overview.yaml"))


def test_set_model_device_keeps_original_zip(tmp_path):
    zip_path = tmp_path / "trained_model_model" / "trained_model_model.zip"
    write_model_zip(zip_path, {"type": "SakaLSTM", "device": "cpu", "hidden_size": 64})

    SamplingOrchestrator.set_model_device(zip_path, "cuda")

    assert read_model_overview(zip_path) == {"type": "SakaLSTM", "device": "cuda", "hidden_size": 64}
    with zipfile.ZipFile(zip_path) as zip_file:
        assert zip_file.read("state_dict.pt") == b"weights"
    backup_path = zip_path.with_name("trained_model_model.zip.orig")
    assert read_model_overview(backup_path)["device"] == "cpu"
    assert sorted(path.name for path in zip_path.parent.glob("*.zip")) == ["trained_model_model.zip"]


def test_set_model_device_twice_keeps_first_backup(tmp_path):
    zip_path = tmp_path / "model.zip"
    write_model_zip(zip_path, {"type": "SakaLSTM", "device": "cpu"})

    SamplingOrchestrator.set_model_device(zip_path, "cuda")
    SamplingOrchestrator.set_model_device(zip_path, "cuda:1")

    assert read_model_overview(zip_path)["device"] == "cuda:1"
    assert read_model_overview(tmp_path / "model.zip.orig")["device"] == "cpu"


def test_set_model_device_same_device_does_not_change_zip(tmp_path):
    zip_path = tmp_path / "model.zip"
    write_model_zip(zip_path, {"type": "SakaLSTM", "device": "cuda"})

    SamplingOrchestrator.set_model_device(zip_path, "cuda")

    assert not (tmp_path / "model.zip.orig").exists()


def test_set_model_device_without_device_parameter(tmp_path):
    zip_path = tmp_path / "model.zip"
    write_model_zip(zip_path, {"type": "PWM", "locus": "beta"})

    SamplingOrchestrator.set_model_device(zip_path, "cuda")

    assert read_model_overview(zip_path) == {"type": "PWM", "locus": "beta"}
    assert not (tmp_path / "model.zip.orig").exists()


def test_run_sampling_sets_device(mocker):
    config = SamplingConfig(model_name="modelB", experiment_name="exp_7", immuneml_config="config.yaml",
                            n_samples=100, root_output_dir="/tmp/root", device="cuda")
    mocker.patch.object(SamplingOrchestrator, "find_paths", return_value=(Path("model.zip"), "datasetZ"))
    m_set_device = mocker.patch.object(SamplingOrchestrator, "set_model_device")
    m_prepare = mocker.patch.object(SamplingOrchestrator, "prepare_immuneml_sampling_config",
                                    return_value=Path("out.yaml"))
    mocker.patch.object(SamplingOrchestrator, "run_immuneml_sampling", return_value="/tmp/res")
    mocker.patch.object(SamplingOrchestrator, "copy_generated_sequences")

    SamplingOrchestrator.run_sampling(config)

    m_set_device.assert_called_once_with(Path("model.zip"), "cuda")
    m_prepare.assert_called_once_with(config, "datasetZ", Path("model.zip"))


def write_outputs(paths):
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("junction_aa\nCASSF\n")


def test_get_output_paths(tmp_path):
    root_dir = Path(tmp_path)
    config = SamplingConfig(model_name="modelA", experiment_name="exp_3", immuneml_config="config.yaml",
                            n_samples=50, root_output_dir=str(root_dir), compute_p_gens=True)

    paths = SamplingOrchestrator.get_output_paths(config, "datasetA")

    assert paths == [root_dir / "resampled_sequences_raw" / "modelA" / "datasetA_3.tsv",
                     root_dir / "train_p_gens" / "modelA" / "datasetA_3.tsv",
                     root_dir / "test_p_gens" / "modelA" / "datasetA_3.tsv"]


def test_get_output_paths_p_gens_only_and_no_p_gens(tmp_path):
    p_gens_only = SamplingConfig(model_name="modelA", experiment_name="exp_3", immuneml_config="config.yaml",
                                 n_samples=0, root_output_dir=str(tmp_path), compute_p_gens=True)
    no_p_gens = SamplingConfig(model_name="modelA", experiment_name="exp_3", immuneml_config="config.yaml",
                               n_samples=50, root_output_dir=str(tmp_path))

    assert [path.parent.parent.name for path in SamplingOrchestrator.get_output_paths(p_gens_only, "d")] == \
           ["train_p_gens", "test_p_gens"]
    assert [path.parent.parent.name for path in SamplingOrchestrator.get_output_paths(no_p_gens, "d")] == \
           ["resampled_sequences_raw"]


def test_is_sampling_done_needs_all_outputs(tmp_path):
    config = make_sampling_config(tmp_path, compute_p_gens=True)
    sequences_path, train_p_gens_path, test_p_gens_path = SamplingOrchestrator.get_output_paths(config, "datasetA")

    write_outputs([sequences_path, train_p_gens_path])
    assert not SamplingOrchestrator.is_sampling_done(config, "datasetA")

    write_outputs([test_p_gens_path])
    assert SamplingOrchestrator.is_sampling_done(config, "datasetA")


def test_run_sampling_skips_finished_run(tmp_path, mocker):
    config = make_sampling_config(tmp_path, device="cuda")
    write_outputs(SamplingOrchestrator.get_output_paths(config, "datasetZ"))
    mocker.patch.object(SamplingOrchestrator, "find_paths", return_value=(Path("model.zip"), "datasetZ"))
    m_set_device = mocker.patch.object(SamplingOrchestrator, "set_model_device")
    m_run = mocker.patch.object(SamplingOrchestrator, "run_immuneml_sampling")

    SamplingOrchestrator.run_sampling(config)

    m_set_device.assert_not_called()
    m_run.assert_not_called()


def test_run_immuneml_sampling_removes_unfinished_results(tmp_path, mocker):
    config = make_sampling_config(tmp_path)
    mock_run = mocker.patch("gen_airr_bm.sampling.sampling_orchestrator.run_immuneml_command")
    results_dir = Path(config.root_output_dir) / "sampling" / config.experiment_name / config.model_name / "datasetB"
    (results_dir / "gen_model").mkdir(parents=True)
    (results_dir / "log.txt").write_text("unfinished")

    out_dir = SamplingOrchestrator.run_immuneml_sampling(config, "datasetB", tmp_path / "config.yaml")

    assert Path(out_dir) == results_dir
    assert results_dir.is_dir() and list(results_dir.iterdir()) == []
    mock_run.assert_called_once_with(tmp_path / "config.yaml", str(results_dir))
