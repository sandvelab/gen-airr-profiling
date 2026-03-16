import os
from pathlib import Path

import yaml

from gen_airr_bm.core.SamplingConfig import SamplingConfig
from gen_airr_bm.sampling.sampling_orchestrator import SamplingOrchestrator


def make_sampling_config(tmp_path, **overrides):
    root_dir = tmp_path / "root_out"
    root_dir.mkdir()
    base = dict(
        model_name="modelA",
        experiment_name="exp_3",
        immuneml_config=str(tmp_path / "base.yaml"),
        train_dir="train",
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
        train_dir="train",
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

    SamplingOrchestrator.run_sampling(config)

    m_find.assert_called_once_with(config)
    m_prepare.assert_called_once_with(config, "datasetZ", Path("model.zip"))
    m_run.assert_called_once_with(config, "datasetZ", Path("out.yaml"))
    m_copy.assert_called_once_with(config, "/tmp/res", "datasetZ")
