import os
from types import SimpleNamespace

import pandas as pd
import pytest

from gen_airr_bm.constants.dataset_split import DatasetSplit
from gen_airr_bm.training.training_orchestrator import TrainingOrchestrator


def make_model_config(tmp_path, **overrides):
    defaults = dict(
        experiment="expA",
        output_dir=str(tmp_path),
        test_dir="test_in",
        name="modelX",
        n_subset_samples=3,
        config="immuneml_model_config.yaml",
        train_dir="train_in",
        locus=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_run_single_training(tmp_path, mocker):
    mock_write = mocker.patch(
        "gen_airr_bm.training.training_orchestrator.write_immuneml_config"
    )
    mock_run = mocker.patch(
        "gen_airr_bm.training.training_orchestrator.run_immuneml_command"
    )

    TrainingOrchestrator.run_single_training(
        immuneml_config_path="cfg.yaml",
        train_data_path="train.tsv",
        immuneml_output_dir=str(tmp_path / "out"),
        locus="TRB",
    )

    out_cfg = tmp_path / "out" / "immuneml_config.yaml"
    out_dir = tmp_path / "out" / "immuneml"
    mock_write.assert_called_once_with("cfg.yaml", "train.tsv", out_cfg, "TRB")
    mock_run.assert_called_once_with(out_cfg, out_dir)


def test_get_default_locus_name(tmp_path):
    p = tmp_path / "train.tsv"
    pd.DataFrame({"locus": ["TRB"] * 3}).to_csv(p, sep="\t", index=False)
    assert TrainingOrchestrator.get_default_locus_name(str(p)) == "TRB"

    pd.DataFrame({"locus": ["TRB", "TRA"]}).to_csv(p, sep="\t", index=False)
    with pytest.raises(ValueError):
        TrainingOrchestrator.get_default_locus_name(str(p))


def test_divide_generated_sequences(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    base = "gen"
    pd.DataFrame({"locus": ["TRB"] * 10, "v": range(10)}).to_csv(
        src_dir / f"{base}.tsv", sep="\t", index=False
    )

    out_dir = tmp_path / "out"
    TrainingOrchestrator.divide_generated_sequences(
        generated_sequences_dir=str(src_dir),
        generated_sequences_filename=base,
        divided_sequences_output_dir=str(out_dir),
        n_samples_per_subset=3,
    )

    assert sorted(p.name for p in out_dir.iterdir()) == ["gen_0.tsv", "gen_1.tsv", "gen_2.tsv"]


def test_save_ref_data(tmp_path, mocker):
    src = tmp_path / "in.tsv"
    src.write_text("locus\tv\nTRB\t1\n")
    model_config = make_model_config(tmp_path)

    mock_system = mocker.patch.object(os, "system", return_value=0)
    mock_prep = mocker.patch(
        "gen_airr_bm.training.training_orchestrator.preprocess_files_for_compairr"
    )

    TrainingOrchestrator.save_ref_data(
        model_config=model_config,
        output_dir=str(tmp_path / "work"),
        ref_data_full_path=str(src),
        ref_data_file_name="trainA",
        ref_name=DatasetSplit.TRAIN
    )

    dst = tmp_path / "work" / "train_sequences" / "trainA_expA.tsv"
    mock_system.assert_called_once_with(f"cp -n {src} {dst}")
    mock_prep.assert_called_once_with(
        str(tmp_path / "work" / "train_sequences"),
        str(tmp_path / "work" / "train_compairr_sequences"),
    )


def test_save_generated_sequences(tmp_path, mocker):
    base_out = tmp_path / "imm"
    model_dir = base_out / "immuneml" / "gen_model" / "generated_sequences" / "model"
    model_dir.mkdir(parents=True)
    src_gen = model_dir / "generated.tsv"
    src_gen.write_text("locus\tv\nTRB\t1\nTRB\t2\n")

    model_config = make_model_config(tmp_path)

    mock_system = mocker.patch.object(os, "system", return_value=0)
    mock_prep = mocker.patch(
        "gen_airr_bm.training.training_orchestrator.preprocess_files_for_compairr"
    )
    mock_divide = mocker.patch.object(
        TrainingOrchestrator, "divide_generated_sequences"
    )

    TrainingOrchestrator.save_generated_sequences(
        model_config=model_config,
        output_dir=str(tmp_path / "work"),
        immuneml_output_dir=str(base_out),
        train_data_file_name="trainA",
    )

    dst = tmp_path / "work" / "generated_sequences" / model_config.name / "trainA_expA.tsv"
    mock_system.assert_called_once_with(f"cp -n {src_gen} {dst}")
    mock_prep.assert_called_once_with(
        str(tmp_path / "work" / "generated_sequences" / model_config.name),
        str(tmp_path / "work" / "generated_compairr_sequences" / model_config.name),
    )
    mock_divide.assert_called_once_with(
        str(tmp_path / "work" / "generated_compairr_sequences" / model_config.name),
        "trainA_expA",
        str(tmp_path / "work" / "generated_compairr_sequences_split" / model_config.name),
        3,
    )


def test_run_training(tmp_path, mocker):
    odir = tmp_path / "proj"
    train_dir = odir / "train_in"
    train_dir.mkdir(parents=True)

    for n in ["A.tsv", "B.tsv"]:
        (train_dir / n).write_text("locus\tv\nTRB\t1\n")

    model_config = make_model_config(tmp_path, output_dir=str(odir))

    m_get_locus = mocker.patch.object(
        TrainingOrchestrator, "get_default_locus_name", return_value="TRB"
    )
    m_save_ref = mocker.patch.object(TrainingOrchestrator, "save_ref_data")
    m_run_single = mocker.patch.object(TrainingOrchestrator, "run_single_training")
    m_save_gen = mocker.patch_object = mocker.patch.object(
        TrainingOrchestrator, "save_generated_sequences"
    )

    TrainingOrchestrator.run_training(
        model_config=model_config, output_dir=str(tmp_path / "work")
    )

    assert m_get_locus.call_count == 2
    assert m_save_ref.call_count == 4
    assert m_run_single.call_count == 2
    assert m_save_gen.call_count == 2
