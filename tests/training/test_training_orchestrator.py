import concurrent.futures
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
        immuneml_model_config="immuneml_model_config.yaml",
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


def test_save_ref_data(tmp_path):
    src = tmp_path / "in.tsv"
    src.write_text("locus\tv\nTRB\t1\n")
    model_config = make_model_config(tmp_path)
    work = tmp_path / "work"
    (work / "train_sequences").mkdir(parents=True)
    (work / "train_sequences" / "otherB_expA.tsv").write_text("locus\tv\nTRB\t2\n")

    TrainingOrchestrator.save_ref_data(
        model_config=model_config,
        output_dir=str(work),
        ref_data_full_path=str(src),
        ref_data_file_name="trainA",
        ref_name=DatasetSplit.TRAIN
    )

    assert (work / "train_sequences" / "trainA_expA.tsv").read_text() == src.read_text()
    compairr = pd.read_csv(work / "train_compairr_sequences" / "trainA_expA.tsv", sep="\t")
    assert list(compairr["sequence_id"]) == ["sequence_1"]
    # Other experiments' files in the shared folder are left alone
    assert sorted(p.name for p in (work / "train_compairr_sequences").iterdir()) == ["trainA_expA.tsv"]


def test_save_ref_data_keeps_existing_copy(tmp_path):
    src = tmp_path / "in.tsv"
    src.write_text("locus\tv\nTRB\t1\n")
    model_config = make_model_config(tmp_path)
    dst = tmp_path / "work" / "train_sequences" / "trainA_expA.tsv"
    dst.parent.mkdir(parents=True)
    dst.write_text("locus\tv\nTRB\t9\n")

    TrainingOrchestrator.save_ref_data(model_config, str(tmp_path / "work"), str(src), "trainA", DatasetSplit.TRAIN)

    assert dst.read_text() == "locus\tv\nTRB\t9\n"


def test_save_generated_sequences(tmp_path):
    base_out = tmp_path / "imm"
    model_dir = base_out / "immuneml" / "gen_model" / "generated_sequences" / "model"
    model_dir.mkdir(parents=True)
    src_gen = model_dir / "generated.tsv"
    pd.DataFrame({"locus": ["TRB"] * 7, "v": range(7)}).to_csv(src_gen, sep="\t", index=False)

    model_config = make_model_config(tmp_path)
    work = tmp_path / "work"
    split_dir = work / "generated_compairr_sequences_split" / model_config.name
    split_dir.mkdir(parents=True)
    (split_dir / "trainA_expA_5.tsv").write_text("stale\n")
    (split_dir / "trainA_expAB_0.tsv").write_text("other set\n")

    TrainingOrchestrator.save_generated_sequences(
        model_config=model_config,
        output_dir=str(work),
        immuneml_output_dir=str(base_out),
        train_data_file_name="trainA",
    )

    dst = work / "generated_sequences" / model_config.name / "trainA_expA.tsv"
    assert dst.read_text() == src_gen.read_text()
    assert (work / "generated_compairr_sequences" / model_config.name / "trainA_expA.tsv").is_file()
    assert sorted(p.name for p in split_dir.iterdir()) == ["trainA_expAB_0.tsv", "trainA_expA_0.tsv",
                                                           "trainA_expA_1.tsv"]
    first_split = pd.read_csv(split_dir / "trainA_expA_0.tsv", sep="\t")
    assert list(first_split["v"]) == [0, 1, 2]


def write_generated_set(work, model_config, file_name, n_rows, n_splits):
    gen_dir = work / "generated_sequences" / model_config.name
    gen_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{file_name}_{model_config.experiment}"
    pd.DataFrame({"junction_aa": ["CASS"] * n_rows}).to_csv(gen_dir / f"{stem}.tsv", sep="\t", index=False)
    split_dir = work / "generated_compairr_sequences_split" / model_config.name
    split_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_splits):
        (split_dir / f"{stem}_{i}.tsv").write_text("junction_aa\nCASS\n")


def test_is_training_done(tmp_path):
    model_config = make_model_config(tmp_path)
    work = tmp_path / "work"

    assert not TrainingOrchestrator.is_training_done(model_config, str(work), "trainA")
    write_generated_set(work, model_config, "trainA", n_rows=7, n_splits=1)
    assert not TrainingOrchestrator.is_training_done(model_config, str(work), "trainA")
    write_generated_set(work, model_config, "trainA", n_rows=7, n_splits=2)
    assert TrainingOrchestrator.is_training_done(model_config, str(work), "trainA")


def test_is_training_done_with_too_few_generated_sequences(tmp_path):
    model_config = make_model_config(tmp_path)
    work = tmp_path / "work"
    write_generated_set(work, model_config, "trainA", n_rows=2, n_splits=0)

    assert not TrainingOrchestrator.is_training_done(model_config, str(work), "trainA")


def test_preprocessing_is_safe_in_parallel(tmp_path):
    """Experiments finishing at the same time must not truncate each other's CompAIRR files."""
    n_experiments, n_rows = 16, 20000
    work = tmp_path / "work"
    configs = []
    for exp in range(n_experiments):
        base_out = tmp_path / f"imm_{exp}"
        model_dir = base_out / "immuneml" / "gen_model" / "generated_sequences" / "model"
        model_dir.mkdir(parents=True)
        pd.DataFrame({"junction_aa": ["CASSLGQGYEQYF"] * n_rows}).to_csv(model_dir / "gen.tsv", sep="\t",
                                                                          index=False)
        configs.append((make_model_config(tmp_path, experiment=exp, n_subset_samples=3000), base_out))

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_experiments) as executor:
        list(executor.map(lambda c: TrainingOrchestrator.save_generated_sequences(c[0], str(work), str(c[1]), "donor"),
                          configs))

    split_dir = work / "generated_compairr_sequences_split" / "modelX"
    for exp in range(n_experiments):
        assert len(TrainingOrchestrator.get_split_files(split_dir, f"donor_{exp}")) == n_rows // 3000


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


def test_run_training_skips_finished_files(tmp_path, mocker):
    odir = tmp_path / "proj"
    train_dir = odir / "train_in"
    train_dir.mkdir(parents=True)

    for n in ["A.tsv", "B.tsv", "C.tsv"]:
        (train_dir / n).write_text("locus\tv\nTRB\t1\n")

    model_config = make_model_config(tmp_path, output_dir=str(odir))
    work_dir = tmp_path / "work"
    write_generated_set(work_dir, model_config, "A", n_rows=7, n_splits=2)
    write_generated_set(work_dir, model_config, "B", n_rows=7, n_splits=1)

    mocker.patch.object(TrainingOrchestrator, "get_default_locus_name", return_value="TRB")
    mocker.patch.object(TrainingOrchestrator, "save_ref_data")
    m_split = mocker.patch.object(TrainingOrchestrator, "split_generated_sequences")
    m_run_single = mocker.patch.object(TrainingOrchestrator, "run_single_training")
    m_save_gen = mocker.patch.object(TrainingOrchestrator, "save_generated_sequences")

    TrainingOrchestrator.run_training(model_config=model_config, output_dir=str(work_dir))

    # A is complete, B only needs re-splitting, C still has to be trained
    m_split.assert_called_once_with(model_config, str(work_dir), "B")
    assert m_run_single.call_count == 1
    assert m_run_single.call_args.args[1].endswith("C.tsv")
    assert m_save_gen.call_count == 1
