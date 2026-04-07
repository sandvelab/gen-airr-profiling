from pathlib import Path

import pandas as pd

from gen_airr_bm.core.postprocessing_config import PostProcessingConfig
from gen_airr_bm.data_postprocessing.postprocessing_orchestrator import PostProcessingOrchestrator


def _write_tsv(path: Path, data: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, sep="\t", index=False)


def test_run_postprocessing(tmp_path, monkeypatch):
    root_dir = tmp_path
    config = PostProcessingConfig("modelA", "experiment_2", 5, str(root_dir), 3)
    train_dir = root_dir / "train_compairr_sequences"
    train_path = train_dir / "alpha_2.tsv"
    _write_tsv(train_path, pd.DataFrame({"junction_aa": ["AAA"]}))

    resampled_no_train_compairr_dir = root_dir / "resampled_no_train_compairr_sequences" / config.model_name
    generated_dir = root_dir / "generated_no_train_compairr_sequences_split" / config.model_name
    novel_split_dir = root_dir / "novel_generated_compairr_sequences_split" / config.model_name
    novel_unique_split_dir = root_dir / "novel_unique_generated_compairr_sequences_split" / config.model_name
    resampled_divided_dir = root_dir / "resampled_no_train_compairr_sequences_split" / config.model_name

    def fake_remove_train(cfg, train_path_arg, dataset_name):
        assert cfg is config
        assert train_path_arg == str(train_path)
        assert dataset_name == "alpha_2"
        return str(resampled_divided_dir)

    def fake_preprocess(input_dir, output_dir, filename):
        assert input_dir == str(resampled_divided_dir)
        assert output_dir == str(resampled_no_train_compairr_dir)
        assert filename == "alpha_2.tsv"
        return "preprocessed.tsv"

    def fake_divide(cfg, compairr_path, dataset_name):
        assert cfg is config
        assert compairr_path == "preprocessed.tsv"
        assert dataset_name == "alpha_2"
        return str(resampled_divided_dir)

    def fake_remove_train_generated(cfg, dataset_name, train_path_arg):
        assert cfg is config
        assert dataset_name == "alpha_2"
        assert train_path_arg == str(train_path)
        return str(generated_dir)

    def fake_collect(cfg, generated_no_train_dir, resampled_dir, dataset_name):
        assert cfg is config
        assert generated_no_train_dir == str(generated_dir)
        assert resampled_dir == str(resampled_divided_dir)
        assert dataset_name == "alpha_2"
        return str(novel_split_dir)

    def fake_deduplicate(cfg, novel_dir, dataset_name):
        assert cfg is config
        assert novel_dir == str(novel_split_dir)
        assert dataset_name == "alpha_2"
        return str(novel_unique_split_dir)

    merge_called = {}

    def fake_merge(cfg, novel_dir, dataset_name):
        assert cfg is config
        assert novel_dir == str(novel_split_dir)
        assert dataset_name == "alpha_2"
        merge_called["done"] = True
        return "merged.tsv"

    monkeypatch.setattr(PostProcessingOrchestrator, "remove_train_from_resampled", fake_remove_train)
    monkeypatch.setattr("gen_airr_bm.data_postprocessing.postprocessing_orchestrator.preprocess_file_for_compairr",
                        fake_preprocess)
    monkeypatch.setattr(PostProcessingOrchestrator, "divide_resampled_sequences", fake_divide)
    monkeypatch.setattr(PostProcessingOrchestrator, "remove_train_from_generated", fake_remove_train_generated)
    monkeypatch.setattr(PostProcessingOrchestrator, "collect_novel_sequences_splits", fake_collect)
    monkeypatch.setattr(PostProcessingOrchestrator, "deduplicate_novel_sequences_splits", fake_deduplicate)
    monkeypatch.setattr(PostProcessingOrchestrator, "merge_novel_sequences_splits", fake_merge)

    PostProcessingOrchestrator.run_postprocessing(config)
    assert merge_called["done"]


def test_remove_train_from_resampled(tmp_path):
    root_dir = tmp_path
    config = PostProcessingConfig("test_model", "exp_1", 0, str(root_dir), 2)
    dataset_name = "dataset_1"
    train_path = root_dir / "train.tsv"
    _write_tsv(train_path, pd.DataFrame({"junction_aa": ["AAA", "BBB"], "value": [1, 2]}))

    resampled_dir = root_dir / "resampled_sequences_raw" / config.model_name
    resampled_path = resampled_dir / f"{dataset_name}.tsv"
    _write_tsv(resampled_path, pd.DataFrame({"junction_aa": ["AAA", "CCC", "DDD"], "value": [5, 6, 7]}))

    no_train_dir = PostProcessingOrchestrator.remove_train_from_resampled(
        config, str(train_path), dataset_name)

    expected_dir = root_dir / "resampled_no_train_sequences" / config.model_name
    assert no_train_dir == str(expected_dir)


def test_divide_resampled_sequences(tmp_path):
    root_dir = tmp_path
    config = PostProcessingConfig("test_model", "exp_1", 0, str(root_dir), 2)
    dataset_name = "dataset_1"
    resampled_path = root_dir / "resampled_compairr.tsv"
    _write_tsv(resampled_path, pd.DataFrame({
        "junction_aa": [f"S{i}" for i in range(6)],
        "sequence_id": list(range(6))
    }))

    output_dir = PostProcessingOrchestrator.divide_resampled_sequences(
        config, str(resampled_path), dataset_name)

    expected_dir = root_dir / "resampled_no_train_compairr_sequences_split" / config.model_name
    assert output_dir == str(expected_dir)

    subset_paths = sorted(expected_dir.glob("*.tsv"))
    assert len(subset_paths) == config.n_subsets

    concatenated = []
    for path in subset_paths:
        subset_df = pd.read_csv(path, sep="\t")
        assert len(subset_df) == 3
        concatenated.append(subset_df)
    rebuilt_df = pd.concat(concatenated, ignore_index=True)
    assert set(rebuilt_df["junction_aa"]) == {f"S{i}" for i in range(6)}


def test_remove_train_from_generated(tmp_path):
    root_dir = tmp_path
    config = PostProcessingConfig("test_model", "exp_1", 0, str(root_dir), 2)
    dataset_name = "dataset_1"
    train_path = root_dir / "train.tsv"
    _write_tsv(train_path, pd.DataFrame({"junction_aa": ["AAA", "EEE"]}))

    generated_dir = root_dir / "generated_compairr_sequences_split" / config.model_name
    _write_tsv(generated_dir / f"{dataset_name}_0.tsv",
               pd.DataFrame({"junction_aa": ["AAA", "BBB", "CCC"]}))
    _write_tsv(generated_dir / f"{dataset_name}_1.tsv",
               pd.DataFrame({"junction_aa": ["DDD", "EEE", "FFF"]}))

    output_dir = PostProcessingOrchestrator.remove_train_from_generated(
        config, dataset_name, str(train_path))

    expected_dir = root_dir / "generated_no_train_compairr_sequences_split" / config.model_name
    assert output_dir == str(expected_dir)

    subset_0 = pd.read_csv(expected_dir / f"{dataset_name}_0.tsv", sep="\t")
    subset_1 = pd.read_csv(expected_dir / f"{dataset_name}_1.tsv", sep="\t")
    assert list(subset_0["junction_aa"]) == ["BBB", "CCC"]
    assert list(subset_1["junction_aa"]) == ["DDD", "FFF"]


def test_generate_novel_sequences_splits(tmp_path):
    root_dir = tmp_path
    config = PostProcessingConfig("test_model", "exp_1", 3, str(root_dir), 2)
    dataset_name = "dataset_1"

    generated_dir = root_dir / "generated_no_train_compairr_sequences_split" / config.model_name
    _write_tsv(generated_dir / f"{dataset_name}_0.tsv",
               pd.DataFrame({"junction_aa": ["AAA"], "sequence_id": ["x0"]}))
    _write_tsv(generated_dir / f"{dataset_name}_1.tsv",
               pd.DataFrame({"junction_aa": ["BBB", "CCC", "DDD"], "sequence_id": ["x1", "x2", "x3"]}))

    resampled_dir = root_dir / "resampled_no_train_compairr_sequences_split" / config.model_name
    _write_tsv(resampled_dir / f"{dataset_name}_0.tsv",
               pd.DataFrame({"junction_aa": ["EEE", "FFF", "GGG"], "sequence_id": ["y0", "y1", "y2"]}))
    _write_tsv(resampled_dir / f"{dataset_name}_1.tsv",
               pd.DataFrame({"junction_aa": ["HHH", "III", "JJJ"], "sequence_id": ["y3", "y4", "y5"]}))

    output_dir = PostProcessingOrchestrator.collect_novel_sequences_splits(
        config, str(generated_dir), str(resampled_dir), dataset_name)

    expected_dir = root_dir / "novel_generated_compairr_sequences_split" / config.model_name
    assert output_dir == str(expected_dir)

    subset_0 = pd.read_csv(expected_dir / f"{dataset_name}_0.tsv", sep="\t")
    subset_1 = pd.read_csv(expected_dir / f"{dataset_name}_1.tsv", sep="\t")
    assert len(subset_0) == config.n_samples
    assert len(subset_1) == config.n_samples
    assert list(subset_0["sequence_id"]) == [f"sequence_{i}" for i in range(len(subset_0))]
    assert list(subset_1["sequence_id"]) == [f"sequence_{i}" for i in range(len(subset_1))]
    combined_sequences = set(subset_0["junction_aa"]) | set(subset_1["junction_aa"])
    expected_sequences = {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"}
    assert combined_sequences <= expected_sequences


def test_merge_novel_sequences_splits(tmp_path):
    root_dir = tmp_path
    config = PostProcessingConfig("test_model", "exp_1", 3, str(root_dir), 2)
    dataset_name = "dataset_1"

    novel_split_dir = root_dir / "novel_generated_compairr_sequences_split" / config.model_name
    _write_tsv(novel_split_dir / f"{dataset_name}_0.tsv",
               pd.DataFrame({"junction_aa": ["AAA", "BBB"], "sequence_id": ["sequence_0", "sequence_1"]}))
    _write_tsv(novel_split_dir / f"{dataset_name}_1.tsv",
               pd.DataFrame({"junction_aa": ["CCC", "DDD"], "sequence_id": ["sequence_0", "sequence_1"]}))

    merged_path = PostProcessingOrchestrator.merge_novel_sequences_splits(
        config, str(novel_split_dir), dataset_name)

    expected_dir = root_dir / "novel_generated_compairr_sequences" / config.model_name
    assert merged_path == str(expected_dir / f"{dataset_name}.tsv")

    merged_df = pd.read_csv(merged_path, sep="\t")
    assert list(merged_df["junction_aa"]) == ["AAA", "BBB", "CCC", "DDD"]
    assert list(merged_df["sequence_id"]) == [f"sequence_{i}" for i in range(len(merged_df))]
