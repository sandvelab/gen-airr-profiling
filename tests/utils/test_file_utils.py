import pytest
import yaml

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.core.main_config import MainConfig
from gen_airr_bm.utils.file_utils import get_generated_sequences_dir_name, get_sequence_files


def make_analysis_config(root_output_dir="/tmp/test_output", analysis="precision_recall", receptor_type="TCR",
                         **kwargs):
    return AnalysisConfig(analysis=analysis, model_names=["model1"], analysis_output_dir=f"{root_output_dir}/analysis",
                          root_output_dir=root_output_dir, default_model_name="humanTRB", reference_data=["test"],
                          subfolder_name="subfolder", receptor_type=receptor_type, n_subsets=2, **kwargs)


@pytest.mark.parametrize("receptor_type,use_novel_sequences,split,expected", [
    ("TCR", True, True, "novel_generated_compairr_sequences_split"),
    ("TCR", True, False, "novel_generated_compairr_sequences"),
    ("TCR", False, True, "generated_compairr_sequences_split"),
    ("TCR", False, False, "generated_compairr_sequences"),
    ("BCR UMI", True, True, "generated_compairr_sequences_split"),
])
def test_get_generated_sequences_dir_name(receptor_type, use_novel_sequences, split, expected):
    config = make_analysis_config(receptor_type=receptor_type, use_novel_sequences=use_novel_sequences)

    assert get_generated_sequences_dir_name(config, split=split) == expected


def test_analysis_config_uses_novel_sequences_by_default():
    assert make_analysis_config().use_novel_sequences is True


@pytest.mark.parametrize("use_novel_sequences,gen_dir_name", [
    (True, "novel_generated_compairr_sequences_split"),
    (False, "generated_compairr_sequences_split"),
])
def test_get_sequence_files_follows_use_novel_sequences(tmp_path, use_novel_sequences, gen_dir_name):
    (tmp_path / "test_compairr_sequences").mkdir()
    (tmp_path / "test_compairr_sequences" / "dataset.tsv").touch()
    gen_dir = tmp_path / gen_dir_name / "model1"
    gen_dir.mkdir(parents=True)
    for i in range(2):
        (gen_dir / f"dataset_{i}.tsv").touch()
    config = make_analysis_config(root_output_dir=str(tmp_path), use_novel_sequences=use_novel_sequences)

    files = get_sequence_files(config, "model1", "test")

    assert files == {str(tmp_path / "test_compairr_sequences" / "dataset.tsv"):
                     {str(gen_dir / "dataset_0.tsv"), str(gen_dir / "dataset_1.tsv")}}


@pytest.mark.parametrize("config_value,expected", [(None, True), (False, False)])
def test_main_config_reads_use_novel_sequences(tmp_path, config_value, expected):
    analysis = {"name": "precision_recall", "model_names": ["model1"], "default_model_name": "humanTRB",
                "subfolder_name": "subfolder", "receptor_type": "TCR"}
    if config_value is not None:
        analysis["use_novel_sequences"] = config_value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({"n_experiments": 1, "output_dir": str(tmp_path / "out"),
                                      "input_dir": str(tmp_path), "seed": 42, "analyses": [analysis]}))

    main_config = MainConfig(str(config_path))

    assert main_config.analysis_configs[0].use_novel_sequences is expected
