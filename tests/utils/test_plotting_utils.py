import pytest
import yaml

from gen_airr_bm.core.main_config import MainConfig
from gen_airr_bm.utils.plotting_utils import get_collection_specification_for_title


@pytest.mark.parametrize("receptor_type,expected", [
    ("TCR", "Collection B (TCR)"),
    ("BCR", "Collection C (BCR)"),
    ("BCR UMI", "Collection A (BCR)"),
])
def test_get_collection_specification_for_title_infers_collection(receptor_type, expected):
    assert get_collection_specification_for_title(receptor_type) == expected


@pytest.mark.parametrize("receptor_type,collection,expected", [
    ("TCR", "D", "Collection D (TCR)"),
    ("TCR", "E", "Collection E (TCR)"),
    ("BCR UMI", "F", "Collection F (BCR)"),
])
def test_get_collection_specification_for_title_uses_explicit_collection(receptor_type, collection, expected):
    assert get_collection_specification_for_title(receptor_type, collection) == expected


def test_get_collection_specification_for_title_rejects_unknown_receptor_type():
    with pytest.raises(ValueError):
        get_collection_specification_for_title("TRA")


@pytest.mark.parametrize("config_value", [None, "D"])
def test_main_config_reads_collection(tmp_path, config_value):
    analysis = {"name": "gene_usage", "model_names": ["model1"], "default_model_name": "humanTRB",
                "subfolder_name": "subfolder", "receptor_type": "TCR"}
    if config_value is not None:
        analysis["collection"] = config_value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({"n_experiments": 1, "output_dir": str(tmp_path / "out"),
                                      "input_dir": str(tmp_path), "seed": 42, "analyses": [analysis]}))

    main_config = MainConfig(str(config_path))

    assert main_config.analysis_configs[0].collection == config_value
