import numpy as np
import pandas as pd
import pytest
import yaml

from gen_airr_bm.core.main_config import MainConfig
from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.plotting_utils import (get_collection_specification_for_title, title_case,
                                              plot_avg_innovation_scores, plot_grouped_bar_precision_recall)


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


@pytest.mark.parametrize("text,expected", [
    ("connectivity", "Connectivity"),
    ("aa (Seq. Len.: 12)", "Aa (Seq. Len.: 12)"),
    ("VJ pairing", "VJ Pairing"),
    ("clone frequency", "Clone Frequency"),
])
def test_title_case_keeps_capitalised_words(text, expected):
    assert title_case(text) == expected


def test_plot_grouped_bar_precision_recall_sd_across_repertoires(tmp_path):
    precision = {"rep_1": {"model1": [0.1, 0.3]}, "rep_2": {"model1": [0.5, 0.7]}}
    recall = {"rep_1": {"model1": [0.2, 0.2]}, "rep_2": {"model1": [0.4, 0.4]}}
    (tmp_path / "test").mkdir()
    # A stale file from an earlier run is replaced
    (tmp_path / "test" / "precision_recall_data.tsv").write_text("stale\n")

    plot_grouped_bar_precision_recall(precision, recall, str(tmp_path), "test", "BCR UMI", 0)

    data = pd.read_csv(tmp_path / "test" / "precision_recall_data.tsv", sep="\t")
    # Repertoire means: precision 0.2 and 0.6, recall 0.2 and 0.4
    assert data["Precision_mean"].tolist() == pytest.approx([0.4])
    assert data["Precision_std"].tolist() == pytest.approx([np.std([0.2, 0.6], ddof=1)])
    assert data["Recall_std"].tolist() == pytest.approx([np.std([0.2, 0.4], ddof=1)])


def test_plot_avg_innovation_scores_with_repertoire_dots(tmp_path):
    analysis_config = AnalysisConfig(analysis="precision_recall", model_names=["model1", "model2"],
                                     analysis_output_dir=str(tmp_path), root_output_dir=str(tmp_path),
                                     default_model_name="humanIGH", reference_data=["train", "test"], n_subsets=2,
                                     subfolder_name="all models", receptor_type="BCR UMI")

    plot_avg_innovation_scores(analysis_config, {"model1": 0.2, "model2": 0.5}, {"model1": 0.1, "model2": 0.05},
                               str(tmp_path), "recall", "recall_overview", "recall", scoring_method="recall",
                               error_bar_caption="Error bars: SD across repertoires (n=2); dots: repertoires",
                               repertoire_scores={"model1": [("d1", 0.1), ("d2", 0.3)],
                                                  "model2": [("d1", 0.45), ("d2", 0.55)]})

    assert (tmp_path / "recall" / "recall_overview.png").exists()
    data = pd.read_csv(tmp_path / "recall" / "recall_overview.tsv", sep="\t")
    assert data["Model"].tolist() == ["model2", "model1"]
