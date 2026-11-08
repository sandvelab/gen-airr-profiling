import os

import numpy as np
import pandas as pd
import pytest

from gen_airr_bm.analysis.analyse_clonal_clusters import (
    bin_cluster_sizes,
    cluster_size_bin,
    cluster_size_bin_label,
    get_cluster_sizes,
    run_clonal_clusters_analysis,
    summarise_cluster_sizes,
    to_share_rows,
)
from gen_airr_bm.analysis.analysis_orchestrator import AnalysisOrchestrator
from gen_airr_bm.core.analysis_config import AnalysisConfig

MODULE = "gen_airr_bm.analysis.analyse_clonal_clusters"


def write_tsv(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return str(path)


def sequences(junctions):
    return pd.DataFrame({"junction_aa": junctions, "v_call": "IGHV1-2*01", "j_call": "IGHJ4*02",
                         "duplicate_count": 1})


def fake_compairr_cluster(cluster_of):
    """ Returns a stand-in for run_compairr_cluster that writes CompAIRR-like output, assigning clusters with
    cluster_of(junction_aa) instead of running CompAIRR. """
    def run(compairr_output_dir, sequences_path, output_file_identifier, distance):
        df = pd.read_csv(sequences_path, sep="\t")
        out = pd.DataFrame({"#cluster_no": [cluster_of(j) for j in df["junction_aa"]],
                            "junction_aa": df["junction_aa"]})
        path = f"{compairr_output_dir}/{output_file_identifier}.tsv"
        out.to_csv(path, sep="\t", index=False)
        return path
    return run


@pytest.fixture
def config(tmp_path):
    return AnalysisConfig(analysis="clonal_clusters", model_names=["modelA"],
                          analysis_output_dir=str(tmp_path / "analyses/clonal_clusters/all_models"),
                          root_output_dir=str(tmp_path), default_model_name="humanIGH",
                          reference_data=["train", "test"], subfolder_name="all models", receptor_type="BCR",
                          n_subsets=2, allowed_mismatches=1, donor_pattern="^([^_]+)_")


def test_registered_in_orchestrator():
    assert AnalysisOrchestrator.ANALYSES_METHODS["clonal_clusters"] is run_clonal_clusters_analysis


@pytest.mark.parametrize("size, lower", [(1, 1), (2, 2), (3, 3), (4, 3), (5, 5), (8, 5), (9, 9), (16, 9), (17, 17)])
def test_cluster_size_bin(size, lower):
    assert cluster_size_bin(size) == lower


@pytest.mark.parametrize("lower, label", [(1, "1"), (2, "2"), (3, "3-4"), (5, "5-8"), (9, "9-16")])
def test_cluster_size_bin_label(lower, label):
    assert cluster_size_bin_label(lower) == label


def test_bin_cluster_sizes_counts_sequences_per_bin():
    sizes = pd.Series([1, 1, 1, 2, 3, 4, 7])
    binned = bin_cluster_sizes(sizes)
    assert binned.name == "count"
    assert binned.to_dict() == {1: 3, 2: 2, 3: 7, 5: 7}


def test_summarise_cluster_sizes():
    summary = summarise_cluster_sizes(pd.Series([1, 1, 3, 5]), "test", "rep_1", np.nan)
    assert summary["n_sequences"] == 10
    assert summary["n_clusters"] == 4
    assert summary["clusters_per_sequence"] == pytest.approx(0.4)
    assert summary["singleton_share"] == pytest.approx(0.2)
    assert summary["largest_cluster_share"] == pytest.approx(0.5)


def test_to_share_rows_sum_to_one():
    rows = to_share_rows(pd.Series({1: 2, 3: 6}, name="count"), "modelA", "rep_1", 0)
    assert sum(r["share"] for r in rows) == pytest.approx(1)
    assert [r["cluster_size_bin"] for r in rows] == [1, 3]


def test_get_cluster_sizes_deduplicates_and_counts(mocker, tmp_path):
    seq_file = write_tsv(tmp_path / "in/rep_1.tsv", sequences(["CARDYW", "CARDYW", "CARDFW", "CTTTTW"]))
    helper, out = tmp_path / "helper", tmp_path / "out"
    helper.mkdir(), out.mkdir()
    mock_cluster = mocker.patch(f"{MODULE}.run_compairr_cluster",
                                side_effect=fake_compairr_cluster(lambda j: 1 if j.startswith("CARD") else 2))

    sizes = get_cluster_sizes(seq_file, "test", str(helper), str(out), 1)

    assert sorted(sizes.tolist()) == [1, 2]
    assert len(pd.read_csv(helper / "rep_1_test_unique.tsv", sep="\t")) == 3
    assert mock_cluster.call_args.kwargs["output_file_identifier"] == "rep_1_test_d1"
    assert mock_cluster.call_args.kwargs["distance"] == 1


def test_get_cluster_sizes_removes_stale_output(mocker, tmp_path):
    seq_file = write_tsv(tmp_path / "in/rep_1.tsv", sequences(["CARDYW"]))
    helper, out = tmp_path / "helper", tmp_path / "out"
    helper.mkdir(), out.mkdir()
    stale = out / "rep_1_test_d1.tsv"
    stale.write_text("#cluster_no\tjunction_aa\n")
    os.utime(stale, (0, 0))
    mocker.patch(f"{MODULE}.run_compairr_cluster", side_effect=fake_compairr_cluster(lambda j: 1))

    sizes = get_cluster_sizes(seq_file, "test", str(helper), str(out), 1)

    assert sizes.tolist() == [1]


def test_run_rejects_zero_distance(config):
    config.allowed_mismatches = 0
    with pytest.raises(ValueError, match="allowed_mismatches"):
        run_clonal_clusters_analysis(config)


def test_run_writes_scores_and_summaries(mocker, config, tmp_path):
    root = tmp_path
    write_tsv(root / "train_compairr_sequences/d1_rep.tsv", sequences(["CARDYW", "CARDFW", "CTTTTW", "CGGGGW"]))
    write_tsv(root / "test_compairr_sequences/d1_rep.tsv", sequences(["CARDYW", "CARDFW", "CARGFW", "CTTTTW"]))
    for subset in [0, 1]:
        write_tsv(root / f"novel_generated_compairr_sequences_split/modelA/d1_rep_{subset}.tsv",
                  sequences(["CARDYW", "CARDFW", "CARGFW", "CARGYW"]))
    mocker.patch(f"{MODULE}.run_compairr_cluster",
                 side_effect=fake_compairr_cluster(lambda j: {"CA": 1, "CT": 2, "CG": 3}[j[:2]]))
    mock_plots = [mocker.patch(f"{MODULE}.{name}") for name in
                  ["plot_cluster_size_jsd", "run_cluster_size_statistics", "plot_cluster_summary_statistics",
                   "plot_cluster_size_distributions"]]

    run_clonal_clusters_analysis(config)

    out = config.analysis_output_dir
    scores = pd.read_csv(f"{out}/cluster_size_jsd_scores.tsv", sep="\t")
    assert len(scores) == 4  # 2 references x 2 subsets
    # Generated: one cluster of 4. Test: a cluster of 3 and a singleton. Train: a cluster of 2 and two singletons.
    assert (scores["jsd"] > 0).all()
    summaries = pd.read_csv(f"{out}/cluster_summary_statistics.tsv", sep="\t")
    assert sorted(summaries["source"]) == ["modelA", "modelA", "test", "train"]
    generated = summaries[summaries["source"] == "modelA"].iloc[0]
    assert generated["n_clusters"] == 1 and generated["largest_cluster_share"] == 1
    reference = pd.read_csv(f"{out}/cluster_size_jsd_train_vs_test.tsv", sep="\t")
    assert reference["repertoire"].tolist() == ["d1_rep"]
    shares = pd.read_csv(f"{out}/cluster_size_bin_shares.tsv", sep="\t")
    assert shares.groupby(["source", "dataset", "subset"], dropna=False)["share"].sum().round(6).eq(1).all()
    for mock_plot in mock_plots:
        mock_plot.assert_called_once()
