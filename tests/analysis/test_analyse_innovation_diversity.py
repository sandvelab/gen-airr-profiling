import os

import pandas as pd
import pytest

from gen_airr_bm.analysis.analyse_innovation_diversity import (
    run_innovation_diversity_analysis,
    save_innovative_sequences_for_compairr,
    count_nearest_neighbors,
    compute_nearest_neighbor_counts,
    plot_nn_counts_across_datasets,
    plot_single_dataset,
    cluster_innovation_sequences,
    plot_cluster_counts,
)
from gen_airr_bm.core.analysis_config import AnalysisConfig

MODULE = "gen_airr_bm.analysis.analyse_innovation_diversity"


@pytest.fixture
def sample_analysis_config():
    return AnalysisConfig(
        analysis="innovation_diversity",
        model_names=["model1", "model2"],
        analysis_output_dir="/tmp/test_output/analysis_innov",
        root_output_dir="/tmp/test_output",
        default_model_name="humanTRB",
        reference_data=["train", "test"],
        n_subsets=3,
        subfolder_name="analysis_subfolder",
        allowed_mismatches=0,
        indels=False,
        receptor_type="TCR",
    )


def test_run_innovation_diversity_analysis(mocker, sample_analysis_config):
    mock_save = mocker.patch(
        "gen_airr_bm.analysis.analyse_innovation_diversity.save_innovative_sequences_for_compairr",
        return_value="/tmp/test_output/innov_dir",
    )
    mock_count = mocker.patch(
        "gen_airr_bm.analysis.analyse_innovation_diversity.count_nearest_neighbors",
        return_value={"model1": mocker.Mock(), "model2": mocker.Mock()},
    )
    mock_plot_nn = mocker.patch(
        "gen_airr_bm.analysis.analyse_innovation_diversity.plot_nn_counts_across_datasets"
    )
    mock_cluster = mocker.patch(
        "gen_airr_bm.analysis.analyse_innovation_diversity.cluster_innovation_sequences",
        return_value={"model1": {"ds_0": {1: 1}}},
    )
    mock_plot_cluster = mocker.patch(
        "gen_airr_bm.analysis.analyse_innovation_diversity.plot_cluster_counts"
    )

    run_innovation_diversity_analysis(sample_analysis_config)

    mock_save.assert_called_once_with(sample_analysis_config)

    assert mock_count.call_count == 2
    # First call: innovation sequences
    assert mock_count.call_args_list[0].args == (
        sample_analysis_config,
        "/tmp/test_output/innov_dir",
        "/tmp/test_output/analysis_innov/nn_counts_innovation",
    )
    # Second call: full generated sequences
    assert mock_count.call_args_list[1].args == (
        sample_analysis_config,
        "/tmp/test_output/novel_generated_compairr_sequences_split",
        "/tmp/test_output/analysis_innov/nn_counts_full_gen",
    )

    assert mock_plot_nn.call_count == 2
    mock_cluster.assert_called_once_with(sample_analysis_config, "/tmp/test_output/innov_dir")
    mock_plot_cluster.assert_called_once()


def make_tmp_config(tmp_path, model_names=("model1", "model2")):
    return AnalysisConfig(
        analysis="innovation_diversity",
        model_names=list(model_names),
        analysis_output_dir=str(tmp_path / "analysis"),
        root_output_dir=str(tmp_path),
        default_model_name="humanTRB",
        reference_data=["train", "test"],
        n_subsets=2,
        subfolder_name="analysis_subfolder",
        receptor_type="TCR",
    )


def write_tsv(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return str(path)


def set_mtime(paths, mtime):
    for path in paths:
        os.utime(path, (mtime, mtime))


def test_save_innovative_sequences_for_compairr(mocker, tmp_path):
    config = make_tmp_config(tmp_path, ["model1"])
    ref_file = write_tsv(tmp_path / "test_compairr_sequences/ds.tsv", pd.DataFrame({"junction_aa": ["B", "C", "D"]}))
    gen_files = [write_tsv(tmp_path / f"gen/model1/ds_{i}.tsv",
                           pd.DataFrame({"junction_aa": ["A", "B", "C"], "other": [1, 2, 3]})) for i in range(2)]
    mocker.patch(f"{MODULE}.get_sequence_files", return_value={ref_file: gen_files})

    out_dir = save_innovative_sequences_for_compairr(config)

    assert out_dir == f"{tmp_path}/innovation_unique_overlap_compairr_sequences_split"
    for i in range(2):
        # Only the generated sequences that are also in the test set
        saved = pd.read_csv(f"{out_dir}/model1/ds_{i}.tsv", sep="\t")
        assert saved["junction_aa"].tolist() == ["B", "C"]
        assert saved.columns.tolist() == ["junction_aa", "other"]


def test_save_innovative_sequences_for_compairr_keeps_up_to_date_files(mocker, tmp_path):
    config = make_tmp_config(tmp_path, ["model1"])
    ref_file = write_tsv(tmp_path / "test_compairr_sequences/ds.tsv", pd.DataFrame({"junction_aa": ["B"]}))
    gen_files = [write_tsv(tmp_path / f"gen/model1/ds_{i}.tsv", pd.DataFrame({"junction_aa": ["A", "B"]}))
                 for i in range(2)]
    mocker.patch(f"{MODULE}.get_sequence_files", return_value={ref_file: gen_files})
    out_dir = save_innovative_sequences_for_compairr(config)
    outputs = [f"{out_dir}/model1/ds_{i}.tsv" for i in range(2)]
    set_mtime([ref_file] + gen_files, 1_000)
    set_mtime(outputs, 2_000)
    # The second subset was regenerated after its innovative sequences were saved
    write_tsv(tmp_path / "gen/model1/ds_1.tsv", pd.DataFrame({"junction_aa": ["B", "B2"]}))
    set_mtime([gen_files[1]], 3_000)
    read_csv = mocker.spy(pd, "read_csv")

    save_innovative_sequences_for_compairr(config)

    # Only the stale subset is computed again: the test file and that subset are read, the other is kept
    assert sorted(call.args[0] for call in read_csv.call_args_list) == sorted([ref_file, gen_files[1]])
    assert os.path.getmtime(outputs[0]) == 2_000
    assert os.path.getmtime(outputs[1]) > 2_000


def make_nn_inputs(tmp_path, model_files):
    """ Innovative sequences per model and the train and test sequences of their repertoires. """
    sequences_dir = tmp_path / "innov_dir"
    files = []
    for model, names in model_files.items():
        for name in names:
            files.append(write_tsv(sequences_dir / model / name, pd.DataFrame({"junction_aa": ["A"]})))
    for dataset in {name.rsplit("_", 1)[0] for names in model_files.values() for name in names}:
        for split in ["train", "test"]:
            files.append(write_tsv(tmp_path / f"{split}_compairr_sequences/{dataset}.tsv",
                                   pd.DataFrame({"junction_aa": ["A"]})))
    set_mtime(files, 1_000)
    return str(sequences_dir)


NN_COUNTS = {"1": 1, "2": 2, "3": 3, ">3": 4, "n_sequences": 10}


def test_count_nearest_neighbors(mocker, tmp_path):
    config = make_tmp_config(tmp_path)
    sequences_dir = make_nn_inputs(tmp_path, {"model1": ["dataset_0.tsv", "dataset_1.tsv"],
                                              "model2": ["dataset_0.tsv"]})
    mock_compute = mocker.patch(f"{MODULE}.compute_nearest_neighbor_counts", return_value=NN_COUNTS)
    output_dir = str(tmp_path / "analysis/nn_counts_innovation")

    result = count_nearest_neighbors(config, sequences_dir, output_dir)

    assert set(result.keys()) == {"model1", "model2", "test"}
    # model1: 2 gen splits + 1 test => 3 calls; model2: 1 gen split + 1 test => 2 calls
    assert mock_compute.call_count == 5
    assert sorted(result["model1"].index) == ["dataset_0", "dataset_1"]
    for _, df in result.items():
        assert all(col in df.columns for col in ["1", "2", "3", ">3", "n_sequences"])
    # The counts are saved per model, named after the model
    assert sorted(os.listdir(f"{output_dir}/counts")) == ["model1_counts.tsv", "model1_test_counts.tsv",
                                                           "model2_counts.tsv", "model2_test_counts.tsv"]


def test_count_nearest_neighbors_reuses_saved_counts(mocker, tmp_path):
    config = make_tmp_config(tmp_path)
    sequences_dir = make_nn_inputs(tmp_path, {"model1": ["dataset_0.tsv", "dataset_1.tsv"],
                                              "model2": ["dataset_0.tsv"]})
    output_dir = str(tmp_path / "analysis/nn_counts_innovation")
    mocker.patch(f"{MODULE}.compute_nearest_neighbor_counts", return_value=NN_COUNTS)
    first = count_nearest_neighbors(config, sequences_dir, output_dir)
    # model2 was regenerated after its counts were saved
    set_mtime([f"{sequences_dir}/model2/dataset_0.tsv"], 10**10)
    mock_compute = mocker.patch(f"{MODULE}.compute_nearest_neighbor_counts", return_value=NN_COUNTS)

    second = count_nearest_neighbors(config, sequences_dir, output_dir)

    # Only model2 is computed again: 1 gen split + 1 test
    assert [call.kwargs["identifier_prefix"] for call in mock_compute.call_args_list] == [
        "dataset_0_model2", "dataset_test"]
    pd.testing.assert_frame_equal(second["model1"], first["model1"], check_dtype=False)


def test_count_nearest_neighbors_recomputes_incomplete_counts(mocker, tmp_path):
    config = make_tmp_config(tmp_path, ["model1"])
    sequences_dir = make_nn_inputs(tmp_path, {"model1": ["dataset_0.tsv"]})
    output_dir = str(tmp_path / "analysis/nn_counts_innovation")
    mocker.patch(f"{MODULE}.compute_nearest_neighbor_counts", return_value=NN_COUNTS)
    count_nearest_neighbors(config, sequences_dir, output_dir)
    # A subset was added, whose file is older than the saved counts
    added = write_tsv(tmp_path / "innov_dir/model1/dataset_1.tsv", pd.DataFrame({"junction_aa": ["A"]}))
    set_mtime([added], 1_000)
    mock_compute = mocker.patch(f"{MODULE}.compute_nearest_neighbor_counts", return_value=NN_COUNTS)

    result = count_nearest_neighbors(config, sequences_dir, output_dir)

    assert mock_compute.call_count == 3
    assert sorted(result["model1"].index) == ["dataset_0", "dataset_1"]


def test_compute_nearest_neighbor_counts(mocker):
    mocker.patch("gen_airr_bm.analysis.analyse_innovation_diversity.run_compairr_existence")

    df_d1 = pd.DataFrame({"sequence_id": [1, 2, 3, 4], "overlap_count": [1, 0, 3, 0]})
    df_d2 = pd.DataFrame({"sequence_id": [1, 2, 3, 4], "overlap_count": [1, 5, 3, 0]})
    df_d3 = pd.DataFrame({"sequence_id": [1, 2, 3, 4], "overlap_count": [1, 5, 3, 0]})

    mocker.patch("pandas.read_csv", side_effect=[df_d1, df_d2, df_d3])

    counts = compute_nearest_neighbor_counts(
        compairr_output_dir="/tmp/out",
        search_for_file="/tmp/gen.tsv",
        search_in_file="/tmp/train.tsv",
        identifier_prefix="ds_model",
        distances=[1, 2, 3],
    )

    assert counts == {"1": 2, "2": 1, "3": 0, ">3": 1, "n_sequences": 4}


def test_plot_nearest_neighbor_counts(mocker, sample_analysis_config):
    mock_fig = mocker.Mock()
    mocker.patch(
        "gen_airr_bm.analysis.analyse_innovation_diversity.plot_single_dataset",
        return_value=mock_fig,
    )

    plotting_dfs = {
        "model1": pd.DataFrame(
            {"1": [1], "2": [2], "3": [3], ">3": [4]}, index=["ds_0"]
        ),
        "test": pd.DataFrame({"1": [1], "2": [1], "3": [1], ">3": [1]}, index=["ds"]),
    }

    plot_nn_counts_across_datasets(
        sample_analysis_config,
        plotting_dfs,
        f"{sample_analysis_config.analysis_output_dir}/nearest_neighbor_counts",
    )

    assert mock_fig.write_image.call_count == 2
    first_path = mock_fig.write_image.call_args_list[0].args[0]
    assert "/nearest_neighbor_counts/" in first_path


def test_make_distance_figure(mocker):
    mocker.patch("gen_airr_bm.analysis.analyse_innovation_diversity.go.Figure")
    mock_go_Scatter = mocker.patch("gen_airr_bm.analysis.analyse_innovation_diversity.go.Scatter")

    fig = mocker.Mock()
    fig.data = []
    mocker.patch(
        "gen_airr_bm.analysis.analyse_innovation_diversity.go.Figure", return_value=fig
    )

    dfs = {
        "modelB": pd.DataFrame(
            {"1": [1, 2], "2": [2, 2], "3": [3, 1], ">3": [0, 1]}, index=["ds_0", "ds_1"]
        ),
        "test": pd.DataFrame(
            {"1": [0, 1], "2": [1, 0], "3": [0, 1], ">3": [2, 2]}, index=["ds", "ds2"]
        ),
        "modelA": pd.DataFrame({"1": [5], "2": [4], "3": [3], ">3": [2]}, index=["ds_0"]),
    }

    def scatter_side_effect(**kwargs):
        trace = mocker.Mock()
        trace.name = kwargs.get("name")
        return trace

    mock_go_Scatter.side_effect = scatter_side_effect

    out = plot_single_dataset(
        dfs,
        title="t",
        xtitle="x",
        ytitle="y",
        distance_cols=["1", "2", "3", ">3"],
    )

    assert out is fig
    assert mock_go_Scatter.call_count == len(dfs)
    assert fig.add_trace.call_count == len(dfs)
    fig.update_layout.assert_called_once()


def test_cluster_innovation_sequences(mocker, tmp_path):
    config = make_tmp_config(tmp_path)
    sequences_dir = make_nn_inputs(tmp_path, {"model1": ["ds_0.tsv", "ds_1.tsv"], "model2": ["ds_0.tsv"]})
    cluster_files = {}
    for i, n_clusters in enumerate([2, 3, 1]):
        cluster_files[i] = write_tsv(tmp_path / f"r{i}.tsv", pd.DataFrame({"#cluster_no": list(range(n_clusters))}))
    mock_cluster = mocker.patch(f"{MODULE}.run_compairr_cluster",
                                side_effect=lambda compairr_output_dir, sequences_path, output_file_identifier,
                                distance: cluster_files[distance - 1])

    out = cluster_innovation_sequences(config, sequences_dir)

    assert set(out.keys()) == {"model1", "model2"}
    assert out["model1"]["ds_0"] == {1: 2, 2: 3, 3: 1}
    assert mock_cluster.call_count == 9

    # A second run reuses the saved counts, with the same distances as keys
    mock_cluster.reset_mock()
    assert cluster_innovation_sequences(config, sequences_dir) == out
    assert mock_cluster.call_count == 0


def test_plot_cluster_counts(mocker, sample_analysis_config):
    mock_fig = mocker.Mock()
    mocker.patch(
        "gen_airr_bm.analysis.analyse_innovation_diversity.plot_single_dataset",
        return_value=mock_fig,
    )

    num_clusters_by_model = {
        "model1": {"ds_0": {1: 2, 2: 3, 3: 1}, "ds_1": {1: 1, 2: 1, 3: 1}},
        "model2": {"ds_0": {1: 5, 2: 4, 3: 3}},
    }

    plot_cluster_counts(sample_analysis_config, num_clusters_by_model)

    assert mock_fig.write_image.call_count == 2
    first_path = mock_fig.write_image.call_args_list[0].args[0]
    assert "/clustering/" in first_path
