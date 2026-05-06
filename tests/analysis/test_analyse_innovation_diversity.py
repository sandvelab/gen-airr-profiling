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
        return_value={"model1": mocker.Mock(), "test": mocker.Mock()},
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
    mock_count.assert_called_once_with(sample_analysis_config, "/tmp/test_output/innov_dir")
    mock_plot_nn.assert_called_once()
    mock_cluster.assert_called_once_with(sample_analysis_config, "/tmp/test_output/innov_dir")
    mock_plot_cluster.assert_called_once()


def test_save_innovative_sequences_for_compairr(mocker, sample_analysis_config):
    mocker.patch("os.makedirs")
    comparison_mapping = {"/ref/test_ds.tsv": ["/gen/ds_0.tsv", "/gen/ds_1.tsv"]}
    mock_get_sequence_files = mocker.patch(
        "gen_airr_bm.analysis.analyse_innovation_diversity.get_sequence_files",
        return_value=comparison_mapping,
    )

    gen_df = pd.DataFrame({"junction_aa": ["A", "B", "C"], "other": [1, 2, 3]})
    ref_df = pd.DataFrame({"junction_aa": ["B", "C", "D"]})

    # Implementation reads (gen, ref) per gen file per model:
    # 2 models * 2 gen files * 2 reads = 8
    mocker.patch(
        "pandas.read_csv",
        side_effect=[gen_df, ref_df] * (len(sample_analysis_config.model_names) * 2),
    )
    mock_to_csv = mocker.patch("pandas.DataFrame.to_csv")

    out_dir = save_innovative_sequences_for_compairr(sample_analysis_config)

    assert out_dir == "/tmp/test_output/innovation_unique_overlap_compairr_sequences_split"

    assert mock_get_sequence_files.call_count == len(sample_analysis_config.model_names)
    for i, model in enumerate(sample_analysis_config.model_names):
        assert mock_get_sequence_files.call_args_list[i].args == (
            sample_analysis_config,
            model,
            "test",
        )

    expected_writes = len(sample_analysis_config.model_names) * sum(
        len(v) for v in comparison_mapping.values()
    )
    assert mock_to_csv.call_count == expected_writes

    for call in mock_to_csv.call_args_list:
        assert call.kwargs["sep"] == "\t"
        assert call.kwargs["index"] is False


def test_count_nearest_neighbors(mocker, sample_analysis_config):
    mocker.patch("os.makedirs")

    def listdir_side_effect(path):
        if path.endswith("/model1"):
            return ["dataset_0.tsv", "dataset_1.tsv"]
        if path.endswith("/model2"):
            return ["dataset_0.tsv"]
        return []

    mocker.patch("os.listdir", side_effect=listdir_side_effect)

    mock_compute = mocker.patch(
        "gen_airr_bm.analysis.analyse_innovation_diversity.compute_nearest_neighbor_counts",
        side_effect=[
            {"1": 1, "2": 2, "3": 3, ">3": 4, "n_sequences": 10},
            {"1": 2, "2": 1, "3": 0, ">3": 7, "n_sequences": 10},
            {"1": 0, "2": 0, "3": 1, ">3": 9, "n_sequences": 10},
            {"1": 3, "2": 2, "3": 1, ">3": 4, "n_sequences": 10},
            {"1": 2, "2": 2, "3": 2, ">3": 4, "n_sequences": 10},
        ],
    )

    result = count_nearest_neighbors(sample_analysis_config, "/tmp/test_output/innov_dir")

    assert set(result.keys()) >= {"model1", "model2", "test"}
    assert mock_compute.call_count == 5

    for _, df in result.items():
        assert all(col in df.columns for col in ["1", "2", "3", ">3", "n_sequences"])


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

    plot_nn_counts_across_datasets(sample_analysis_config, plotting_dfs)

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
    fig.update_traces.assert_called_once()


def test_cluster_innovation_sequences(mocker, sample_analysis_config):
    mocker.patch("os.makedirs")

    def listdir_side_effect(path):
        if path.endswith("/model1"):
            return ["ds_0.tsv", "ds_1.tsv"]
        if path.endswith("/model2"):
            return ["ds_0.tsv"]
        return []

    mocker.patch("os.listdir", side_effect=listdir_side_effect)

    mocker.patch(
        "gen_airr_bm.analysis.analyse_innovation_diversity.run_compairr_cluster",
        side_effect=[
            "/tmp/r1.tsv",
            "/tmp/r2.tsv",
            "/tmp/r3.tsv",
            "/tmp/r4.tsv",
            "/tmp/r5.tsv",
            "/tmp/r6.tsv",
            "/tmp/r7.tsv",
            "/tmp/r8.tsv",
            "/tmp/r9.tsv",
        ],
    )

    def read_csv_side_effect(path, sep="\t"):
        if path in {"/tmp/r1.tsv", "/tmp/r4.tsv", "/tmp/r7.tsv"}:
            return pd.DataFrame({"#cluster_no": [1, 1, 2]})
        if path in {"/tmp/r2.tsv", "/tmp/r5.tsv", "/tmp/r8.tsv"}:
            return pd.DataFrame({"#cluster_no": [1, 2, 3, 3]})
        return pd.DataFrame({"#cluster_no": [1]})

    mocker.patch("pandas.read_csv", side_effect=read_csv_side_effect)

    out = cluster_innovation_sequences(sample_analysis_config, "/tmp/test_output/innov_dir")

    assert set(out.keys()) == set(sample_analysis_config.model_names)


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
