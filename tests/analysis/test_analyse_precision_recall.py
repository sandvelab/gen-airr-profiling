import pytest
import pandas as pd

from gen_airr_bm.analysis.analyse_precision_recall import (
    run_precision_recall_analysis,
    compute_precision_recall_scores,
    collect_model_scores,
    add_upper_reference,
    get_precision_recall_metrics,
    get_precision_recall_reference,
    compute_compairr_overlap_ratio,
    ScoreStorage,
)
from gen_airr_bm.core.analysis_config import AnalysisConfig


@pytest.fixture
def sample_analysis_config():
    return AnalysisConfig(
        analysis="precision_recall",
        model_names=["modelA", "modelB"],
        analysis_output_dir="/tmp/test_output/analysis",
        root_output_dir="/tmp/test_output",
        default_model_name="humanTRB",
        reference_data=["train", "test"],  # include both for upper reference
        n_subsets=3,
        n_unique_samples=10,
    )


def test_run_precision_recall_analysis_full_pipeline(mocker, sample_analysis_config):
    os_makedirs = mocker.patch("os.makedirs")
    mock_compute = mocker.patch("gen_airr_bm.analysis.analyse_precision_recall.compute_precision_recall_scores")

    run_precision_recall_analysis(sample_analysis_config)

    os_makedirs.assert_any_call("/tmp/test_output/analysis", exist_ok=True)
    os_makedirs.assert_any_call("/tmp/test_output/analysis/compairr_output", exist_ok=True)
    mock_compute.assert_called_once()


def test_compute_precision_recall_scores_happy_path(mocker, sample_analysis_config):
    # Mocks for low-level helpers and plotting
    mock_collect = mocker.patch(
        "gen_airr_bm.analysis.analyse_precision_recall.collect_model_scores", autospec=True
    )
    mock_add_ref = mocker.patch(
        "gen_airr_bm.analysis.analyse_precision_recall.add_upper_reference", autospec=True
    )
    mock_plot_avg = mocker.patch("gen_airr_bm.analysis.analyse_precision_recall.plot_avg_scores")
    mock_plot_grouped = mocker.patch("gen_airr_bm.analysis.analyse_precision_recall.plot_grouped_bar_precision_recall")

    # Patch ScoreStorage so mean_precision dict isn't just empty
    fake_scores = ScoreStorage()
    datasets = ["ds1", "ds2"]
    for ds in datasets:
        fake_scores.mean_precision[ds] = {"modelA": 0.8, "modelB": 0.5}
        fake_scores.std_precision[ds] = {"modelA": 0.05, "modelB": 0.10}
        fake_scores.mean_recall[ds] = {"modelA": 0.7, "modelB": 0.6}
        fake_scores.std_recall[ds] = {"modelA": 0.03, "modelB": 0.09}
        fake_scores.precision_all[ds] = {"modelA": [0.8], "modelB": [0.5]}
        fake_scores.recall_all[ds] = {"modelA": [0.7], "modelB": [0.6]}
    mock_collect.side_effect = lambda ac, m, t, c, s: fake_scores

    # Patch ScoreStorage instantiation
    mocker.patch("gen_airr_bm.analysis.analyse_precision_recall.ScoreStorage", return_value=fake_scores)

    compute_precision_recall_scores(sample_analysis_config, "/tmp/test_output/analysis/compairr_output")
    assert mock_collect.call_count == len(sample_analysis_config.model_names)
    mock_add_ref.assert_called_once()
    # Should plot for each dataset
    assert mock_plot_avg.call_count == 2 * len(datasets)
    mock_plot_grouped.assert_called_once()


def test_collect_model_scores_invokes_deps_and_updates_scores(mocker, sample_analysis_config):
    # Setup fixture and mocks
    mock_get_seq_files = mocker.patch(
        "gen_airr_bm.analysis.analyse_precision_recall.get_sequence_files",
        return_value={
            "/tmp/test_output/test_compairr_sequences/ds1.tsv": {
                "/tmp/test_output/generated_compairr_sequences_split/modelA/ds1_0.tsv"},
            "/tmp/test_output/test_compairr_sequences/ds2.tsv": {
                "/tmp/test_output/generated_compairr_sequences_split/modelA/ds2_0.tsv"}
        }
    )
    mock_get_metrics = mocker.patch(
        "gen_airr_bm.analysis.analyse_precision_recall.get_precision_recall_metrics",
        side_effect=lambda ref, gen, out, model: ([0.9], [0.8])
    )
    storage = ScoreStorage()
    collect_model_scores(sample_analysis_config, "modelA", "test", "/tmp/compairr_out", storage)
    assert storage.mean_precision["ds1"]["modelA"] == 0.9
    assert storage.std_recall["ds2"]["modelA"] == 0.0  # Only one value, std=0


def test_add_upper_reference_flows_and_updates_scores(mocker, sample_analysis_config):
    mock_get_ref = mocker.patch(
        "gen_airr_bm.analysis.analyse_precision_recall.get_precision_recall_reference",
        return_value=(0.99, 0.88)
    )
    ss = ScoreStorage()
    # Must pre-populate mean_precision keys for loop
    ss.mean_precision["ds3"] = {}
    add_upper_reference(sample_analysis_config, "train", "test", ss, "/tmp/compairr_out")
    assert ss.precision_all["ds3"]["upper_reference"] == [0.99]
    assert ss.recall_all["ds3"]["upper_reference"] == [0.88]


def test_get_precision_recall_metrics(mocker):
    mock_overlap = mocker.patch(
        "gen_airr_bm.analysis.analyse_precision_recall.compute_compairr_overlap_ratio",
        side_effect=[0.81, 0.61]
    )
    precision, recall = get_precision_recall_metrics("ref.tsv", ["gen1.tsv"], "/tmp/out", "myModel")
    mock_overlap.assert_any_call("gen1.tsv", "ref.tsv", "/tmp/out", "myModel", "precision")
    mock_overlap.assert_any_call("ref.tsv", "gen1.tsv", "/tmp/out", "myModel", "recall")
    assert precision == [0.81]
    assert recall == [0.61]


def test_get_precision_recall_reference(mocker):
    mock_overlap = mocker.patch("gen_airr_bm.analysis.analyse_precision_recall.compute_compairr_overlap_ratio",
                                side_effect=[0.76, 0.55])
    p, r = get_precision_recall_reference("train.tsv", "test.tsv", "/tmp/out")
    mock_overlap.assert_any_call("train.tsv", "test.tsv", "/tmp/out", "upper_reference", "precision")
    mock_overlap.assert_any_call("test.tsv", "train.tsv", "/tmp/out", "upper_reference", "recall")
    assert p == 0.76
    assert r == 0.55


def test_compute_compairr_overlap_ratio_reads_result(mocker, tmp_path):
    out_dir = tmp_path
    simple_path = out_dir / "somefile.tsv"
    # The path that will be READ is (according to function logic):
    overlap_fname = f"{simple_path.stem}_modelA_precision_overlap.tsv"
    overlap_path = out_dir / overlap_fname
    df = pd.DataFrame({"sequence_id": ["id1", "id2", "id3"], "overlap_count": [1, 0, 2]})
    df.to_csv(overlap_path, sep="\t", index=False)
    mocker.patch("gen_airr_bm.analysis.analyse_precision_recall.run_compairr_existence", return_value=None)
    ratio = compute_compairr_overlap_ratio(
        str(simple_path), "other.tsv", str(out_dir), "modelA", "precision"
    )
    assert ratio == pytest.approx(2 / 3, 0.01)

