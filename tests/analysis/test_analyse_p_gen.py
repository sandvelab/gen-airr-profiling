import numpy as np
import pandas as pd
import pytest

from gen_airr_bm.analysis.analyse_p_gen import collect_p_gens, compute_auc, summarize_p_gens, run_p_gen_analysis, \
    compute_density, get_density_grid, get_grid_shape, get_family_colors, plot_p_gen_distributions
from gen_airr_bm.core.analysis_config import AnalysisConfig


def write_p_gens(root_dir, split, model, file_name, p_gens):
    p_gens_dir = root_dir / f"{split}_p_gens" / model
    p_gens_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"junction_aa": [f"CAS{i}F" for i in range(len(p_gens))], "v_call": "TRBV1", "j_call": "TRBJ1",
                  "p_gen_model": p_gens}).to_csv(p_gens_dir / f"{file_name}.tsv", sep="\t", index=False)


@pytest.fixture
def analysis_config(tmp_path):
    root_dir = tmp_path / "root"
    write_p_gens(root_dir, "train", "modelA", "rep_0", [1e-10, 1e-12, 1e-14])
    write_p_gens(root_dir, "test", "modelA", "rep_0", [1e-12, 1e-14, 0.0])
    write_p_gens(root_dir, "train", "modelB", "rep_0", [1e-20, 1e-20])
    write_p_gens(root_dir, "test", "modelB", "rep_0", [1e-20, 1e-22])
    return AnalysisConfig(analysis="p_gen", model_names=["modelA", "modelB"],
                          analysis_output_dir=str(tmp_path / "analyses" / "p_gen"), root_output_dir=str(root_dir),
                          default_model_name="humanTRB", reference_data=["train", "test"], subfolder_name="",
                          receptor_type="TCR")


def test_collect_p_gens(analysis_config):
    p_gens_df = collect_p_gens(analysis_config)

    assert list(p_gens_df.columns) == ["model", "split", "file", "junction_aa", "p_gen"]
    assert len(p_gens_df) == 10
    assert p_gens_df.groupby(["model", "split"]).size().to_dict() == {
        ("modelA", "test"): 3, ("modelA", "train"): 3, ("modelB", "test"): 2, ("modelB", "train"): 2}


def test_collect_p_gens_missing_model(analysis_config):
    analysis_config.model_names = ["modelA", "modelC"]

    with pytest.raises(FileNotFoundError):
        collect_p_gens(analysis_config)


def test_summarize_p_gens(analysis_config):
    summary_df = summarize_p_gens(collect_p_gens(analysis_config)).set_index("model")

    assert summary_df.loc["modelA", "n_zero_test"] == 1
    assert summary_df.loc["modelA", "median_log10_p_gen_train"] == pytest.approx(-12)
    # the p_gen 0 sequence is kept as the lowest value: median of [-12, -14, -inf] is -14
    assert summary_df.loc["modelA", "median_log10_p_gen_test"] == pytest.approx(-14)
    assert summary_df.loc["modelA", "median_log10_p_gen_gap"] == pytest.approx(2)
    assert summary_df.loc["modelB", "median_log10_p_gen_gap"] == pytest.approx(1)
    # train [-10, -12, -14] vs test [-12, -14, 0]: 6 wins and 2 ties out of 9 pairs
    assert summary_df.loc["modelA", "membership_auc"] == pytest.approx(7 / 9)


def test_summarize_p_gens_mostly_zero(tmp_path):
    p_gens_df = pd.DataFrame({"model": "m", "file": "rep_0", "junction_aa": "CASF",
                              "split": ["train", "train", "test", "test", "test"],
                              "p_gen": [1e-10, 1e-12, 0.0, 0.0, 1e-12]})

    summary_df = summarize_p_gens(p_gens_df)

    assert summary_df.loc[0, "median_log10_p_gen_test"] == -np.inf
    assert summary_df.loc[0, "median_log10_p_gen_gap"] == np.inf


@pytest.mark.parametrize("train, test, expected", [
    ([3, 4], [1, 2], 1.0),
    ([1, 2], [3, 4], 0.0),
    ([1, 2], [1, 2], 0.5),
    ([0.0, 0.0], [0.0, 1.0], 0.25),
])
def test_compute_auc(train, test, expected):
    assert compute_auc(pd.Series(train), pd.Series(test)) == pytest.approx(expected)


@pytest.mark.parametrize("n_panels, expected", [(1, (1, 1)), (3, (3, 1)), (5, (5, 1)), (6, (5, 2)), (15, (5, 3))])
def test_get_grid_shape(n_panels, expected):
    assert get_grid_shape(n_panels) == expected


def test_get_family_colors():
    family_colors = get_family_colors(["PWM", "TCRpeg_kmer", "TCRpeg_overlap", "Saka_kmer"])

    assert list(family_colors) == ["PWM", "TCRpeg", "Saka"]
    assert len(set(family_colors.values())) == 3


def test_plot_p_gen_distributions_grid(analysis_config, mocker):
    m_write = mocker.patch("plotly.graph_objects.Figure.write_image", autospec=True)
    model_names = [f"model{i}_kmer" for i in range(7)]
    p_gens_df = pd.DataFrame([{"model": model_name, "split": split, "file": "rep_0", "junction_aa": "CASF",
                               "p_gen": p_gen} for model_name in model_names for split in ["train", "test"]
                              for p_gen in [1e-10, 1e-12, 1e-14]])

    plot_p_gen_distributions(p_gens_df, model_names, analysis_config.analysis_output_dir, "TCR")

    fig = m_write.call_args.args[0]
    # 7 panels wrap into 2 rows of at most 5, with a train and a test curve per panel plus 2 legend traces
    assert fig.layout.height == 100 + 450 * 2
    assert len(fig.data) == 7 * 2 + 2
    # all models are of different families, so every panel has its own color
    assert len({trace.line.color for trace in fig.data if trace.showlegend is False}) == 7


def test_get_density_grid():
    grid = get_density_grid(np.array([-20.0, -10.0]), n_points=5)

    assert grid[0] == pytest.approx(-21) and grid[-1] == pytest.approx(-9)
    assert len(grid) == 5


def test_compute_density():
    values = np.array([-12.0, -11.0, -10.0, -9.0, -8.0])
    grid = np.linspace(-30, 10, 2001)

    density = compute_density(values, grid)

    assert np.trapz(density, grid) == pytest.approx(1, abs=1e-3)
    assert grid[np.argmax(density)] == pytest.approx(-10, abs=0.1)
    assert compute_density(np.array([-10.0, -10.0]), grid) is None


def test_run_p_gen_analysis(analysis_config, mocker):
    m_dist = mocker.patch("gen_airr_bm.analysis.analyse_p_gen.plot_p_gen_distributions")
    m_scores = mocker.patch("gen_airr_bm.analysis.analyse_p_gen.plot_per_repertoire_scores")

    run_p_gen_analysis(analysis_config)

    summary_df = pd.read_csv(f"{analysis_config.analysis_output_dir}/p_gen_summary.tsv", sep="\t")
    assert list(summary_df["model"]) == ["modelA", "modelB"]
    assert "membership_auc" in summary_df.columns
    m_dist.assert_called_once()
    assert [call.args[1] for call in m_scores.call_args_list] == ["median_log10_p_gen_gap", "membership_auc"]
