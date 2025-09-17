import pytest

from gen_airr_bm.analysis.distribution.aa_distribution_strategy import AADistributionStrategy
from gen_airr_bm.core.analysis_config import AnalysisConfig


# TO DO: Cover whole AA distribution strategy with tests
def test_aa_distribution_strategy_compute_divergence(mocker):
    mock_compute_positional_aa_dist = mocker.patch(
        "gen_airr_bm.analysis.distribution.aa_distribution_strategy.compute_positional_aa_dist")
    mock_compute_jsd_aa = mocker.patch("gen_airr_bm.analysis.distribution.aa_distribution_strategy.compute_jsd_aa")
    mock_compute_jsd_aa.return_value = 0.5

    strategy = AADistributionStrategy()
    gen = ["ACDEFGHIKL", "MNOPQRSTVW"]
    ref = ["ACDEFGHIKL"]

    result = strategy.compute_divergence(gen, ref)

    assert isinstance(result, dict)
    for length in range(10, 21):
        assert result[length] == [0.5]
    assert len(result) == len(range(10, 21))
    assert mock_compute_positional_aa_dist.call_count == 2 * len(range(10, 21))
    assert mock_compute_jsd_aa.call_count == len(range(10, 21))


def test_aa_distribution_strategy_init_mean_std_scores():
    strategy = AADistributionStrategy()
    mean, std = strategy.init_mean_std_scores()
    assert mean == {}
    assert std == {}


def test_aa_distribution_strategy_init_divergence_scores():
    strategy = AADistributionStrategy()
    scores = strategy.init_divergence_scores()
    assert scores == {}


def test_aa_distribution_strategy_update_divergence_scores():
    strategy = AADistributionStrategy()
    scores = {10: [0.1, 0.2]}
    new = {10: [0.3, 0.4]}
    strategy.update_divergence_scores(scores, new)
    assert scores == {10: [0.1, 0.2, 0.3, 0.4]}


def test_aa_distribution_strategy_update_mean_std_scores():
    strategy = AADistributionStrategy()
    mean_scores, std_scores = strategy.init_mean_std_scores()
    dummy_length = 10
    divergence_scores = {dummy_length: [1.0, 1.0, 1.0]}

    strategy.update_mean_std_scores(divergence_scores, "test_model", mean_scores, std_scores)

    assert dummy_length in mean_scores
    assert dummy_length in std_scores
    assert mean_scores[dummy_length]["test_model"] == pytest.approx(1.0)
    assert std_scores[dummy_length]["test_model"] == pytest.approx(0.0)


def test_aa_distribution_strategy_plot_scores(mocker):
    mock_plot_jsd_scores = mocker.patch("gen_airr_bm.analysis.distribution.aa_distribution_strategy.plot_grouped_avg_scores")

    strategy = AADistributionStrategy()
    mean_scores = {"ref": {length: {"model": 0.5} for length in range(10, 21)}}
    std_scores = {"ref": {length: {"model": 0.1} for length in range(10, 21)}}

    config = AnalysisConfig(
        model_names=["model"],
        reference_data=["ref"],
        root_output_dir="output",
        analysis_output_dir="output/analysis",
        analysis="test",
        default_model_name="model"
    )

    strategy.plot_scores_by_reference(mean_scores, std_scores, config, "dummy", [0.5]*len(range(10, 21)))

    assert mock_plot_jsd_scores.call_count == len(range(10, 21))
