import numpy as np
import pytest

from gen_airr_bm.analysis.distribution.base_distribution_strategy import BaseDistributionStrategy
from gen_airr_bm.core.analysis_config import AnalysisConfig


# Dummy subclass for testing concrete methods
class DummyStrategyBase(BaseDistributionStrategy):
    def compute_divergence(self, gen_seqs, ref_seqs):
        return ["dummy"]


def test_cannot_instantiate_abstract_class():
    with pytest.raises(TypeError):
        BaseDistributionStrategy()


def test_init_mean_std_scores():
    strategy = DummyStrategyBase()
    mean, std = strategy.init_mean_std_scores()
    assert mean == {}
    assert std == {}


def test_init_divergence_scores():
    strategy = DummyStrategyBase()
    scores = strategy.init_divergence_scores()
    assert scores == []


def test_update_divergence_scores():
    strategy = DummyStrategyBase()
    scores = [0.1, 0.2]
    new = [0.3, 0.4]
    strategy.update_divergence_scores(scores, new)
    assert scores == [0.1, 0.2, 0.3, 0.4]


def test_update_mean_std_scores():
    strategy = DummyStrategyBase()
    mean_scores, std_scores = {}, {}
    divergence_scores = [0.5, 1.5, 2.5]

    strategy.update_mean_std_scores(divergence_scores, "test_model", mean_scores, std_scores)

    assert "test_model" in mean_scores
    assert "test_model" in std_scores
    assert mean_scores["test_model"] == pytest.approx(np.mean(divergence_scores))
    assert std_scores["test_model"] == pytest.approx(np.std(divergence_scores))


def test_plot_scores_calls_plot_jsd_scores(mocker):
    mock_plot_jsd_scores = mocker.patch("gen_airr_bm.analysis.distribution.base_distribution_strategy.plot_avg_scores")

    strategy = DummyStrategyBase()
    mean_scores = {"model": 0.5}
    std_scores = {"model": 0.1}

    config = AnalysisConfig(
        model_names=["model"],
        reference_data="ref",
        root_output_dir="output",
        analysis_output_dir="output/analysis",
        analysis="test",
        default_model_name="model"
    )

    strategy.plot_scores(mean_scores, std_scores, config, "dummy")

    mock_plot_jsd_scores.assert_called_once()
    args = mock_plot_jsd_scores.call_args[0]

    assert args[0] == mean_scores
    assert args[1] == std_scores
    assert args[2] == "output/analysis"
    assert args[3] == "ref"
    assert args[4] == "dummy"
    assert args[5] == "dummy"
