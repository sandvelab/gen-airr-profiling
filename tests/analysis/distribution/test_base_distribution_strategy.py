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


def test_get_metric_scores():
    strategy = DummyStrategyBase()
    assert strategy.get_metric_scores([0.5]) == {"all": 0.5}


def test_get_metric_label_and_file_stem():
    strategy = DummyStrategyBase()
    assert strategy.get_metric_label("amino acid", "all") == "amino acid"
    assert strategy.get_file_stem("amino acid", "all") == "aminoacid"


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
        default_model_name="model",
        subfolder_name="analysis_subfolder",
        receptor_type="TCR"
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
