import pytest

from gen_airr_bm.analysis.distribution.aa_distribution_strategy import AADistributionStrategy


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


def test_aa_distribution_strategy_get_metric_scores():
    strategy = AADistributionStrategy()
    divergence = {length: [length / 100] for length in range(10, 21)}

    metric_scores = strategy.get_metric_scores(divergence)

    assert metric_scores == {length: pytest.approx(length / 100) for length in range(10, 21)}


def test_aa_distribution_strategy_get_metric_label_and_file_stem():
    strategy = AADistributionStrategy()
    assert strategy.get_metric_label("amino acid", 12) == "amino acid (Seq. Len.: 12)"
    assert strategy.get_file_stem("amino acid", 12) == "aminoacid_12"
