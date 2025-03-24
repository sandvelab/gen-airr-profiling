import pytest

from gen_airr_bm.analysis.distribution.aa_strategy import (
    compute_positional_aa_dist,
    compute_jsd_aa,
    AADistributionStrategy
)


def test_aa_distribution_strategy_compute_divergence(mocker):
    mock_compute_positional_aa_dist = mocker.patch("gen_airr_bm.analysis.distribution.aa_strategy.compute_positional_aa_dist")
    mock_compute_jsd_aa = mocker.patch("gen_airr_bm.analysis.distribution.aa_strategy.compute_jsd_aa")
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


def test_compute_positional_aa_dist_non_empty():
    sequences = ["ACD", "ACD", "ACD"]
    dist = compute_positional_aa_dist(sequences)

    assert isinstance(dist, dict)
    assert len(dist) == 3
    assert all(aa in dist[0] for aa in 'ACDEFGHIKLMNPQRSTVWY')
    assert dist[0]['A'] == 1.0
    assert dist[1]['C'] == 1.0
    assert dist[2]['D'] == 1.0


def test_compute_positional_aa_dist_empty():
    dist = compute_positional_aa_dist([])
    assert isinstance(dist, dict)
    assert len(dist) == 11  # positions 10 through 20
    for pos in range(10, 21):
        assert all(v == 0.0 for v in dist[pos].values())


def test_compute_jsd_aa_identical():
    dist1 = compute_positional_aa_dist(["AAA", "AAA"])
    dist2 = compute_positional_aa_dist(["AAA", "AAA"])
    jsd = compute_jsd_aa(dist1, dist2)
    assert jsd == pytest.approx(0.0)


def test_compute_jsd_aa_completely_different():
    dist1 = compute_positional_aa_dist(["AAA", "AAA"])
    dist2 = compute_positional_aa_dist(["CCC", "CCC"])
    jsd = compute_jsd_aa(dist1, dist2)
    assert jsd > 0.0


def test_compute_jsd_aa_both_empty():
    dist1 = compute_positional_aa_dist([])
    dist2 = compute_positional_aa_dist([])
    jsd = compute_jsd_aa(dist1, dist2)
    assert jsd == 0.0
