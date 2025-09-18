from collections import Counter

import pytest

from gen_airr_bm.analysis.distribution.length_distribution_strategy import (
    LengthDistributionStrategy,
    compute_jsd_length
)


def test_length_distribution_strategy_divergence(mocker):
    mock_counter = mocker.patch("gen_airr_bm.analysis.distribution.length_distribution_strategy.Counter")
    mock_compute_jsd_length = mocker.patch("gen_airr_bm.analysis.distribution.length_distribution_strategy.compute_jsd_length")
    strategy = LengthDistributionStrategy()

    gen = ["AAAAA", "BBBB", "CCC"]     # lengths: 5, 4, 3
    ref = ["DDD", "EEE", "FFFFF"]      # lengths: 3, 3, 5

    strategy.compute_divergence(gen, ref)

    assert mock_counter.call_count == 2
    mock_counter.assert_any_call([5, 4, 3])
    mock_counter.assert_any_call([3, 3, 5])
    mock_compute_jsd_length.assert_called_once()


def test_compute_jsd_length_identical_distributions():
    dist = Counter({10: 3, 12: 2})
    jsd = compute_jsd_length(dist, dist)
    assert jsd == pytest.approx(0.0)


def test_compute_jsd_length_different_distributions():
    dist1 = Counter({10: 5})
    dist2 = Counter({12: 5})
    jsd = compute_jsd_length(dist1, dist2)
    assert jsd > 0.9  # Should be close to max divergence


def test_compute_jsd_length_partial_overlap():
    dist1 = Counter({10: 2, 12: 3})
    dist2 = Counter({12: 5})
    jsd = compute_jsd_length(dist1, dist2)
    assert 0.0 < jsd < 1.0
