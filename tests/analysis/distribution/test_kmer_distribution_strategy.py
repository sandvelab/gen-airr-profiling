import pytest

from gen_airr_bm.analysis.distribution.kmer_distribution_strategy import (
    KmerDistributionStrategy,
    compute_kmer_distribution,
    compute_jsd_kmers
)


def test_kmer_distribution_strategy_divergence(mocker):
    mock_compute_kmer_distribution = mocker.patch(
        "gen_airr_bm.analysis.distribution.kmer_distribution_strategy.compute_kmer_distribution")
    mock_compute_jsd_kmers = mocker.patch("gen_airr_bm.analysis.distribution.kmer_distribution_strategy.compute_jsd_kmers")

    strategy = KmerDistributionStrategy(k=3)
    gen = ["ABCDEF", "GHIJKL"]
    ref = ["MNOPQR", "STUVWX"]

    strategy.compute_divergence(gen, ref)

    assert mock_compute_kmer_distribution.call_count == 2
    mock_compute_kmer_distribution.assert_any_call(gen, 3)
    mock_compute_kmer_distribution.assert_any_call(ref, 3)
    mock_compute_jsd_kmers.assert_called_once()


def test_compute_kmer_distribution_basic():
    seqs = ["ABCD", "BCD"]
    k = 3
    dist = compute_kmer_distribution(seqs, k)

    expected_kmers_dist = {
        "ABC": 1.0,
        "BCD": 2.0,
    }

    for kmer, number in expected_kmers_dist.items():
        assert kmer in dist
        assert dist[kmer] == pytest.approx(number / 3)  # Total 3 kmers: ABC, BCD (twice)

    assert len(dist) == 2
    assert pytest.approx(sum(dist.values())) == 1.0


def test_compute_kmer_distribution_empty_input():
    dist = compute_kmer_distribution([], 3)
    assert dist == {}


def test_compute_jsd_kmers_identical_distributions():
    dist = {"AAA": 0.5, "BBB": 0.5}
    jsd = compute_jsd_kmers(dist, dist)
    assert jsd == pytest.approx(0.0)


def test_compute_jsd_kmers_different_distributions():
    dist1 = {"AAA": 1.0}
    dist2 = {"BBB": 1.0}
    jsd = compute_jsd_kmers(dist1, dist2)
    assert jsd > 0.9  # Should be close to max divergence
