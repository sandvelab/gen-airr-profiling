import pytest

from gen_airr_bm.analysis.distribution.aa_strategy import AADistributionStrategy
from gen_airr_bm.analysis.distribution.distribution_factory import get_distribution_strategy
from gen_airr_bm.analysis.distribution.kmer_strategy import KmerDistributionStrategy
from gen_airr_bm.analysis.distribution.length_strategy import LengthDistributionStrategy
from gen_airr_bm.constants.distribution_type import DistributionType


@pytest.mark.parametrize(
    "distribution_type,k,expected_class,expected_k",
    [
        (DistributionType.AA, 3, AADistributionStrategy, None),
        (DistributionType.KMER, 3, KmerDistributionStrategy, 3),
        (DistributionType.KMER, 5, KmerDistributionStrategy, 5),
        (DistributionType.LENGTH, 3, LengthDistributionStrategy, None),
    ]
)
def test_get_distribution_strategy_valid(distribution_type, k, expected_class, expected_k):
    strategy = get_distribution_strategy(distribution_type, k)
    assert isinstance(strategy, expected_class)
    if expected_k is not None:
        assert getattr(strategy, "k", None) == expected_k


def test_get_distribution_strategy_invalid_type():
    with pytest.raises(ValueError, match="Unsupported distribution type"):
        get_distribution_strategy("INVALID_TYPE")
