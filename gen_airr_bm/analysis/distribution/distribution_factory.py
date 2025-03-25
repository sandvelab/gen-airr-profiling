from gen_airr_bm.analysis.distribution.aa_distribution_strategy import AADistributionStrategy
from gen_airr_bm.analysis.distribution.base_distribution_strategy import BaseDistributionStrategy
from gen_airr_bm.analysis.distribution.kmer_distribution_strategy import KmerDistributionStrategy
from gen_airr_bm.analysis.distribution.length_distribution_strategy import LengthDistributionStrategy
from gen_airr_bm.constants.distribution_type import DistributionType


def get_distribution_strategy(distribution_type: DistributionType, k: int = 3) -> BaseDistributionStrategy:
    if distribution_type == DistributionType.AA:
        return AADistributionStrategy()
    elif distribution_type == DistributionType.KMER:
        return KmerDistributionStrategy(k)
    elif distribution_type == DistributionType.LENGTH:
        return LengthDistributionStrategy()
    else:
        raise ValueError("Unsupported distribution type")
