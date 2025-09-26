import os

from gen_airr_bm.core.tuning_config import TuningConfig
from gen_airr_bm.utils.tuning_utils import validate_analyses_data


def run_sequence_features_tuning(tuning_config: TuningConfig):
    """ Runs parameter tuning by sequence feature recovery.
        Args:
            tuning_config: Configuration for the tuning, including paths and model names.
        Returns:
            None
    """
    validated_analyses_paths = validate_analyses_data(tuning_config, required_analyses=['reduced_dimensionality'])
    print(f"Validated analyses for tuning: {validated_analyses_paths}")

    os.makedirs(tuning_config.tuning_output_dir, exist_ok=True)


