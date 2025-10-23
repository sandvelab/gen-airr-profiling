import os
from pathlib import Path

import numpy as np

from gen_airr_bm.core.tuning_config import TuningConfig


def format_value(x):
    """Return int if looks like int, otherwise rounded float."""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        # show 3 decimals max, drop trailing zeros
        return f"{x:.3f}".rstrip("0").rstrip(".")
    return str(x)


def validate_analyses_data(tuning_config: TuningConfig, required_analyses: list) -> list:
    """ Validates that the necessary analyses for the tuning method have been run.
    Args:
        tuning_config: Configuration for the tuning, including paths and model names.
        required_analyses (list): List of required analyses to validate.
    Returns:
        list: List of validated analysis paths.
    """
    subfolder_name = tuning_config.subfolder_name
    model_names = tuning_config.model_names
    analyses_dir = Path(tuning_config.root_output_dir) / "analyses"

    validated_analyses_paths = []
    for analysis in required_analyses:
        analysis_path = analyses_dir / analysis / '_'.join(subfolder_name.split())
        if not os.path.exists(analysis_path):
            raise FileNotFoundError(f"Required analysis '{analysis}' with models {model_names} not found in "
                                    f"{analysis_path}. Please run this analysis before tuning using method "
                                    f"'{tuning_config.tuning_method}'.")
        else:
            validated_analyses_paths.append(analysis_path)

    return validated_analyses_paths
