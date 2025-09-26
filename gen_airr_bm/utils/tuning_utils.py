import os

from gen_airr_bm.core.tuning_config import TuningConfig


def validate_analyses_data(tuning_config: TuningConfig, required_analyses: list) -> list:
    """ Validates that the necessary analyses for sequence feature tuning have been run.
    Args:
        tuning_config: Configuration for the tuning, including paths and model names.
        required_analyses (list): List of required analyses to validate.
    Returns:
        list: List of validated analysis paths.
    """
    model_names = tuning_config.model_names
    analyses_dir = os.path.join(tuning_config.root_output_dir, "analyses")

    validated_analyses_paths = []
    for analysis in required_analyses:
        analysis_path = os.path.join(analyses_dir, analysis, '_'.join(model_names))
        if not os.path.exists(analysis_path):
            raise FileNotFoundError(f"Required analysis '{analysis}' for models {model_names} not found in "
                                    f"{analyses_dir}. Please run this analysis before tuning.")
        else:
            validated_analyses_paths.append(analysis_path)

    return validated_analyses_paths
