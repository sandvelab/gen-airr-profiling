from gen_airr_bm.core.tuning_config import TuningConfig
from gen_airr_bm.tuning.tuning_sequence_features import run_sequence_features_tuning


class TuningOrchestrator:
    """Orchestrates which tuning method to run based on the config."""
    TUNING_METHODS = {
        "sequence_features": run_sequence_features_tuning,
    }

    def run_tuning(self, tuning_config: TuningConfig):
        """Runs the appropriate analysis based on config."""
        tuning_method = tuning_config.tuning_method
        if tuning_method not in self.TUNING_METHODS:
            raise ValueError(f"Unknown tuning method: {tuning_method}")

        print(f"Running parameter tuning using method: {tuning_method}")
        return self.TUNING_METHODS[tuning_method](tuning_config)
