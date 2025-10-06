from gen_airr_bm.core.tuning_config import TuningConfig
from gen_airr_bm.tuning.tuning_reduced_dim import run_reduced_dim_tuning
from gen_airr_bm.tuning.tuning_overlap import run_overlap_tuning


class TuningOrchestrator:
    """Orchestrates which tuning method to run based on the config."""
    TUNING_METHODS = {
        "reduced_dimensionality": run_reduced_dim_tuning,
        "overlap": run_overlap_tuning
    }

    def run_tuning(self, tuning_config: TuningConfig):
        """Runs the appropriate analysis based on config."""
        tuning_method = tuning_config.tuning_method
        if tuning_method not in self.TUNING_METHODS:
            raise ValueError(f"Unknown tuning method: {tuning_method}")

        print(f"Running parameter tuning using method: {tuning_method}")
        return self.TUNING_METHODS[tuning_method](tuning_config)
