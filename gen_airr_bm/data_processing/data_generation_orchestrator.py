from gen_airr_bm.core.data_generation_config import DataGenerationConfig
from gen_airr_bm.data_processing.data_generation_methods import (simulate_rare_and_frequent_olga_sequences,
                                                                 preprocess_experimental_data)


class DataGenerationOrchestrator:
    """Orchestrates which simulation method to run based on the config."""
    DATA_GENERATION_METHODS = {
        "rare_and_frequent": simulate_rare_and_frequent_olga_sequences,
        "experimental": preprocess_experimental_data
    }

    def run_data_generation(self, config: DataGenerationConfig):
        """Runs the appropriate simulation method based on config."""
        method = config.method
        n_samples = config.n_samples

        if method not in self.DATA_GENERATION_METHODS:
            raise ValueError(f"Unknown simulation method: {method}")

        print(f"Running simulation: {method} with {n_samples} samples.")
        return self.DATA_GENERATION_METHODS[method](config)
