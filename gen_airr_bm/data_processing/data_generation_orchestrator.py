from gen_airr_bm.core.data_generation_config import DataGenerationConfig
from gen_airr_bm.data_processing.data_generation_methods import simulate_rare_and_frequent_olga_sequences, \
    simulate_experimental_and_olga_sequences, preprocess_experimental_data


class SimulationOrchestrator:
    """Orchestrates which simulation method to run based on the config."""
    SIMULATION_METHODS = {
        "rare_and_frequent": simulate_rare_and_frequent_olga_sequences,
        "experimental": preprocess_experimental_data
    }

    def run_simulation(self, config: DataGenerationConfig):
        """Runs the appropriate simulation method based on config."""
        method = config.method
        n_samples = config.n_samples
        output_dir = config.output_dir
        model = config.model
        seed = config.seed
        data_file = config.data_file
        input_columns = config.input_columns

        if method not in self.SIMULATION_METHODS:
            raise ValueError(f"Unknown simulation method: {method}")

        print(f"Running simulation: {method} with {n_samples} samples.")
        return self.SIMULATION_METHODS[method](n_samples, model, seed, output_dir, data_file, input_columns)
