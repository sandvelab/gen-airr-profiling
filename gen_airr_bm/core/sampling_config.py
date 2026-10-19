class SamplingConfig:
    def __init__(self, model_name: str, experiment_name: str, immuneml_config: str,
                 n_samples: int, root_output_dir: str, compute_p_gens: bool = False):
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.immuneml_config = immuneml_config
        self.n_samples = n_samples
        self.root_output_dir = root_output_dir
        self.compute_p_gens = compute_p_gens

    @property
    def p_gens_only(self) -> bool:
        """With n_samples 0, no sequences are generated and only the p_gens of the train and test sequences are
        computed, e.g. for models whose sequences were already sampled and should be kept."""
        return self.n_samples == 0 and self.compute_p_gens

    def __repr__(self):
        return (f"SamplingConfig(name={self.model_name}, experiment_name={self.experiment_name}, "
                f"immuneml_config={self.immuneml_config}, n_samples={self.n_samples}, "
                f"root_output_dir={self.root_output_dir}, compute_p_gens={self.compute_p_gens})")
