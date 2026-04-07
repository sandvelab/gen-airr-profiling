class PostProcessingConfig:
    def __init__(self, model_name: str, experiment_name: str, n_samples: int, root_output_dir: str, n_subsets: int):
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.n_samples = n_samples
        self.root_output_dir = root_output_dir
        self.n_subsets = n_subsets

    def __repr__(self):
        return (f"SamplingConfig(name={self.model_name}, experiment_name={self.experiment_name}, "
                f"n_samples={self.n_samples}, root_output_dir={self.root_output_dir}, n_subsets={self.n_subsets})")
