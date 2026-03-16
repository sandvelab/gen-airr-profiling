class SamplingConfig:
    def __init__(self, model_name: str, experiment_name: str, immuneml_config: str, train_dir: str,
                 n_samples: int, root_output_dir: str):
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.immuneml_config = immuneml_config
        self.train_dir = train_dir
        self.n_samples = n_samples
        self.root_output_dir = root_output_dir

    def __repr__(self):
        return (f"SamplingConfig(name={self.model_name}, experiment_name={self.experiment_name}, "
                f"immuneml_config={self.immuneml_config}, n_samples={self.n_samples}, "
                f"train_dir={self.train_dir}, root_output_dir={self.root_output_dir})")
