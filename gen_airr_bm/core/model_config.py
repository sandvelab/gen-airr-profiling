class ModelConfig:
    """Handles settings for a single model (e.g., sonnia, pwm)."""

    def __init__(self, name: str, immuneml_model_config: str, experiment: int, train_dir: str, test_dir: str,
                 output_dir: str, n_subset_samples: int):
        self.name = name
        self.immuneml_model_config = immuneml_model_config
        self.experiment = experiment
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.n_subset_samples = n_subset_samples
        self.locus = None

    def __repr__(self):
        return (f"ModelConfig(name={self.name}, immuneml_model_config={self.immuneml_model_config}, "
                f"experiment={self.experiment}, n_subset_samples={self.n_subset_samples}, train_dir={self.train_dir}, "
                f"test_dir={self.test_dir}, output_dir={self.output_dir})")
