class ModelConfig:
    """Handles settings for a single model (e.g., sonnia, pwm)."""

    def __init__(self, name: str, config: str, experiment: int, train_dir: str, test_dir: str, output_dir: str,
                 n_subset_samples: int):
        self.name = name
        self.config = config
        self.experiment = experiment
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.n_subset_samples = n_subset_samples

    def __repr__(self):
        return (f"ModelConfig(name={self.name}, config={self.config}, experiment={self.experiment}, "
                f"n_subset_samples={self.n_subset_samples}, train_dir={self.train_dir}, test_dir={self.test_dir}, "
                f"output_dir={self.output_dir})")
