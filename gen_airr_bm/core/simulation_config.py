class SimulationConfig:
    """Handles simulation-related settings."""

    def __init__(self, method, n_samples, model, experiment, seed, output_dir):
        self.method = method
        self.n_samples = n_samples
        self.output_dir = output_dir
        self.model = model
        self.experiment = experiment
        self.seed = seed

    def __repr__(self):
        return (f"SimulationConfig(method={self.method}, n_samples={self.n_samples}, model={self.model}, "
                f"experiment={self.experiment}, seed={self.seed}, output_dir={self.output_dir})")
