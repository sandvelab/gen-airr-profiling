class ModelConfig:
    """Handles settings for a single model (e.g., sonnia, pwm)."""

    def __init__(self, name: str, config: str, experiment: int, train_dir: str, output_dir: str):
        self.name = name
        self.config = config
        self.experiment = experiment
        self.train_dir = train_dir
        self.output_dir = output_dir

    def __repr__(self):
        return (f"ModelConfig(name={self.name}, config={self.config}, experiment={self.experiment},"
                f" output_dir={self.output_dir})")
