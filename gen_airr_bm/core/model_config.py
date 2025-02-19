class ModelConfig:
    """Handles settings for a single model (e.g., sonnia, pwm)."""

    def __init__(self, name, config, output_dir):
        self.name = name
        self.config = config
        self.output_dir = output_dir

    def __repr__(self):
        return f"ModelConfig(name={self.name}, config={self.config}, output_dir={self.output_dir})"
