class DataGenerationConfig:
    """Handles simulation-related settings."""

    def __init__(self, method, n_samples, experimental, data_file, default_model_name, experiment, seed, output_dir, input_columns):
        self.method = method
        self.n_samples = n_samples
        self.experimental = experimental
        self.data_file = data_file
        self.output_dir = output_dir
        self.default_model_name = default_model_name
        self.experiment = experiment
        self.seed = seed
        self.input_columns = input_columns

    def __repr__(self):
        return (f"DataGenerationConfig(method={self.method}, n_samples={self.n_samples}, data_file={self.data_file}, "
                f"experimental={self.experimental}, model={self.default_model_name}, experiment={self.experiment}, "
                f"seed={self.seed}, output_dir={self.output_dir}), input_columns={self.input_columns}")
