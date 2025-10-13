class TuningConfig:
    def __init__(self, tuning_method: str, model_names: list, reference_data: list, tuning_output_dir: str,
                 root_output_dir: str, k_values: list):
        self.tuning_method = tuning_method
        self.model_names = model_names
        self.reference_data = reference_data
        self.tuning_output_dir = tuning_output_dir
        self.root_output_dir = root_output_dir
        self.k_values = k_values

    def __repr__(self):
        return (f"TuningConfig(tuning_method={self.tuning_method}, model_names={self.model_names}, "
                f"reference_data={self.reference_data}, tuning_output_dir={self.tuning_output_dir}, "
                f"root_output_dir={self.root_output_dir}, k_values={self.k_values})")
