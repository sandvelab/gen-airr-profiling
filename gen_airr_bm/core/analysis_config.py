class AnalysisConfig:
    def __init__(self, analysis: str, model_names: list, analysis_output_dir: str, root_output_dir: str,
                 default_model_name: str, reference_data: str, n_subsets: int):
        self.analysis = analysis
        self.model_names = model_names
        self.analysis_output_dir = analysis_output_dir
        self.root_output_dir = root_output_dir
        self.default_model_name = default_model_name
        self.reference_data = reference_data
        self.n_subsets = n_subsets


    def __repr__(self):
        return (f"AnalysisConfig(analysis={self.analysis}, model_names={self.model_names}, "
                f"analysis_output_dir={self.analysis_output_dir}, root_output_dir={self.root_output_dir}, "
                f"default_model_name={self.default_model_name}, reference_data={self.reference_data}, "
                f"n_subsets={self.n_subsets})")
