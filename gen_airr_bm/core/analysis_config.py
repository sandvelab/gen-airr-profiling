class AnalysisConfig:
    def __init__(self, analysis, model_name, analysis_output_dir, root_output_dir):
        self.analysis = analysis
        self.model_name = model_name
        self.analysis_output_dir = analysis_output_dir
        self.root_output_dir = root_output_dir

    def __repr__(self):
        return (f"AnalysisConfig(analysis={self.analysis}, model_name={self.model_name}, "
                f"analysis_output_dir={self.analysis_output_dir}, root_output_dir={self.root_output_dir})")
