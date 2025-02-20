class AnalysisConfig:
    def __init__(self, analysis, model_name, output_dir):
        self.analysis = analysis
        self.model_name = model_name
        self.output_dir = output_dir

    def __repr__(self):
        return f"AnalysisConfig(analysis={self.analysis}, model_name={self.model_name}, output_dir={self.output_dir})"
