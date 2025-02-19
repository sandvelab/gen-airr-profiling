from gen_airr_bm.core.model_config import ModelConfig
from gen_airr_bm.scripts.immuneml_formatting import write_immuneml_config
from gen_airr_bm.training.immuneml_runner import run_immuneml_app


class TrainingOrchestrator:
    """Orchestrates the immuneML training process."""

    def run_training(self, config: ModelConfig, data_path: str):
        """Runs immuneML training for all models in the config."""
        print(f"\nRunning training for model: {config.name}")

        input_model_template = config.config
        input_simulated_data = data_path
        output_config_file = f"{config.output_dir}/immuneML_config.yaml"
        output_dir = config.output_dir

        write_immuneml_config(input_model_template, input_simulated_data, output_config_file)

        run_immuneml_app(output_config_file, output_dir)
