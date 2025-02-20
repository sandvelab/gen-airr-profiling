import os

from gen_airr_bm.core.model_config import ModelConfig
from gen_airr_bm.scripts.immuneml_formatting import write_immuneml_config
from gen_airr_bm.training.immuneml_runner import run_immuneml_app


class TrainingOrchestrator:
    """Orchestrates the immuneML training process."""

    def run_training(self, immuneml_config: str, data_path: str, output_dir: str):
        """Runs immuneML training for model in the config."""
        output_immuneml_config = f"{output_dir}/immuneML_config.yaml"

        write_immuneml_config(immuneml_config, data_path, output_immuneml_config)
        #run_immuneml_app(output_immuneml_config, output_dir)

    def run_phenotypes_training(self, model_config: ModelConfig):
        """Runs immuneML training for model in the config with phenotype data."""
        phenotype_files = [f for f in os.listdir(model_config.output_dir) if
                           os.path.isfile(os.path.join(model_config.output_dir, f))]
        for phenotype_file in phenotype_files:
            phenotype_output_dir = f"{model_config.output_dir}/{model_config.name}/{phenotype_file.split('.')[0]}"
            phenotype_full_path = f"{model_config.output_dir}/{phenotype_file}"
            os.makedirs(phenotype_output_dir, exist_ok=True)

            self.run_training(model_config.config, phenotype_full_path, phenotype_output_dir)
