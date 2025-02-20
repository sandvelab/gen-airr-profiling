import os

from gen_airr_bm.core.model_config import ModelConfig
from gen_airr_bm.scripts.immuneml_formatting import write_immuneml_config
from gen_airr_bm.training.immuneml_runner import run_immuneml_command


class TrainingOrchestrator:
    """Orchestrates the immuneML training process."""

    def run_training(self, immuneml_config: str, data_path: str, output_dir: str):
        """Runs immuneML training for model in the config."""
        output_immuneml_config = f"{output_dir}/immuneml_config.yaml"
        output_immuneml_dir = f"{output_dir}/immuneml"

        write_immuneml_config(immuneml_config, data_path, output_immuneml_config)
        run_immuneml_command(output_immuneml_config, output_immuneml_dir)

    def run_phenotypes_training(self, model_config: ModelConfig, output_dir: str):
        """Runs immuneML training for model in the config with phenotype data."""
        phenotype_files = [f for f in os.listdir(model_config.output_dir) if
                           os.path.isfile(os.path.join(model_config.output_dir, f))]
        for phenotype_file in phenotype_files:
            phenotype = phenotype_file.split('.')[0]
            phenotype_output_dir = f"{model_config.output_dir}/{model_config.name}/{phenotype}"
            phenotype_full_path = f"{model_config.output_dir}/{phenotype_file}"
            os.makedirs(phenotype_output_dir, exist_ok=True)

            self.run_training(model_config.config, phenotype_full_path, phenotype_output_dir)

            immuneml_generated_sequences_dir = f"{phenotype_output_dir}/immuneml/gen_model/generated_sequences"
            #Generated sequences files might have different names, so we need to find the correct one
            immuneml_generated_sequences_file = [
                f for f in os.listdir(immuneml_generated_sequences_dir)
                if f.endswith(".tsv") and os.path.isfile(os.path.join(immuneml_generated_sequences_dir, f))
            ][0]
            generated_sequences_dir = f"{output_dir}/generated_sequences/{model_config.name}"
            os.makedirs(generated_sequences_dir, exist_ok=True)
            os.system(f"cp {immuneml_generated_sequences_dir}/{immuneml_generated_sequences_file} "
                      f"{generated_sequences_dir}/{phenotype}_{model_config.experiment}.tsv")
