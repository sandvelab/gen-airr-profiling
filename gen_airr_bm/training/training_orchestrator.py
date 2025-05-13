import os

import pandas as pd

from gen_airr_bm.core.model_config import ModelConfig
from gen_airr_bm.training.immuneml_runner import run_immuneml_command, write_immuneml_config
from gen_airr_bm.utils.compairr_utils import preprocess_files_for_compairr


def divide_generated_sequences(generated_sequences_dir: str, generated_sequences_filename: str, output_dir: str,
                               n_samples: int, dedup_fields: list = None):
    os.makedirs(output_dir, exist_ok=True)
    generated_sequences = pd.read_csv(f"{generated_sequences_dir}/{generated_sequences_filename}.tsv", sep='\t')
    n_datasets = len(generated_sequences) // n_samples
    for i in range(n_datasets):
        start_idx = i * n_samples
        end_idx = (i + 1) * n_samples
        split_dataset = generated_sequences.iloc[start_idx:end_idx]
        split_dataset.drop_duplicates(subset=dedup_fields)
        if len(split_dataset) < n_samples * 0.9:
            raise ValueError(
                f"Too many duplicates removed: only {len(split_dataset)} rows remaining out of {len(generated_sequences)}."
            )
        else:
            split_dataset.to_csv(f"{output_dir}/{generated_sequences_filename}_{i}.tsv", sep='\t', index=False)


class TrainingOrchestrator:
    """Orchestrates the immuneML training process."""

    def run_single_training(self, immuneml_config: str, data_path: str, output_dir: str):
        """Runs immuneML training for model in the config."""
        output_immuneml_config = f"{output_dir}/immuneml_config.yaml"
        output_immuneml_dir = f"{output_dir}/immuneml"

        write_immuneml_config(immuneml_config, data_path, output_immuneml_config)
        run_immuneml_command(output_immuneml_config, output_immuneml_dir)

    # TODO: It became spaghetti code. We might want to refactor this method.
    def run_training(self, model_config: ModelConfig, output_dir: str):
        """Runs immuneML training for model in the config with phenotype data."""
        train_data_dir = os.path.join(model_config.output_dir, model_config.train_dir)
        train_data_files = [f for f in os.listdir(train_data_dir) if
                            os.path.isfile(os.path.join(train_data_dir, f))]
        for train_data_file in train_data_files:
            data_file_name = train_data_file.split('.')[0]
            model_output_dir = f"{model_config.output_dir}/{model_config.name}/{data_file_name}"
            train_data_full_path = f"{train_data_dir}/{train_data_file}"
            os.makedirs(f"{output_dir}/train_sequences", exist_ok=True)
            os.system(
                f"cp -n {train_data_full_path} {output_dir}/train_sequences/{data_file_name}_{model_config.experiment}.tsv")
            compairr_train_dir = f"{output_dir}/train_compairr_sequences"
            preprocess_files_for_compairr(f"{output_dir}/train_sequences", compairr_train_dir)

            # TODO: this is a quick dirty solution. We need to refactor this part.
            if model_config.test_dir:
                os.makedirs(f"{output_dir}/test_sequences/", exist_ok=True)
                # Copy test data to the output directory
                test_data_dir = os.path.join(model_config.output_dir, model_config.test_dir)
                test_data_files = [f for f in os.listdir(test_data_dir) if
                                   os.path.isfile(os.path.join(test_data_dir, f))]
                for test_data_file in test_data_files:
                    os.system(
                        f"cp -n {test_data_dir}/{test_data_file} {output_dir}/test_sequences/{test_data_file.split('.')[0]}_{model_config.experiment}.tsv")
                    compairr_test_dir = f"{output_dir}/test_compairr_sequences"
                    preprocess_files_for_compairr(f"{output_dir}/test_sequences", compairr_test_dir)

            os.makedirs(model_output_dir, exist_ok=True)
            self.run_single_training(model_config.config, train_data_full_path, model_output_dir)

            # Copy generated sequences to the output directory
            # The generated sequences are stored in a subdirectory of the immuneML output directory (always static path)
            immuneml_generated_sequences_dir = f"{model_output_dir}/immuneml/gen_model/generated_sequences"
            # Generated sequences files might have different names, so we need to find the correct one
            immuneml_generated_sequences_file = [
                f for f in os.listdir(immuneml_generated_sequences_dir)
                if f.endswith(".tsv") and os.path.isfile(os.path.join(immuneml_generated_sequences_dir, f))
            ][0]
            generated_sequences_dir = f"{output_dir}/generated_sequences/{model_config.name}"
            os.makedirs(generated_sequences_dir, exist_ok=True)
            os.system(f"cp -n {immuneml_generated_sequences_dir}/{immuneml_generated_sequences_file} "
                      f"{generated_sequences_dir}/{data_file_name}_{model_config.experiment}.tsv")

            # process files for CompAIRR
            compairr_model_dir = f"{output_dir}/generated_compairr_sequences/{model_config.name}"
            preprocess_files_for_compairr(generated_sequences_dir, compairr_model_dir)

            divide_generated_sequences(compairr_model_dir,
                                       f"{data_file_name}_{model_config.experiment}",
                                       f"{output_dir}/generated_compairr_sequences_split/{model_config.name}",
                                       model_config.n_subset_samples, ['junction_aa', 'v_call', 'j_call'])
