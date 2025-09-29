from pathlib import Path
import os

import pandas as pd

from gen_airr_bm.core.model_config import ModelConfig
from gen_airr_bm.training.immuneml_runner import run_immuneml_command, write_immuneml_config
from gen_airr_bm.utils.compairr_utils import preprocess_files_for_compairr


class TrainingOrchestrator:
    """Orchestrates the immuneML training process."""

    @staticmethod
    def run_single_training(immuneml_config_path: str, train_data_path: str, immuneml_output_dir: str, locus: str) \
            -> None:
        """Runs immuneML training for model in the config.
        Args:
            immuneml_config_path (str): Path to the immuneML configuration file.
            train_data_path (str): Path to the training data file.
            immuneml_output_dir (str): Directory to save the immuneML output.
            locus (str): Locus to be used for training (e.g., TRA, TRB, IGH).
        """
        output_immuneml_config = Path(immuneml_output_dir) / "immuneml_config.yaml"
        output_immuneml_dir = Path(immuneml_output_dir) / "immuneml"

        write_immuneml_config(immuneml_config_path, train_data_path, output_immuneml_config, locus)
        run_immuneml_command(output_immuneml_config, output_immuneml_dir)

    @staticmethod
    def get_default_locus_name(train_data_path: str) -> str:
        """Returns the default locus name from the training data.
        Args:
            train_data_path (str): Path to the training data file.
        Returns:
            str: Default locus name.
        """
        train_df = pd.read_csv(train_data_path, sep='\t', usecols=['locus'])
        if len(train_df['locus'].unique()) != 1:
            raise ValueError(f"Multiple loci found in the training data: {train_df['locus'].unique()}. "
                             f"Please provide a single locus for the model.")
        else:
            return train_df['locus'].unique()[0]

    @staticmethod
    def divide_generated_sequences(generated_sequences_dir: str, generated_sequences_filename: str,
                                   divided_sequences_output_dir: str, n_samples_per_subset: int) -> None:
        """Divides generated sequences into smaller datasets for analysis.
        Args:
            generated_sequences_dir (str): Directory containing the generated sequences file.
            generated_sequences_filename (str): Filename of the generated sequences file (without extension).
            divided_sequences_output_dir (str): Directory to save the divided datasets.
            n_samples_per_subset (int): Number of samples per subset.
        Returns:
            None
        """
        os.makedirs(divided_sequences_output_dir, exist_ok=True)
        generated_sequences = pd.read_csv(Path(generated_sequences_dir) / f"{generated_sequences_filename}.tsv",
                                          sep='\t')
        if len(generated_sequences) < n_samples_per_subset:
            raise ValueError(
                f"Not enough samples to split: {len(generated_sequences)} rows found, but {n_samples_per_subset} required."
            )
        n_datasets = len(generated_sequences) // n_samples_per_subset
        for i in range(n_datasets):
            start_idx = i * n_samples_per_subset
            end_idx = (i + 1) * n_samples_per_subset
            split_dataset = generated_sequences.iloc[start_idx:end_idx]
            split_dataset.to_csv(Path(divided_sequences_output_dir) / f"{generated_sequences_filename}_{i}.tsv",
                                 sep='\t', index=False)

    @staticmethod
    def save_train_data(model_config: ModelConfig, output_dir: str, train_data_full_path: str,
                        train_data_file_name: str) -> None:
        """Saves the training data to the output directory and preprocesses it for CompAIRR.
        Args:
            model_config (ModelConfig): Configuration for the model training.
            output_dir (str): Directory to save the output.
            train_data_full_path (str): Full path to the training data file.
            train_data_file_name (str): Filename of the training data file (without extension).
        Returns:
            None
        """
        train_dir_dst = Path(output_dir) / "train_sequences"
        train_dir_dst.mkdir(parents=True, exist_ok=True)

        train_data_dst = train_dir_dst / f"{train_data_file_name}_{model_config.experiment}.tsv"
        os.system(f"cp -n {train_data_full_path} {train_data_dst}")

        compairr_train_dir = Path(output_dir) / "train_compairr_sequences"
        preprocess_files_for_compairr(str(train_dir_dst), str(compairr_train_dir))

    # TODO: If we decide to always have one test set, we can remove the if condition here and merge this function
    #  with the train data saving.
    @staticmethod
    def save_test_data(model_config: ModelConfig, output_dir: str) -> None:
        """Saves the test data to the output directory and preprocesses it for CompAIRR if test_dir is specified.
        Args:
            model_config (ModelConfig): Configuration for the model training.
            output_dir (str): Directory to save the output.
        Returns:
            None
        """
        if model_config.test_dir:
            test_out_dir = Path(output_dir) / "test_sequences"
            test_out_dir.mkdir(parents=True, exist_ok=True)

            test_data_dir = Path(model_config.output_dir) / model_config.test_dir
            test_data_files = [f for f in os.listdir(test_data_dir) if (test_data_dir / f).is_file()]

            for test_data_file in test_data_files:
                test_data_src = test_data_dir / test_data_file
                train_data_dst = test_out_dir / f"{Path(test_data_file).stem}_{model_config.experiment}.tsv"
                os.system(f"cp -n {test_data_src} {train_data_dst}")

            compairr_test_dir = Path(output_dir) / "test_compairr_sequences"
            preprocess_files_for_compairr(str(test_out_dir), str(compairr_test_dir))

    @staticmethod
    def save_generated_sequences(model_config: ModelConfig, output_dir: str, immuneml_output_dir: str,
                                 train_data_file_name: str) -> None:
        """Saves the generated sequences to the output directory, preprocesses them for CompAIRR and divides into smaller subsets.
        Args:
            model_config (ModelConfig): Configuration for the model training.
            output_dir (str): Directory to save the output.
            immuneml_output_dir (str): Directory where immuneML output is saved.
            train_data_file_name (str): Filename of the training data file (without extension).
        Returns:
            None
        """
        # Immuneml saves generated sequences in a specific directory structure
        immuneml_generated_sequences_dir = Path(immuneml_output_dir) / "immuneml/gen_model/generated_sequences/model"
        # Generated sequences files might have different names, so we need to find the correct one
        immuneml_generated_sequences_file = [
            f for f in os.listdir(immuneml_generated_sequences_dir)
            if f.endswith(".tsv") and (immuneml_generated_sequences_dir / f).is_file()
        ][0]
        generated_sequences_dir = Path(output_dir) / "generated_sequences" / model_config.name
        generated_sequences_dir.mkdir(parents=True, exist_ok=True)

        gen_data_src = immuneml_generated_sequences_dir / immuneml_generated_sequences_file
        gen_data_dst = generated_sequences_dir / f"{train_data_file_name}_{model_config.experiment}.tsv"
        os.system(f"cp -n {gen_data_src} {gen_data_dst}")

        compairr_model_dir = Path(output_dir) / "generated_compairr_sequences" / model_config.name
        preprocess_files_for_compairr(str(generated_sequences_dir), str(compairr_model_dir))

        TrainingOrchestrator.divide_generated_sequences(
            str(compairr_model_dir),
            f"{train_data_file_name}_{model_config.experiment}",
            str(Path(output_dir) / "generated_compairr_sequences_split" / model_config.name),
            model_config.n_subset_samples
        )

    def run_training(self, model_config: ModelConfig, output_dir: str) -> None:
        """Runs immuneML training for model in the config.
        Args:
            model_config (ModelConfig): Configuration for the model training.
            output_dir (str): Directory to save the output.
        Returns:
            None
        """
        train_data_dir = Path(model_config.output_dir) / model_config.train_dir
        train_data_files = [f for f in os.listdir(train_data_dir) if (train_data_dir / f).is_file()]

        for train_data_file in train_data_files:
            train_data_file_name = Path(train_data_file).stem

            immuneml_output_dir = Path(model_config.output_dir) / model_config.name / train_data_file_name
            train_data_full_path = train_data_dir / train_data_file
            model_config.locus = self.get_default_locus_name(str(train_data_full_path))

            self.save_train_data(model_config, output_dir, str(train_data_full_path), train_data_file_name)
            self.save_test_data(model_config, output_dir)

            immuneml_output_dir.mkdir(parents=True, exist_ok=True)
            self.run_single_training(model_config.config, str(train_data_full_path), str(immuneml_output_dir),
                                     model_config.locus)
            self.save_generated_sequences(model_config, output_dir, str(immuneml_output_dir), train_data_file_name)
