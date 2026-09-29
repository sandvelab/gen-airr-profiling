from pathlib import Path
import os
import re
import shutil
import tempfile

import pandas as pd

from gen_airr_bm.constants.dataset_split import DatasetSplit
from gen_airr_bm.core.model_config import ModelConfig
from gen_airr_bm.training.immuneml_runner import run_immuneml_command, write_immuneml_config
from gen_airr_bm.utils.compairr_utils import preprocess_file_for_compairr


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
    def copy_file_if_missing(src_path: str, dst_path: str) -> None:
        """Copies a file unless the destination already exists. The copy is written to a temporary file first and
        then renamed, so other threads never see a partly written destination.
        Args:
            src_path (str): Path to the file to copy.
            dst_path (str): Path to copy the file to.
        Returns:
            None
        """
        if os.path.exists(dst_path):
            return
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(dst_path), suffix=".tmp")
        os.close(tmp_fd)
        try:
            shutil.copyfile(src_path, tmp_path)
            os.replace(tmp_path, dst_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def get_split_files(split_dir: Path, generated_sequences_filename: str) -> list[Path]:
        """Returns the split files belonging to one generated sequences file.
        Args:
            split_dir (Path): Directory containing the split files.
            generated_sequences_filename (str): Filename of the generated sequences file (without extension).
        Returns:
            list[Path]: Paths of the split files, e.g. <filename>_0.tsv, <filename>_1.tsv.
        """
        if not split_dir.is_dir():
            return []
        pattern = re.compile(rf"{re.escape(generated_sequences_filename)}_\d+\.tsv")
        return [f for f in split_dir.iterdir() if pattern.fullmatch(f.name)]

    @staticmethod
    def get_generated_sequences_path(model_config: ModelConfig, output_dir: str, train_data_file_name: str) -> Path:
        """Returns the path the generated sequences are copied to after immuneML training."""
        return (Path(output_dir) / "generated_sequences" / model_config.name /
                f"{train_data_file_name}_{model_config.experiment}.tsv")

    @staticmethod
    def is_training_done(model_config: ModelConfig, output_dir: str, train_data_file_name: str) -> bool:
        """Checks whether training for this model and training file already finished in an earlier run.
        Args:
            model_config (ModelConfig): Configuration for the model training.
            output_dir (str): Directory to save the output.
            train_data_file_name (str): Filename of the training data file (without extension).
        Returns:
            bool: True if the generated sequences exist and all their splits have been written.
        """
        generated_sequences_path = TrainingOrchestrator.get_generated_sequences_path(model_config, output_dir,
                                                                                     train_data_file_name)
        if not generated_sequences_path.is_file():
            return False
        n_rows = len(pd.read_csv(generated_sequences_path, sep='\t', usecols=[0]))
        split_files = TrainingOrchestrator.get_split_files(
            Path(output_dir) / "generated_compairr_sequences_split" / model_config.name,
            generated_sequences_path.stem)
        # Leftover rows beyond the last full split are dropped, as in divide_generated_sequences. A set too small
        # for a single split is never done, so a rerun reports the error again instead of skipping it.
        n_expected_splits = n_rows // model_config.n_subset_samples
        return n_expected_splits > 0 and len(split_files) == n_expected_splits

    @staticmethod
    def save_ref_data(model_config: ModelConfig, output_dir: str, ref_data_full_path: str,
                      ref_data_file_name: str, ref_name: DatasetSplit) -> None:
        """Saves the reference data (test or train) to the output directory and preprocesses it for CompAIRR.
        Args:
            model_config (ModelConfig): Configuration for the model training.
            output_dir (str): Directory to save the output.
            ref_data_full_path (str): Full path to the reference data file.
            ref_data_file_name (str): Filename of the reference data file (without extension).
            ref_name (DatasetSplit): Name of the reference dataset split (e.g., TRAIN, TEST).
        Returns:
            None
        """
        ref_data_dir_dst = Path(output_dir) / f"{ref_name.value}_sequences"
        ref_data_dir_dst.mkdir(parents=True, exist_ok=True)

        ref_data_file_dst = ref_data_dir_dst / f"{ref_data_file_name}_{model_config.experiment}.tsv"
        TrainingOrchestrator.copy_file_if_missing(ref_data_full_path, str(ref_data_file_dst))

        # Only this file is preprocessed: the folder is shared by all experiments, which may run in parallel
        compairr_ref_dir = Path(output_dir) / f"{ref_name.value}_compairr_sequences"
        compairr_ref_dir.mkdir(parents=True, exist_ok=True)
        preprocess_file_for_compairr(str(ref_data_dir_dst), str(compairr_ref_dir), ref_data_file_dst.name)

    @staticmethod
    def save_generated_sequences(model_config: ModelConfig, output_dir: str, immuneml_output_dir: str,
                                 train_data_file_name: str) -> None:
        """Copies the generated sequences from the immuneML output to the output directory, then preprocesses them
        for CompAIRR and divides them into smaller subsets.
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
        (immuneml_generated_sequences_file,) = [
            f for f in os.listdir(immuneml_generated_sequences_dir)
            if f.endswith(".tsv") and (immuneml_generated_sequences_dir / f).is_file()
        ]

        gen_data_file_path_dst = TrainingOrchestrator.get_generated_sequences_path(model_config, output_dir,
                                                                                   train_data_file_name)
        gen_data_file_path_dst.parent.mkdir(parents=True, exist_ok=True)
        gen_data_file_path_src = immuneml_generated_sequences_dir / immuneml_generated_sequences_file
        TrainingOrchestrator.copy_file_if_missing(str(gen_data_file_path_src), str(gen_data_file_path_dst))

        TrainingOrchestrator.split_generated_sequences(model_config, output_dir, train_data_file_name)

    @staticmethod
    def split_generated_sequences(model_config: ModelConfig, output_dir: str, train_data_file_name: str) -> None:
        """Preprocesses the copied generated sequences for CompAIRR and divides them into smaller subsets,
        replacing any splits left by an earlier run.
        Args:
            model_config (ModelConfig): Configuration for the model training.
            output_dir (str): Directory to save the output.
            train_data_file_name (str): Filename of the training data file (without extension).
        Returns:
            None
        """
        gen_data_file_path = TrainingOrchestrator.get_generated_sequences_path(model_config, output_dir,
                                                                               train_data_file_name)
        compairr_model_dir = Path(output_dir) / "generated_compairr_sequences" / model_config.name
        split_dir = Path(output_dir) / "generated_compairr_sequences_split" / model_config.name

        # Only this file is preprocessed: the folder is shared by all experiments, which may run in parallel
        compairr_model_dir.mkdir(parents=True, exist_ok=True)
        preprocess_file_for_compairr(str(gen_data_file_path.parent), str(compairr_model_dir), gen_data_file_path.name)

        for split_file in TrainingOrchestrator.get_split_files(split_dir, gen_data_file_path.stem):
            split_file.unlink()
        TrainingOrchestrator.divide_generated_sequences(
            str(compairr_model_dir),
            gen_data_file_path.stem,
            str(split_dir),
            model_config.n_subset_samples
        )

    @staticmethod
    def run_training(model_config: ModelConfig, output_dir: str) -> None:
        """Runs ImmuneML training and handles data saving and preprocessing for CompAIRR.
        Training files that were fully processed in an earlier run are skipped. If a file was trained but not fully
        split, only the splitting is redone.
        Args:
            model_config (ModelConfig): Configuration for the model training.
            output_dir (str): Directory to save the output.
        Returns:
            None
        """
        train_data_dir = Path(model_config.output_dir) / model_config.train_dir
        test_data_dir = Path(model_config.output_dir) / model_config.test_dir
        train_data_files = [f for f in os.listdir(train_data_dir) if (train_data_dir / f).is_file()]

        for ref_data_file in train_data_files:
            ref_data_file_name = Path(ref_data_file).stem
            run_description = f"{model_config.name} on {ref_data_file_name} (experiment {model_config.experiment})"
            if TrainingOrchestrator.is_training_done(model_config, output_dir, ref_data_file_name):
                print(f"Skipping {run_description}: generated sequences and all splits already exist.")
                continue

            train_data_full_path = train_data_dir / ref_data_file
            test_data_full_path = test_data_dir / ref_data_file
            model_config.locus = TrainingOrchestrator.get_default_locus_name(str(train_data_full_path))

            TrainingOrchestrator.save_ref_data(model_config, output_dir, str(train_data_full_path),
                                               ref_data_file_name, DatasetSplit.TRAIN)
            TrainingOrchestrator.save_ref_data(model_config, output_dir, str(test_data_full_path), ref_data_file_name,
                                               DatasetSplit.TEST)

            # Generated sequences are only copied after immuneML finished, so they can be split without retraining
            if TrainingOrchestrator.get_generated_sequences_path(model_config, output_dir,
                                                                 ref_data_file_name).is_file():
                print(f"Re-splitting {run_description}: generated sequences exist but splits are incomplete.")
                TrainingOrchestrator.split_generated_sequences(model_config, output_dir, ref_data_file_name)
                continue

            immuneml_output_dir = Path(model_config.output_dir) / model_config.name / ref_data_file_name
            immuneml_output_dir.mkdir(parents=True, exist_ok=True)

            TrainingOrchestrator.run_single_training(model_config.immuneml_model_config, str(train_data_full_path),
                                                     str(immuneml_output_dir),
                                                     model_config.locus)
            TrainingOrchestrator.save_generated_sequences(model_config, output_dir, str(immuneml_output_dir),
                                                          ref_data_file_name)
