import glob
import os

import numpy as np
import pandas as pd

from gen_airr_bm.core.postprocessing_config import PostProcessingConfig
from gen_airr_bm.utils.compairr_utils import preprocess_file_for_compairr


class PostProcessingOrchestrator:

    @staticmethod
    def run_postprocessing(postprocessing_config: PostProcessingConfig):
        """ Orchestrates the post-processing steps to prepare generated sequences for statistical analysis. It aims
        to merge initially generated sequences with resampled sequences to ensure a sufficient number of novel sequences
        for analysis, while removing any sequences that were present in the training data. The steps include:
        1. Removing training sequences from the resampled sequences.
        2. Preprocessing the resampled sequences for CompAIRR.
        3. Dividing the resampled sequences into subsets to ensure fair division.
        4. Removing training sequences from the initially generated sequences.
        5. Collecting novel sequence splits by merging the initially generated and resampled sequences, ensuring
        a sufficient number of novel sequences.
        6. Creating deduplicated, novel sequence splits to ensure unique sequences for the innovation analysis.
        7. Merging the novel sequence splits back into a single file for phenotype analysis.
        Args:
            postprocessing_config (PostProcessingConfig): Configuration object containing paths and parameters for
            post-processing.
        Returns:
            None
        """
        exp_id = int(postprocessing_config.experiment_name.split("_")[1])
        train_sequences_path = glob.glob(f"{postprocessing_config.root_output_dir}"
                                         f"/train_compairr_sequences/*_{exp_id}.tsv")[0]
        dataset_name = os.path.splitext(os.path.basename(train_sequences_path))[0]
        resampled_no_train_sequences_path, resampled_no_train_sequences_dir = \
            (PostProcessingOrchestrator.remove_train_from_resampled(postprocessing_config, train_sequences_path,
                                                                    dataset_name))
        resampled_no_train_sequences_compairr_dir = (f"{postprocessing_config.root_output_dir}"
                                                     f"/resampled_no_train_compairr_sequences/"
                                                     f"{postprocessing_config.model_name}")
        os.makedirs(resampled_no_train_sequences_compairr_dir, exist_ok=True)
        resampled_no_train_sequences_compairr_path = preprocess_file_for_compairr(
            resampled_no_train_sequences_dir, resampled_no_train_sequences_compairr_dir, f"{dataset_name}.tsv")
        divided_resampled_no_train_sequences_compairr_dir = PostProcessingOrchestrator.divide_resampled_sequences(
            postprocessing_config, resampled_no_train_sequences_compairr_path, dataset_name)
        generated_compairr_sequences_no_train_dir = (PostProcessingOrchestrator.remove_train_from_generated
                                                     (postprocessing_config, dataset_name, train_sequences_path))
        novel_sequences_split_dir = (PostProcessingOrchestrator.collect_novel_sequences_splits
                                     (postprocessing_config, generated_compairr_sequences_no_train_dir,
                                      divided_resampled_no_train_sequences_compairr_dir, dataset_name))
        _ = PostProcessingOrchestrator.deduplicate_novel_sequences_splits(postprocessing_config,
                                                                          novel_sequences_split_dir, dataset_name)
        _ = (PostProcessingOrchestrator.merge_novel_sequences_splits
             (postprocessing_config, novel_sequences_split_dir, dataset_name))

    @staticmethod
    def remove_train_from_resampled(postprocessing_config: PostProcessingConfig, train_sequences_path: str,
                                    dataset_name: str):
        """ Removes sequences present in the training data from the resampled sequences to ensure that only novel
        sequences are retained for analysis. It reads both the training and resampled sequences, identifies the unique
        sequences in the resampled set that are not present in the training set, and saves the resulting novel sequences
        to a new file in "resampled_no_train_sequences" folder.
        Args:
            postprocessing_config (PostProcessingConfig): Configuration object containing paths and parameters for
            post-processing.
            train_sequences_path (str): Path to the file containing training sequences.
            dataset_name (str): Name of the dataset being processed, used for naming the output file.
        Returns:
            Tuple[str, str]: A tuple containing the path to the file with novel resampled sequences and the directory
            where it is stored.
        """
        resampled_sequences_path = (f"{postprocessing_config.root_output_dir}/resampled_sequences_raw/"
                                    f"{postprocessing_config.model_name}/{dataset_name}.tsv")
        resampled_no_train_sequences_dir = f"{postprocessing_config.root_output_dir}/resampled_no_train_sequences/" \
                                           f"{postprocessing_config.model_name}"
        os.makedirs(resampled_no_train_sequences_dir, exist_ok=True)
        resampled_no_train_sequences_path = f"{resampled_no_train_sequences_dir}/{dataset_name}.tsv"

        train_sequences_df = pd.read_csv(train_sequences_path, sep="\t")
        resampled_sequences_df = pd.read_csv(resampled_sequences_path, sep="\t")

        resampled_sequences_no_train_df = resampled_sequences_df[~resampled_sequences_df["junction_aa"].isin(
            train_sequences_df["junction_aa"])]
        resampled_sequences_no_train_df.to_csv(resampled_no_train_sequences_path, sep="\t", index=False)

        return resampled_no_train_sequences_path, resampled_no_train_sequences_dir

    @staticmethod
    def divide_resampled_sequences(postprocessing_config: PostProcessingConfig,
                                   resampled_no_train_sequences_compairr_path, dataset_name):
        """ Divides the resampled sequences that do not overlap with the training data into subsets to ensure a fair
        division of sequences for merging with the initially generated sequences splits. It reads the novel resampled
        sequences, and splits the sequences into equal parts. Each subset is saved to a separate file
        in the "resampled_no_train_compairr_sequences_split" folder for later use in merging with the generated
        sequences.
        Args:
            postprocessing_config (PostProcessingConfig): Configuration object containing paths and parameters for
            post-processing.
            resampled_no_train_sequences_compairr_path (str): Path to the file containing novel resampled sequences that
            have been preprocessed for CompAIRR.
            dataset_name (str): Name of the dataset being processed, used for naming the output files.
        Returns:
            str: Path to the directory where the divided resampled sequences are stored.
        """
        divided_sequences_output_dir = (f"{postprocessing_config.root_output_dir}/"
                                        f"resampled_no_train_compairr_sequences_split/"
                                        f"{postprocessing_config.model_name}")
        os.makedirs(divided_sequences_output_dir, exist_ok=True)
        resampled_no_train_sequences = pd.read_csv(resampled_no_train_sequences_compairr_path, sep="\t")
        n_samples_per_subset = len(resampled_no_train_sequences) // postprocessing_config.n_subsets

        if len(resampled_no_train_sequences) < n_samples_per_subset:
            raise ValueError(f"Not enough sequences to divide! Requested {n_samples_per_subset}, but only "
                             f"{len(resampled_no_train_sequences)} available.")
        splits = np.array_split(resampled_no_train_sequences, postprocessing_config.n_subsets)

        for i, subset_df in enumerate(splits):
            subset_df.to_csv(f"{divided_sequences_output_dir}/{dataset_name}_{i}.tsv",
                             sep="\t", index=False)

        return divided_sequences_output_dir

    @staticmethod
    def remove_train_from_generated(postprocessing_config, dataset_name, train_sequences_path):
        """ Removes sequences present in the training data from the initially generated sequences, and saves
        the resulting novel sequences to new files in "generated_no_train_compairr_sequences_split" folder for each
        subset of generated sequences.
        Args:
            postprocessing_config (PostProcessingConfig): Configuration object containing paths and parameters for
            post-processing.
            dataset_name (str): Name of the dataset being processed, used for naming the output files.
            train_sequences_path (str): Path to the file containing training sequences.
        Returns:
            str: Path to the directory where the generated sequences with training sequences removed are stored.
        """
        generated_sequences_split_dir = (f"{postprocessing_config.root_output_dir}/generated_compairr_sequences_split/"
                                         f"{postprocessing_config.model_name}")
        generated_no_train_sequences_split_dir = (f"{postprocessing_config.root_output_dir}/"
                                                  f"generated_no_train_compairr_sequences_split/"
                                                  f"{postprocessing_config.model_name}")
        os.makedirs(generated_no_train_sequences_split_dir, exist_ok=True)
        train_sequences_df = pd.read_csv(train_sequences_path, sep="\t")
        for i in range(postprocessing_config.n_subsets):
            generated_sequences_path = f"{generated_sequences_split_dir}/{dataset_name}_{i}.tsv"
            generated_sequences_df = pd.read_csv(generated_sequences_path, sep="\t")
            generated_sequences_no_train_df = generated_sequences_df[~generated_sequences_df["junction_aa"].isin(
                train_sequences_df["junction_aa"])]
            generated_sequences_no_train_df.to_csv(f"{generated_no_train_sequences_split_dir}/{dataset_name}_{i}.tsv",
                                                   sep="\t", index=False)

        return generated_no_train_sequences_split_dir

    @staticmethod
    def collect_novel_sequences_splits(postprocessing_config: PostProcessingConfig,
                                       divided_generated_no_train_sequences_dir: str,
                                       divided_resampled_no_train_sequences_dir: str, dataset_name: str):
        """ Collects novel sequence splits by merging the initially generated sequences (with training sequences
        removed) with the resampled sequences (with training sequences removed) to ensure a sufficient number of novel
        sequences for analysis. For each subset of generated sequences, it reads the corresponding subset of resampled
        sequences, identifies how many novel sequences are missing from the generated subset to reach the desired number
        of samples, and randomly samples the required number of novel sequences from the resampled subset. The merged
        novel sequences are then saved to new files in "novel_generated_compairr_sequences_split" folder for each
        subset.
        Args:
            postprocessing_config (PostProcessingConfig): Configuration object containing paths and parameters for
            post-processing.
            divided_generated_no_train_sequences_dir (str): Path to the directory containing the generated sequences
            with training sequences removed, divided into subsets.
            divided_resampled_no_train_sequences_dir (str): Path to the directory containing the resampled sequences
            with training sequences removed, divided into subsets.
            dataset_name (str): Name of the dataset being processed, used for naming the output files.
        Returns:
            str: Path to the directory where the merged novel sequence splits are stored.
        """
        connected_sequences_dir = (f"{postprocessing_config.root_output_dir}/novel_generated_compairr_sequences_split/"
                                   f"{postprocessing_config.model_name}")
        os.makedirs(connected_sequences_dir, exist_ok=True)

        for i in range(postprocessing_config.n_subsets):
            generated_no_train_sequences_path = f"{divided_generated_no_train_sequences_dir}/{dataset_name}_{i}.tsv"
            resampled_no_train_sequences_path = f"{divided_resampled_no_train_sequences_dir}/{dataset_name}_{i}.tsv"

            generated_no_train_sequences_df = pd.read_csv(generated_no_train_sequences_path, sep="\t")
            resampled_no_train_sequences_df = pd.read_csv(resampled_no_train_sequences_path, sep="\t")

            missing_sequences_n = postprocessing_config.n_samples - len(generated_no_train_sequences_df)

            if missing_sequences_n > 0:
                resampled_no_train_sequences_for_backfill_df = (resampled_no_train_sequences_df.sample
                                                                (n=missing_sequences_n, random_state=42))
                generated_no_train_sequences_df = pd.concat([generated_no_train_sequences_df,
                                                             resampled_no_train_sequences_for_backfill_df],
                                                            ignore_index=True)

            generated_no_train_sequences_df["sequence_id"] = [f"sequence_{j}" for j in
                                                              range(len(generated_no_train_sequences_df))]
            generated_no_train_sequences_df.to_csv(f"{connected_sequences_dir}/{dataset_name}_{i}.tsv",
                                                   sep="\t", index=False)

        return connected_sequences_dir

    @staticmethod
    def deduplicate_novel_sequences_splits(postprocessing_config, novel_sequences_split_dir, dataset_name):
        """ For each subset of merged novel sequences, it reads the sequences, removes any duplicates based on the
        "junction_aa" column, and saves the resulting unique sequences to new files in
        "novel_unique_generated_compairr_sequences_split" folder for each subset. This step is crucial for the
        innovation analysis, which is performed on unique sequences.
        Args:
            postprocessing_config (PostProcessingConfig): Configuration object containing paths and parameters for
            post-processing.
            novel_sequences_split_dir (str): Path to the directory containing the merged novel sequence splits.
            dataset_name (str): Name of the dataset being processed, used for naming the output files.
        Returns:
            str: Path to the directory where the deduplicated novel sequence splits are stored.
        """
        deduplicated_novel_sequences_dir = (f"{postprocessing_config.root_output_dir}/"
                                            f"novel_unique_generated_compairr_sequences_split/"
                                            f"{postprocessing_config.model_name}")
        os.makedirs(deduplicated_novel_sequences_dir, exist_ok=True)
        for i in range(postprocessing_config.n_subsets):
            split_path = f"{novel_sequences_split_dir}/{dataset_name}_{i}.tsv"
            split_df = pd.read_csv(split_path, sep="\t")
            deduplicated_split_df = split_df.drop_duplicates(subset=["junction_aa"])
            deduplicated_split_df.to_csv(f"{deduplicated_novel_sequences_dir}/{dataset_name}_{i}.tsv",
                                         sep="\t", index=False)

        return deduplicated_novel_sequences_dir

    @staticmethod
    def merge_novel_sequences_splits(postprocessing_config, novel_sequences_split_dir, dataset_name):
        """ Merges the deduplicated novel sequence splits into a single file for phenotype analysis. It reads each
        subset of novel sequences, concatenates them into a single DataFrame, assigns unique sequence IDs, and saves the
        merged novel sequences to a new file in "novel_generated_compairr_sequences" folder for the phenotype analysis.
        Args:
            postprocessing_config (PostProcessingConfig): Configuration object containing paths and parameters for
            post-processing.
            novel_sequences_split_dir (str): Path to the directory containing the deduplicated novel sequence splits.
            dataset_name (str): Name of the dataset being processed, used for naming the output file.
        Returns:
            str: Path to the file containing the merged novel sequences for phenotype analysis.
        """
        merged_novel_sequences_dir = (f"{postprocessing_config.root_output_dir}/novel_generated_compairr_sequences/"
                                      f"{postprocessing_config.model_name}")
        os.makedirs(merged_novel_sequences_dir, exist_ok=True)
        all_df_splits = []
        for i in range(postprocessing_config.n_subsets):
            split_path = f"{novel_sequences_split_dir}/{dataset_name}_{i}.tsv"
            split_df = pd.read_csv(split_path, sep="\t")
            all_df_splits.append(split_df)
        merged_df = pd.concat(all_df_splits, ignore_index=True)
        merged_df["sequence_id"] = [f"sequence_{i}" for i in range(len(merged_df))]
        merged_df.to_csv(f"{merged_novel_sequences_dir}/{dataset_name}.tsv", sep="\t", index=False)

        return f"{merged_novel_sequences_dir}/{dataset_name}.tsv"
