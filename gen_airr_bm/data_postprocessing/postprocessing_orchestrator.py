import glob
import os

import numpy as np
import pandas as pd

from gen_airr_bm.core.postprocessing_config import PostProcessingConfig
from gen_airr_bm.utils.compairr_utils import preprocess_file_for_compairr


class PostProcessingOrchestrator:

    @staticmethod
    def run_postprocessing(postprocessing_config: PostProcessingConfig):
        print("Running post-processing...")
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
        novel_sequences_split_dir = (PostProcessingOrchestrator.generate_novel_sequences_splits
                                     (postprocessing_config, generated_compairr_sequences_no_train_dir,
                                      divided_resampled_no_train_sequences_compairr_dir, dataset_name))
        _ = PostProcessingOrchestrator.deduplicate_novel_sequences_splits(postprocessing_config,
                                                                          novel_sequences_split_dir, dataset_name)
        _ = (PostProcessingOrchestrator.merge_novel_sequences_splits
             (postprocessing_config, novel_sequences_split_dir, dataset_name))

    @staticmethod
    def remove_train_from_resampled(postprocessing_config: PostProcessingConfig, train_sequences_path: str,
                                    dataset_name: str):
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
    def generate_novel_sequences_splits(postprocessing_config: PostProcessingConfig,
                                        divided_generated_no_train_sequences_dir: str,
                                        divided_resampled_no_train_sequences_dir: str, dataset_name: str):
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
                sampled_resampled_no_train_sequences_df = resampled_no_train_sequences_df.sample(n=missing_sequences_n,
                                                                                                 random_state=42)
                generated_no_train_sequences_df = pd.concat([generated_no_train_sequences_df,
                                                             sampled_resampled_no_train_sequences_df],
                                                            ignore_index=True)

            generated_no_train_sequences_df["sequence_id"] = [f"sequence_{j}" for j in
                                                              range(len(generated_no_train_sequences_df))]
            generated_no_train_sequences_df.to_csv(f"{connected_sequences_dir}/{dataset_name}_{i}.tsv",
                                                   sep="\t", index=False)

        return connected_sequences_dir

    @staticmethod
    def deduplicate_novel_sequences_splits(postprocessing_config, novel_sequences_split_dir, dataset_name):
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
