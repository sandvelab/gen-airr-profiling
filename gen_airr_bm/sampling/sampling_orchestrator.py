import os
from glob import glob
from pathlib import Path

import yaml

from gen_airr_bm.core.sampling_config import SamplingConfig
from gen_airr_bm.training.immuneml_runner import run_immuneml_command


class SamplingOrchestrator:

    @staticmethod
    def run_sampling(sampling_config: SamplingConfig):
        print("Running sampling...")
        model_zip_path, dataset_name = SamplingOrchestrator.find_paths(sampling_config)
        immuneml_config_path = SamplingOrchestrator.prepare_immuneml_sampling_config(sampling_config, dataset_name,
                                                                                     model_zip_path)
        immuneml_results_path = SamplingOrchestrator.run_immuneml_sampling(sampling_config, dataset_name,
                                                                           immuneml_config_path)
        SamplingOrchestrator.copy_generated_sequences(sampling_config, immuneml_results_path, dataset_name)

    @staticmethod
    def find_paths(sampling_config: SamplingConfig):
        immuneml_path = (f"{sampling_config.root_output_dir}/{sampling_config.experiment_name}/"
                         f"{sampling_config.model_name}")
        model_zips = glob(os.path.join(
            immuneml_path,
            "*",
            "immuneml/gen_model/trained_model_model/",
            "*.zip"
        ))
        zip_path = Path(model_zips[0])
        dataset_name = zip_path.parts[-5]

        return zip_path, dataset_name

    @staticmethod
    def prepare_immuneml_sampling_config(sampling_config: SamplingConfig, dataset_name: str, model_zip_path: Path):
        sampling_configs_dir = f"{sampling_config.root_output_dir}/sampling_configs"
        os.makedirs(sampling_configs_dir, exist_ok=True)

        generic_immuneml_config_path = Path(sampling_config.immuneml_config)
        immuneml_config_path = Path(
            f"{sampling_configs_dir}/{sampling_config.experiment_name}_{sampling_config.model_name}_"
            f"{dataset_name}.yaml")

        with open(generic_immuneml_config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        data["instructions"]["gen_model"]["ml_config_path"] = str(model_zip_path)
        data["instructions"]["gen_model"]["gen_examples_count"] = sampling_config.n_samples

        with open(immuneml_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        return immuneml_config_path

    @staticmethod
    def run_immuneml_sampling(sampling_config: SamplingConfig, dataset_name, immuneml_config_path: Path):
        immuneml_results_dir = (f"{sampling_config.root_output_dir}/sampling/{sampling_config.experiment_name}/"
                                f"{sampling_config.model_name}/{dataset_name}")
        os.makedirs(immuneml_results_dir, exist_ok=True)
        run_immuneml_command(immuneml_config_path, immuneml_results_dir)

        return immuneml_results_dir

    @staticmethod
    def copy_generated_sequences(sampling_configs: SamplingConfig, immuneml_results_dir: str, dataset_name: str):
        resampled_sequences_dir = (f"{sampling_configs.root_output_dir}/resampled_sequences_raw/"
                                   f"{sampling_configs.model_name}")
        os.makedirs(resampled_sequences_dir, exist_ok=True)

        immuneml_sequences_path = Path(immuneml_results_dir) / "gen_model/generated_sequences"
        immuneml_sequences_path_tsv = next(immuneml_sequences_path.glob("*.tsv"))

        experiment_n = sampling_configs.experiment_name.split("_")[-1]
        output_sequences_path = Path(resampled_sequences_dir) / f"{dataset_name}_{experiment_n}.tsv"

        os.system(f"cp {immuneml_sequences_path_tsv} {output_sequences_path}")
