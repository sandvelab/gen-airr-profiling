import os
from glob import glob
from pathlib import Path

import pandas as pd
import yaml

from gen_airr_bm.constants.dataset_split import DatasetSplit
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
        if sampling_config.compute_p_gens:
            SamplingOrchestrator.copy_p_gens(sampling_config, immuneml_results_path, dataset_name)

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
        if sampling_config.compute_p_gens:
            SamplingOrchestrator.add_p_gen_datasets(data, sampling_config, dataset_name)

        with open(immuneml_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        return immuneml_config_path

    @staticmethod
    def add_p_gen_datasets(data: dict, sampling_config: SamplingConfig, dataset_name: str):
        """Adds the train and test sequences of this experiment as immuneML datasets and lists them under
        p_gen_datasets, so ApplyGenModel computes their generation probabilities with the loaded model. Only models
        that can compute p_gens (e.g. SakaLSTM, TCRpeg, TCRsep) accept p_gen_datasets; immuneML raises an error for
        the others.
        """
        if not data.get("definitions"):
            data["definitions"] = {}
        datasets = data["definitions"].setdefault("datasets", {})

        for split in DatasetSplit.train_and_test():
            ref_sequences_path = SamplingOrchestrator.get_ref_sequences_path(sampling_config, dataset_name, split)
            if not ref_sequences_path.is_file():
                raise FileNotFoundError(f"Cannot compute p_gens for {split.value} sequences: {ref_sequences_path} "
                                        f"does not exist.")
            # Same import settings as the dataset the models were trained on
            datasets[split.value] = {
                "format": "AIRR",
                "params": {
                    "path": str(ref_sequences_path),
                    "is_repertoire": False,
                    "paired": False,
                    "region_type": "IMGT_JUNCTION",
                    "separator": "\t",
                }
            }

        data["instructions"]["gen_model"]["p_gen_datasets"] = [split.value for split in DatasetSplit.train_and_test()]

    @staticmethod
    def get_experiment_number(sampling_config: SamplingConfig) -> str:
        return sampling_config.experiment_name.split("_")[-1]

    @staticmethod
    def get_ref_sequences_path(sampling_config: SamplingConfig, dataset_name: str, split: DatasetSplit) -> Path:
        """Returns the train or test sequences that training copied to the root output directory."""
        experiment_n = SamplingOrchestrator.get_experiment_number(sampling_config)
        return (Path(sampling_config.root_output_dir) / f"{split.value}_sequences" /
                f"{dataset_name}_{experiment_n}.tsv")

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

        experiment_n = SamplingOrchestrator.get_experiment_number(sampling_configs)
        output_sequences_path = Path(resampled_sequences_dir) / f"{dataset_name}_{experiment_n}.tsv"

        os.system(f"cp {immuneml_sequences_path_tsv} {output_sequences_path}")

    @staticmethod
    def copy_p_gens(sampling_config: SamplingConfig, immuneml_results_dir: str, dataset_name: str):
        """Copies the train and test sequences with their p_gens from the immuneML output to
        <root_output_dir>/<split>_p_gens/<model_name>/<dataset_name>_<experiment_n>.tsv. Only junction_aa, v_call,
        j_call and the p_gen column are kept, as immuneML exports all AIRR fields, most of them empty."""
        experiment_n = SamplingOrchestrator.get_experiment_number(sampling_config)
        immuneml_p_gens_dir = Path(immuneml_results_dir) / "gen_model/datasets_with_p_gens"

        for split in DatasetSplit.train_and_test():
            p_gens_dir = Path(sampling_config.root_output_dir) / f"{split.value}_p_gens" / sampling_config.model_name
            p_gens_dir.mkdir(parents=True, exist_ok=True)
            p_gens_df = pd.read_csv(immuneml_p_gens_dir / f"{split.value}_p_gens.tsv", sep="\t")
            # immuneML names the p_gen column after the model, e.g. p_gen_model
            p_gen_columns = [column for column in p_gens_df.columns if column.startswith("p_gen_")]
            p_gens_df[["junction_aa", "v_call", "j_call", *p_gen_columns]].to_csv(
                p_gens_dir / f"{dataset_name}_{experiment_n}.tsv", sep="\t", index=False)
