import os
import shutil
import zipfile
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
        if SamplingOrchestrator.is_sampling_done(sampling_config, dataset_name):
            print(f"Skipping sampling of {sampling_config.model_name} for {sampling_config.experiment_name}: "
                  f"outputs already exist.")
            return
        immuneml_results_path = SamplingOrchestrator.get_immuneml_results_dir(sampling_config, dataset_name)
        if SamplingOrchestrator.is_immuneml_run_finished(sampling_config, immuneml_results_path):
            print(f"immuneML already finished in {immuneml_results_path}, copying its outputs.")
        else:
            if sampling_config.model_parameters:
                SamplingOrchestrator.set_model_parameters(model_zip_path, sampling_config.model_parameters)
            immuneml_config_path = SamplingOrchestrator.prepare_immuneml_sampling_config(sampling_config, dataset_name,
                                                                                         model_zip_path)
            immuneml_results_path = SamplingOrchestrator.run_immuneml_sampling(sampling_config, dataset_name,
                                                                               immuneml_config_path)
        if not sampling_config.p_gens_only:
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
    def get_output_paths(sampling_config: SamplingConfig, dataset_name: str) -> list:
        """Returns the files a finished sampling run writes: the generated sequences (unless only p_gens are computed)
        and the train and test p_gens (if computed)."""
        experiment_n = SamplingOrchestrator.get_experiment_number(sampling_config)
        root_dir = Path(sampling_config.root_output_dir)
        file_name = f"{dataset_name}_{experiment_n}.tsv"
        paths = []
        if not sampling_config.p_gens_only:
            paths.append(root_dir / "resampled_sequences_raw" / sampling_config.model_name / file_name)
        if sampling_config.compute_p_gens:
            paths.extend(root_dir / f"{split.value}_p_gens" / sampling_config.model_name / file_name
                         for split in DatasetSplit.train_and_test())
        return paths

    @staticmethod
    def is_sampling_done(sampling_config: SamplingConfig, dataset_name: str) -> bool:
        """The outputs are only copied after immuneML finished, so a run with all of them in place is complete."""
        return all(path.is_file() for path in SamplingOrchestrator.get_output_paths(sampling_config, dataset_name))

    @staticmethod
    def set_model_parameters(model_zip_path: Path, parameters: dict):
        """Sets parameters in the model_overview.yaml of the trained model zip, since immuneML loads the model for
        sampling with the parameters it was trained with and ApplyGenModel cannot change them, e.g. the device or the
        number of importance samples for the VAE p_gens (iter_count_prob_estimation). Before the first change, the
        original zip is kept as <name>.zip.orig next to it, which find_paths does not pick up. Parameters the model does
        not have (e.g. device for PWM or SoNNia) are left out, since the model could not be loaded with them.
        """
        with zipfile.ZipFile(model_zip_path) as zip_file:
            entries = [(info, zip_file.read(info.filename)) for info in zip_file.infolist()]

        overview_index = next((i for i, (info, _) in enumerate(entries) if info.filename == "model_overview.yaml"),
                              None)
        if overview_index is None:
            raise FileNotFoundError(f"{model_zip_path} has no model_overview.yaml, so its parameters cannot be set.")
        overview_info, overview_content = entries[overview_index]
        model_overview = yaml.safe_load(overview_content)

        missing = [name for name in parameters if name not in model_overview]
        if missing:
            print(f"{model_overview.get('type')} in {model_zip_path} has no parameter(s) {missing}, leaving them out.")
        changes = {name: value for name, value in parameters.items()
                   if name in model_overview and model_overview[name] != value}
        if not changes:
            return

        backup_path = model_zip_path.with_name(f"{model_zip_path.name}.orig")
        if not backup_path.exists():
            shutil.copy2(model_zip_path, backup_path)

        model_overview.update(changes)
        entries[overview_index] = (overview_info, yaml.safe_dump(model_overview, sort_keys=False).encode("utf-8"))

        tmp_path = model_zip_path.with_name(f"{model_zip_path.name}.tmp")
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for info, content in entries:
                zip_file.writestr(info, content)
        os.replace(tmp_path, model_zip_path)
        print(f"Set {changes} in {model_zip_path} (original kept as {backup_path}).")

    @staticmethod
    def prepare_immuneml_sampling_config(sampling_config: SamplingConfig, dataset_name: str, model_zip_path: Path):
        sampling_configs_dir = (f"{sampling_config.root_output_dir}/"
                                f"{SamplingOrchestrator.get_sampling_dir_name(sampling_config)}_configs")
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
    def get_sampling_dir_name(sampling_config: SamplingConfig) -> str:
        """p_gen-only runs write their immuneML configs and results to p_gen_runs(_configs) instead of
        sampling(_configs), so that the outputs of an earlier sampling run of the same model are kept."""
        return "p_gen_runs" if sampling_config.p_gens_only else "sampling"

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
    def get_immuneml_results_dir(sampling_config: SamplingConfig, dataset_name: str) -> str:
        return (f"{sampling_config.root_output_dir}/{SamplingOrchestrator.get_sampling_dir_name(sampling_config)}/"
                f"{sampling_config.experiment_name}/{sampling_config.model_name}/{dataset_name}")

    @staticmethod
    def is_immuneml_run_finished(sampling_config: SamplingConfig, immuneml_results_dir: str) -> bool:
        """Whether immuneML finished in this directory and wrote all outputs, while their copies are missing, e.g. if
        the pipeline was stopped while immuneML kept running. Its outputs can then be copied without sampling again."""
        results_dir = Path(immuneml_results_dir)
        log_path = results_dir / "log.txt"
        if not log_path.is_file():
            return False
        # the log can be very large (e.g. one line per generated sequence), and immuneML logs this at the very end
        with log_path.open("rb") as log_file:
            log_file.seek(max(0, log_path.stat().st_size - 64 * 1024))
            if b"finished analysis" not in log_file.read():
                return False
        if not sampling_config.p_gens_only and not any((results_dir / "gen_model/generated_sequences").glob("*.tsv")):
            return False
        if sampling_config.compute_p_gens and not all(
                (results_dir / "gen_model/datasets_with_p_gens" / f"{split.value}_p_gens.tsv").is_file()
                for split in DatasetSplit.train_and_test()):
            return False
        return True

    @staticmethod
    def run_immuneml_sampling(sampling_config: SamplingConfig, dataset_name, immuneml_config_path: Path):
        immuneml_results_dir = SamplingOrchestrator.get_immuneml_results_dir(sampling_config, dataset_name)
        # Left over from an earlier run that did not finish (finished runs are skipped before). immuneML would write
        # to a new timestamped directory instead of a non-empty one, and the outputs would then not be found here.
        if os.path.isdir(immuneml_results_dir):
            print(f"Removing the results of an unfinished earlier run in {immuneml_results_dir}.")
            shutil.rmtree(immuneml_results_dir)
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
