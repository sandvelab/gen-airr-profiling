import os
import yaml

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.core.data_generation_config import DataGenerationConfig
from gen_airr_bm.core.model_config import ModelConfig
from gen_airr_bm.core.tuning_config import TuningConfig


class MainConfig:
    """Main configuration class that loads YAML and initializes configs."""

    def __init__(self, yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        self.n_experiments = data["n_experiments"]
        self.output_dir = data["output_dir"]
        self.input_dir = data.get("input_dir", None)
        self.data_generation_configs = []
        self.model_configs = []
        self.analysis_configs = []
        self.tuning_configs = []

        base_seed = data["seed"]
        experimental_datasets = []
        if self.input_dir:
            experimental_datasets.extend(os.listdir(self.input_dir))
            experimental_datasets = [f"{self.input_dir}/{dataset}" for dataset in experimental_datasets]

        for exp_idx in range(self.n_experiments):
            exp_seed = base_seed + exp_idx
            exp_output_dir = self.output_dir + f"/exp_{exp_idx}"
            if "data_generation" in data:
                data_generation = data["data_generation"]
                experimental_dataset = experimental_datasets[exp_idx] if data_generation["experimental"] else None
                self.data_generation_configs.append(
                    DataGenerationConfig(
                        method=data_generation["method"],
                        n_samples=data_generation["n_samples"],
                        data_file=experimental_dataset,
                        experimental=data_generation["experimental"],
                        default_model_name=data_generation["default_model_name"],
                        experiment=exp_idx,
                        seed=exp_seed,
                        output_dir=exp_output_dir,
                        input_columns=data_generation.get("input_columns", None)
                    )
                )
            if "models" in data:
                self.model_configs.extend(
                    [ModelConfig(
                        name=model_data["name"],
                        immuneml_model_config=model_data["immuneml_model_config"],
                        experiment=exp_idx,
                        train_dir=model_data.get("train_dir", ""),
                        test_dir=model_data.get("test_dir", None),
                        output_dir=exp_output_dir,
                        n_subset_samples=model_data.get("n_subset_samples", None))
                        for model_data in data["models"]]
                )

        if "analyses" in data:
            self.analysis_configs.extend([
                AnalysisConfig(analysis=analysis["name"],
                               model_names=analysis["model_names"],
                               analysis_output_dir=f"{self.output_dir}/analyses/{analysis['name']}/"
                                                   f"{'_'.join(m.lower() for m in analysis['model_names'])}",
                               root_output_dir=self.output_dir,
                               default_model_name=analysis["default_model_name"],
                               reference_data=analysis.get("reference_data", None),
                               n_subsets=analysis.get("n_subsets", None))
                for analysis in data.get("analyses", [])
            ])

        if "tuning" in data:
            self.tuning_configs.extend([
                TuningConfig(tuning_method=tuning["tuning_method"],
                             model_names=tuning["model_names"],
                             reference_data=tuning["reference_data"],
                             tuning_output_dir=f"{self.output_dir}/tuning/{tuning['tuning_method']}/"
                                               f"{'_'.join(m.lower() for m in tuning['model_names'])}",
                             root_output_dir=self.output_dir)
                for tuning in data.get("tuning", [])
            ])

    def __repr__(self):
        return (f"MainConfig(n_experiments={self.n_experiments}, simulation_configs={self.data_generation_configs},"
                f" training={self.model_configs}, analyses={self.analysis_configs}, tuning={self.tuning_configs})")
