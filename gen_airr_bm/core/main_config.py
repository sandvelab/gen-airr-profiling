import os

import yaml

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.core.data_generation_config import DataGenerationConfig
from gen_airr_bm.core.model_config import ModelConfig

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
        # if analyses are not present in the config, we want empty list
        self.analysis_configs = [
            AnalysisConfig(analysis["name"], analysis["model"],
                           f"{self.output_dir}/analyses/{analysis['name']}/{analysis['model']}")
            for analysis in data.get("analyses", [])
        ] if data.get("analyses") else []

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
                        model=data_generation["model"],
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
                        config=model_data["config"],
                        experiment=exp_idx,
                        train_dir=model_data.get("train_dir", ""),
                        output_dir=exp_output_dir)
                        for model_data in data["models"]]
                )

    def __repr__(self):
        return (f"MainConfig(n_experiments={self.n_experiments}, simulation_configs={self.data_generation_configs},"
                f" training={self.model_configs})")
