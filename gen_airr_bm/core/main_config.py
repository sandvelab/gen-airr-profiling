import yaml
from gen_airr_bm.core.simulation_config import SimulationConfig
from gen_airr_bm.core.model_config import ModelConfig

class MainConfig:
    """Main configuration class that loads YAML and initializes configs."""

    def __init__(self, yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        self.n_experiments = data.get("n_experiments", 1)
        self.simulation_configs = []
        self.models = [ModelConfig(name, model_data["config"], model_data["output_dir"])
            for name, model_data in data["models"].items()]

        base_seed = data["data_processing"]["simulation"]["seed"]
        for exp_idx in range(self.n_experiments):
            exp_seed = base_seed + exp_idx
            sim_data = data["data_processing"]["simulation"]
            self.simulation_configs.append(
                SimulationConfig(
                    method=sim_data["method"],
                    n_samples=sim_data["n_samples"],
                    model=sim_data["model"],
                    seed=exp_seed,
                    output_dir=sim_data["output_dir"] + f"_{exp_idx}"
                )
            )

    def __repr__(self):
        return (f"MainConfig(n_experiments={self.n_experiments}, simulation_configs={self.simulation_configs},"
                f" training={self.models})")
