import yaml

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.core.simulation_config import SimulationConfig
from gen_airr_bm.core.model_config import ModelConfig

class MainConfig:
    """Main configuration class that loads YAML and initializes configs."""

    def __init__(self, yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        self.n_experiments = data["n_experiments"]
        self.output_dir = data["output_dir"]
        self.simulation_configs = []
        self.model_configs = []
        # if analyses are not present in the config, we want empty list
        self.analysis_configs = [
            AnalysisConfig(analysis["name"], analysis["model"],
                           f"{self.output_dir}/analyses/{analysis['name']}/{analysis['model']}")
            for analysis in data.get("analyses", [])
        ] if data.get("analyses") else []

        base_seed = data["seed"]
        for exp_idx in range(self.n_experiments):
            exp_seed = base_seed + exp_idx
            exp_output_dir = self.output_dir + f"/exp_{exp_idx}"
            if "simulation" in data:
                sim_data = data["simulation"]
                self.simulation_configs.append(
                    SimulationConfig(
                        method=sim_data["method"],
                        n_samples=sim_data["n_samples"],
                        model=sim_data["model"],
                        experiment=exp_idx,
                        seed=exp_seed,
                        output_dir=exp_output_dir
                    )
                )
            if "models" in data:
                self.model_configs.extend(
                    [ModelConfig(
                        name=model_data["name"],
                        config=model_data["config"],
                        experiment=exp_idx,
                        output_dir=exp_output_dir)
                        for model_data in data["models"]]
                )

    def __repr__(self):
        return (f"MainConfig(n_experiments={self.n_experiments}, simulation_configs={self.simulation_configs},"
                f" training={self.model_configs})")
