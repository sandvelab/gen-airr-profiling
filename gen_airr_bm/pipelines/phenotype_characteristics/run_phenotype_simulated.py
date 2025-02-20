from gen_airr_bm.core.main_config import MainConfig
from gen_airr_bm.data_processing.simulation_orchestrator import SimulationOrchestrator
from gen_airr_bm.training.training_orchestrator import TrainingOrchestrator

config = MainConfig("/Users/marimam/PycharmProjects/gen-air-benchmark/configs/phenotype_simulated_1.yaml")
simulation_orchestrator = SimulationOrchestrator()
training_orchestrator = TrainingOrchestrator()

for simulation in config.simulation_configs:
    simulation_orchestrator.run_simulation(simulation)

for model in config.model_configs:
    training_orchestrator.run_phenotypes_training(model)
