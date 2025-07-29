import argparse
import concurrent.futures

from gen_airr_bm.analysis.analysis_orchestrator import AnalysisOrchestrator
from gen_airr_bm.core.main_config import MainConfig
from gen_airr_bm.data_processing.data_generation_orchestrator import DataGenerationOrchestrator
from gen_airr_bm.training.training_orchestrator import TrainingOrchestrator


def run_data_generation(data_generation, orchestrator):
    print(f"Running data generation: {data_generation}")
    orchestrator.run_data_generation(data_generation)


def run_training(model, orchestrator, output_dir):
    print(f"Training model: {model}")
    orchestrator.run_training(model, output_dir)


def run_analysis(analysis, orchestrator):
    print(f"Running analysis: {analysis}")
    orchestrator.run_analysis(analysis)


def main(config_path):
    config = MainConfig(config_path)

    data_generation_orchestrator = DataGenerationOrchestrator()
    training_orchestrator = TrainingOrchestrator()
    analysis_orchestrator = AnalysisOrchestrator()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda data_generation: run_data_generation(data_generation, data_generation_orchestrator), config.data_generation_configs)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda model: run_training(model, training_orchestrator, config.output_dir), config.model_configs)

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     executor.map(lambda analysis: run_analysis(analysis, analysis_orchestrator),
    #                  config.analysis_configs)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for _ in executor.map(lambda analysis: run_analysis(analysis, analysis_orchestrator),
                              config.analysis_configs):
            pass  # This will re-raise exceptions from worker threads in the main thread


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AIRR benchmark pipelines.")
    parser.add_argument("config", type=str, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    main(args.config)
