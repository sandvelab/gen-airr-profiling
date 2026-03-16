import argparse
import concurrent.futures

from gen_airr_bm.analysis.analysis_orchestrator import AnalysisOrchestrator
from gen_airr_bm.core.main_config import MainConfig
from gen_airr_bm.data_processing.data_generation_orchestrator import DataGenerationOrchestrator
from gen_airr_bm.sampling.sampling_orchestrator import SamplingOrchestrator
from gen_airr_bm.training.training_orchestrator import TrainingOrchestrator
from gen_airr_bm.tuning.tuning_orchestrator import TuningOrchestrator


def run_data_generation(data_generation, orchestrator):
    print(f"Running data generation: {data_generation}")
    orchestrator.run_data_generation(data_generation)


def run_training(model, orchestrator, output_dir):
    print(f"Training model: {model}")
    orchestrator.run_training(model, output_dir)


def run_analysis(analysis, orchestrator):
    print(f"Running analysis: {analysis}")
    orchestrator.run_analysis(analysis)


def run_tuning(tuning, orchestrator):
    print(f"Running tuning: {tuning}")
    orchestrator.run_tuning(tuning)


def run_sampling(sampling, orchestrator):
    print(f"Running sampling: {sampling}")
    orchestrator.run_sampling(sampling)


def main(config_path, break_main=False, parallel=True):
    config = MainConfig(config_path)

    data_generation_orchestrator = DataGenerationOrchestrator()
    training_orchestrator = TrainingOrchestrator()
    analysis_orchestrator = AnalysisOrchestrator()
    tuning_orchestrator = TuningOrchestrator()
    sampling_orchestrator = SamplingOrchestrator()

    if parallel:
        if break_main:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for _ in executor.map(
                        lambda data_generation: run_data_generation(data_generation, data_generation_orchestrator),
                        config.data_generation_configs):
                    pass

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for _ in executor.map(
                        lambda model: run_training(model, training_orchestrator, config.output_dir),
                        config.model_configs):
                    pass

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for _ in executor.map(
                        lambda analysis: run_analysis(analysis, analysis_orchestrator),
                        config.analysis_configs):
                    pass

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for _ in executor.map(
                        lambda tuning: run_tuning(tuning, tuning_orchestrator),
                        config.tuning_configs):
                    pass

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for _ in executor.map(
                        lambda sampling: run_sampling(sampling, sampling_orchestrator),
                        config.sampling_configs):
                    pass

        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(
                    lambda data_generation: run_data_generation(data_generation, data_generation_orchestrator),
                    config.data_generation_configs)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(
                    lambda model: run_training(model, training_orchestrator, config.output_dir),
                    config.model_configs)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(
                    lambda analysis: run_analysis(analysis, analysis_orchestrator),
                    config.analysis_configs)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(
                    lambda tuning: run_tuning(tuning, tuning_orchestrator),
                    config.tuning_configs)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(
                    lambda sampling: run_sampling(sampling, sampling_orchestrator),
                    config.sampling_configs)
    else:
        for data_generation in config.data_generation_configs:
            run_data_generation(data_generation, data_generation_orchestrator)

        for model in config.model_configs:
            run_training(model, training_orchestrator, config.output_dir)

        for analysis in config.analysis_configs:
            run_analysis(analysis, analysis_orchestrator)

        for tuning in config.tuning_configs:
            run_tuning(tuning, tuning_orchestrator)

        for sampling in config.sampling_configs:
            run_sampling(sampling, sampling_orchestrator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AIRR benchmark pipelines.")
    parser.add_argument("config", type=str, help="Path to the configuration YAML file.")
    parser.add_argument("--break_main", type=bool, default=False,
                        help="If true, the main program will break in case of an error in any of the processes.")
    parser.add_argument("--no_parallel", action="store_true",
                        help="Run pipeline stages sequentially instead of in parallel.")
    args = parser.parse_args()

    main(args.config, args.break_main, parallel=not args.no_parallel)
