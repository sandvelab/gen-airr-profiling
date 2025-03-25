import os

from gen_airr_bm.core.analysis_config import AnalysisConfig
from gen_airr_bm.utils.compairr_utils import setup_directories


def run_connectivity_analysis(analysis_config: AnalysisConfig):
    print("Running connectivity analysis")
    output_dir = analysis_config.analysis_output_dir
    os.makedirs(output_dir, exist_ok=True)

    compairr_train_dir, train_datasets = setup_directories(analysis_config, "train")
    compairr_test_dir, test_datasets = setup_directories(analysis_config, "test")

    for model_name in analysis_config.model_names:
        compairr_model_dir = f"{analysis_config.root_output_dir}/generated_compairr_sequences/{model_name}"
        generated_datasets = os.listdir(compairr_model_dir)

        for generated_dataset in generated_datasets:
