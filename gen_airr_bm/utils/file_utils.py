import os
import re
from collections import defaultdict
from gen_airr_bm.core.analysis_config import AnalysisConfig


def get_sequence_files(analysis_config: AnalysisConfig, model: str, reference_data: str):
    comparison_files_dir = defaultdict(set)

    ref_dir = f"{analysis_config.root_output_dir}/{reference_data}_compairr_sequences"
    gen_dir = f"{analysis_config.root_output_dir}/generated_compairr_sequences_split/{model}"

    ref_files = set(os.listdir(ref_dir))
    gen_files = set(os.listdir(gen_dir))

    for file in ref_files:
        base_name = os.path.splitext(file)[0]
        filtered_gen_files = sorted([f for f in gen_files if base_name in f],
                                    key=lambda x: int(re.search(r'_(\d+)\.tsv$', x).group(1)))

        if len(filtered_gen_files) < analysis_config.n_subsets:
            raise ValueError(
                f"Not enough generated files for {base_name} in {gen_dir}. Expected {analysis_config.n_subsets}, found {len(filtered_gen_files)}.")

        comparison_files_dir[os.path.join(ref_dir, file)] = set(
            [os.path.join(gen_dir, f) for f in filtered_gen_files[:analysis_config.n_subsets]])

    return comparison_files_dir


def get_reference_files(analysis_config: AnalysisConfig):
    comparison_files_dir = []

    train_dir = f"{analysis_config.root_output_dir}/train_compairr_sequences"
    test_dir = f"{analysis_config.root_output_dir}/test_compairr_sequences"

    train_files = set(os.listdir(train_dir))

    for file in train_files:
        base_name = os.path.splitext(file)[0]
        comparison_files_dir.append((os.path.join(train_dir, file), [os.path.join(test_dir, base_name + ".tsv")]))

    return comparison_files_dir
