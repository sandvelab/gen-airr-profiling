import os
import subprocess
import pandas as pd


def run_command(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    print(f"=== Starting: {cmd} ===")
    for line in process.stdout:
        print(f"[{cmd[:20]}...] {line.strip()}")
    process.wait()
    print(f"=== Finished: {cmd} ===\n")


def preprocess_files_for_compairr(sequences_dir, compairr_sequences_dir):
    datasets = os.listdir(sequences_dir)
    os.makedirs(f"{compairr_sequences_dir}", exist_ok=True)
    for dataset in datasets:
        data = pd.read_csv(f"{sequences_dir}/{dataset}", sep='\t')

        if 'duplicate_count' in data.columns:
            data.replace({'duplicate_count': {-1: 1}}, inplace=True)
        else:
            data['duplicate_count'] = 1

        data['sequence_id'] = [f"sequence_{i + 1}" for i in range(len(data))]

        data.to_csv(f"{compairr_sequences_dir}/{dataset}", sep='\t', index=False)


def run_compairr_existence(compairr_output_dir, search_sequences_path, reference_sequences_path, file_name, model_name=None):
    os.makedirs(compairr_output_dir, exist_ok=True)
    #TODO: For ImmunoHub execution we might need to use binaries instead of the command line
    #TODO: Maybe replace -u method ignoring illegal characters in sequences
    compairr_binary_path = "compairr-1.13.0-linux-x86_64"
    compairr_call = "./" + compairr_binary_path if os.path.exists(compairr_binary_path) else "compairr"

    compairr_command = (f"{compairr_call} -x {search_sequences_path} {reference_sequences_path} -d 1 -f -t 8 -u -g -o "
                        f"{compairr_output_dir}/{file_name}_overlap.tsv "
                        f"--log {compairr_output_dir}/{file_name}_log.txt --indels")

    # TODO: Add better support for PWM model
    # if model_name == "pwm":
    #     compairr_command += " -g"

    if os.path.exists(f"{compairr_output_dir}/{file_name}_overlap.tsv"):
        print(f"Compairr output already exists for {file_name}. Skipping execution.")
    else:
        run_command(compairr_command)


def run_compairr_cluster(compairr_output_dir, sequnces_path, file_name, model_name=None):
    os.makedirs(compairr_output_dir, exist_ok=True)
    # TODO: Maybe replace -u method ignoring illegal characters in sequences
    compairr_command = (f"compairr -c {sequnces_path} -o {compairr_output_dir}/{file_name}.tsv -g -d 1 -u "
                        f"--log {compairr_output_dir}/{file_name}_log.txt --indels")
    os.system(compairr_command)

    if os.path.exists(f"{compairr_output_dir}/{file_name}.tsv"):
        print(f"Compairr output already exists for {file_name}. Skipping execution.")
    else:
        run_command(compairr_command)


def deduplicate_and_merge_two_datasets(data1_path, data2_path, output_file_unique, output_file_concat):
    data1 = pd.read_csv(data1_path, sep='\t')
    data2 = pd.read_csv(data2_path, sep='\t')
    data1['sequence_id'] = [f"dataset_1_{i + 1}" for i in range(len(data1))]
    data2['sequence_id'] = [f"dataset_2_{i + 1}" for i in range(len(data2))]

    unique_sequences = pd.concat([data1, data2]).drop_duplicates(subset=['junction_aa'])
    unique_sequences.to_csv(output_file_unique, sep='\t', index=False)

    data1['repertoire_id'] = "dataset_1"
    data2['repertoire_id'] = "dataset_2"

    concat_data = pd.concat([data1, data2])
    concat_data.to_csv(output_file_concat, sep='\t', index=False)


def deduplicate_single_dataset(input_sequences_path, output_file_unique, n_unique_samples=None):
    data = pd.read_csv(input_sequences_path, sep='\t')
    data['sequence_id'] = [f"dataset_{i + 1}" for i in range(len(data))]

    #TODO: We need to decide if we want to include v,j genes in the deduplication
    unique_sequences = data.drop_duplicates(subset=["junction_aa"])

    if n_unique_samples is not None:
        if len(unique_sequences) >= n_unique_samples:
            unique_sequences = unique_sequences.sample(n=n_unique_samples, random_state=42)
        else:
            raise ValueError(f"Not enough unique sequences in {input_sequences_path} to sample {n_unique_samples} "
                             f"sequences.")

    unique_sequences.to_csv(output_file_unique, sep='\t', index=False)


def setup_directories(analysis_config, dataset_type):
    """Collect preprocessed directories for train/test sequences."""
    compairr_dir = f"{analysis_config.root_output_dir}/{dataset_type}_compairr_sequences"
    return compairr_dir, os.listdir(compairr_dir)
