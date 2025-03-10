import os
import pandas as pd


def preprocess_files_for_compairr(sequences_dir, compairr_sequences_dir):
    datasets = os.listdir(sequences_dir)
    os.makedirs(f"{compairr_sequences_dir}", exist_ok=True)
    for dataset in datasets:
        data = pd.read_csv(f"{sequences_dir}/{dataset}", sep='\t')

        if 'duplicate_count' in data.columns:
            data.replace({'duplicate_count': {-1: 1}}, inplace=True)
        else:
            data['duplicate_count'] = 1

        data.to_csv(f"{compairr_sequences_dir}/{dataset}", sep='\t', index=False)


def run_compairr(compairr_output_dir, unique_sequences_path, concat_sequences_path, file_name, model_name):
    os.makedirs(compairr_output_dir, exist_ok=True)
    #TODO: For ImmunoHub execution we might need to use binaries instead of the command line
    #TODO: Maybe replace -u method ignoring illegal characters in sequences
    compairr_command = (f"compairr -x {unique_sequences_path} {concat_sequences_path} -d 1 -f -t 8 -u -o "
                        f"{compairr_output_dir}/{file_name}_overlap.tsv -p {compairr_output_dir}/{file_name}_pairs.tsv "
                        f"--log {compairr_output_dir}/{file_name}_log.txt --indels")

    # TODO: Add better support for PWM model
    if model_name == "pwm":
        compairr_command += " -g"
    os.system(compairr_command)


def process_and_save_sequences(data1_path, data2_path, output_file_unique, output_file_concat):
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