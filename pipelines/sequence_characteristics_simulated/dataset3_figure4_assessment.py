import glob
import os
from sys import prefix

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def preprocess_files_for_compairr(model_dir, preprocessed_model_dir):
    generated_datasets = os.listdir(model_dir)
    os.makedirs(f"{preprocessed_model_dir}", exist_ok=True)
    for dataset in generated_datasets:
        file_name = glob.glob(f"{model_dir}/{dataset}/gen_model/exported_gen_dataset/*.tsv")[0]
        data = pd.read_csv(file_name, sep='\t')
        data['duplicate_count'].replace(-1, 1, inplace=True)
        data.to_csv(f"{preprocessed_model_dir}/{dataset}.tsv", sep='\t', index=False)


def run_compairr(model_dir, model_name):
    generated_datasets = os.listdir(model_dir)
    similarities_matrix = []
    dataset_names = [dataset.strip('.tsv') for dataset in generated_datasets]
    os.makedirs(f"results/{model_name}", exist_ok=True)

    for i in range(len(generated_datasets)):
        similarities = []
        for j in range(len(generated_datasets)):
            generated1 = f"{model_dir}/{generated_datasets[i]}"
            generated2 = f"{model_dir}/{generated_datasets[j]}"
            gen_name1, gen_name2 = generated_datasets[i].strip('.tsv'), generated_datasets[j].strip('.tsv')

            unique_sequences_filename = f"results/{model_name}/{gen_name1}_{gen_name2}_unique.tsv"
            combined_sequences_filename = f"results/{model_name}/{gen_name1}_{gen_name2}_combined.tsv"
            write_unique_and_combined_sequence_datasets_to_file(generated1, generated2, unique_sequences_filename, combined_sequences_filename)

            compairr_command = f"./compairr-1.13.0-linux-x86_64 -x {unique_sequences_filename} {combined_sequences_filename} -d 1 -f -t 8 -o results/{model_name}/{gen_name1}_{gen_name2}_rep_overlap.tsv -p results/{model_name}/{gen_name1}_{gen_name2}_rep_overlap_pairs.tsv --log results/{model_name}/{gen_name1}_{gen_name2}_log.txt --indels"

            if model_name == "PWM":
                compairr_command += " -g"
            #compairr_command = f"compairr -m -g -f {generated1} {generated2} --out results/{model_name}/{gen_name1}_{gen_name2}.txt -p results/{model_name}/{gen_name1}_{gen_name2}_pairs.tsv --log results/{model_name}/{gen_name1}_{gen_name2}_log.txt --differences 1 --indels"
            #compairr_command = f"compairr -m -g -f {generated1} {generated2} --out results/{model_name}/{gen_name1}_{gen_name2}.txt -p results/{model_name}/{gen_name1}_{gen_name2}_pairs.tsv --log results/{model_name}/{gen_name1}_{gen_name2}_log.txt"
            try:
                os.system(compairr_command)
            except Exception as e:
                print(e)

            rep_overlap_df = pd.read_csv(f"results/{model_name}/{gen_name1}_{gen_name2}_rep_overlap.tsv", sep='\t')
            count_nonzero_rows = rep_overlap_df[(rep_overlap_df['gen_model_1'] != 0) & (rep_overlap_df['gen_model_2'] != 0)].shape[0]

            union = pd.read_csv(unique_sequences_filename, sep='\t').shape[0]
            jaccard_similarity = count_nonzero_rows / union
            similarities.append(jaccard_similarity)

        similarities_matrix.append(similarities)

    return similarities_matrix, dataset_names


def write_unique_and_combined_sequence_datasets_to_file(generated1, generated2, output_file_unique, output_file_combined):
    generated1_data = pd.read_csv(generated1, sep='\t')
    generated2_data = pd.read_csv(generated2, sep='\t')
    generated1_data['sequence_id'] = [f"gen_model_1_{i + 1}"for i in range(len(generated1_data))]
    generated2_data['sequence_id'] = [f"gen_model_2_{i + 1}"for i in range(len(generated2_data))]

    unique_sequences = pd.concat([generated1_data, generated2_data]).drop_duplicates(subset=['junction_aa'])
    unique_sequences.to_csv(output_file_unique, sep='\t', index=False)

    generated1_data['repertoire_id'] = "gen_model_1"
    generated2_data['repertoire_id'] = "gen_model_2"
    combined_data = pd.concat([generated1_data, generated2_data])
    combined_data.to_csv(output_file_combined, sep='\t', index=False)


def plot_cluster_heatmap(output_dir, similarities_matrix, model_name):
    sns.clustermap(similarities_matrix, annot=True, method='average', metric='euclidean')
    plt.title(f"Jaccard similarity: {model_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_heatmap.png")


def run_cluster_analysis(models_dir, model_names):

    for model in model_names:
        model_dir = f"{models_dir}/{model}"
        preprocess_files_for_compairr(model_dir, f"preprocessed_models/{model}")
        similarities_matrix, dataset_names = run_compairr(f"preprocessed_models/{model}", model)

        similarities_df = pd.DataFrame(similarities_matrix, index=dataset_names,
                                             columns=dataset_names)

        plot_cluster_heatmap(f"results/{model}", similarities_df, model)


def main():
    model_dir = "models"
    model_names = ["PWM", "soNNia"]

    run_cluster_analysis(model_dir, model_names)


if __name__ == '__main__':
    main()

