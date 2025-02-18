import glob
import os

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
    os.makedirs(f"results/{model_name}", exist_ok=True)
    for i in range(len(generated_datasets)):
        similarities = []
        for j in range(len(generated_datasets)):
            generated1 = f"{model_dir}/{generated_datasets[i]}"
            generated2 = f"{model_dir}/{generated_datasets[j]}"
            gen_name1, gen_name2 = generated_datasets[i].strip('.tsv'), generated_datasets[j].strip('.tsv')

            #compairr_command = f"compairr -m -g -f {generated1} {generated2} --out results/{model_name}/{gen_name1}_{gen_name2}.txt -p results/{model_name}/{gen_name1}_{gen_name2}_pairs.tsv --differences 1 --indels"
            compairr_command = f"compairr -m -g -f {generated1} {generated2} --out results/{model_name}/{gen_name1}_{gen_name2}.txt -p results/{model_name}/{gen_name1}_{gen_name2}_pairs.tsv --log results/{model_name}/{gen_name1}_{gen_name2}_log.txt"
            try:
                os.system(compairr_command)
            except Exception as e:
                print(e)

            # pairs_df = pd.read_csv(f"results/{model_name}/{gen_name1}_{gen_name2}_pairs.tsv", sep='\t')
            # # get num unique sequences
            # similarities.append(num_unique_seqs)

            with open(f"results/{model_name}/{gen_name1}_{gen_name2}.txt", 'r') as file:
                similarity = file.read()
                similarities.append(similarity)

        similarities_matrix.append(similarities)

    return similarities_matrix


def plot_cluster_heatmap(similarities_matrix):
    sns.clustermap(similarities_matrix, annot=True, method='average', metric='euclidean')
    plt.title("Jaccard similarity")
    plt.tight_layout()
    plt.savefig('jaccard_similarity.png')


def run_cluster_analysis(model_dir, model_name):
    preprocess_files_for_compairr(model_dir, f"preprocessed_models/{model_name}")


def main():
    model_dir = "models"

    for model in ["PWM", "soNNia"]:
        run_cluster_analysis(f"{model_dir}/{model}", model)
        similarities_matrix = run_compairr(f"preprocessed_models/{model}", model)

        print(similarities_matrix)


if __name__ == '__main__':
    main()

