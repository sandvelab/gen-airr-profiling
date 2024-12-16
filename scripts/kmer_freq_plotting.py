import os
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from scripts.utils import get_shared_region_type


def get_kmer_counts(sequences, k):
    kmer_counts = {}
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
    return kmer_counts


def find_significantly_different_kmers(dataset1, name1, dataset2, name2, k=3, kmer_count_threshold=5):
    dataset1_kmer_counts = get_kmer_counts(dataset1, k)
    dataset2_kmer_counts = get_kmer_counts(dataset2, k)
    all_kmers_comparison = set(dataset1_kmer_counts.keys()).union(set(dataset2_kmer_counts.keys()))

    data_comparison = []
    for kmer in all_kmers_comparison:
        dataset1_count = dataset1_kmer_counts.get(kmer, 0)
        dataset2_count = dataset2_kmer_counts.get(kmer, 0)
        if dataset1_count + dataset2_count > kmer_count_threshold:
            dataset1_freq = dataset1_count / sum(dataset1_kmer_counts.values())
            dataset2_freq = dataset2_count / sum(dataset2_kmer_counts.values())
            data_comparison.append([kmer, dataset1_count, dataset2_count, dataset1_freq, dataset2_freq])
    kmer_comparison_df = pd.DataFrame(data_comparison,
                                      columns=['kmer', name1 + '_count', name2 + '_count', name1 + '_freq',
                                               name2 + '_freq'])

    if kmer_comparison_df.empty:
        return kmer_comparison_df, None

    p_values = []
    for _, row in kmer_comparison_df.iterrows():
        dataset1_count = row[name1 + '_count']
        dataset2_count = row[name2 + '_count']
        total_dataset1 = sum(kmer_comparison_df[name1 + '_count'])
        total_dataset2 = sum(kmer_comparison_df[name2 + '_count'])

        contingency_table = [[dataset1_count, total_dataset1 - dataset1_count],
                             [dataset2_count, total_dataset2 - dataset2_count]]

        _, p_value = fisher_exact(contingency_table)
        p_values.append(p_value)

    adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]

    kmer_comparison_df['p_value'] = p_values
    kmer_comparison_df['adjusted_p_value'] = adjusted_p_values

    significant_kmers = kmer_comparison_df[kmer_comparison_df['adjusted_p_value'] < 0.05]

    return kmer_comparison_df, significant_kmers


def pseudo_log_transform(x, threshold=1e-3):
    return np.sign(x) * np.log1p(np.abs(x / threshold))


def plot_kmers_distribution(kmer_comparison_df, name1, name2, output_dir, k=3):
    kmer_comparison_df[f"pseudo_{name1}_freq"] = pseudo_log_transform(kmer_comparison_df[f"{name1}_freq"])
    kmer_comparison_df[f"pseudo_{name2}_freq"] = pseudo_log_transform(kmer_comparison_df[f"{name2}_freq"])
    kmer_comparison_df['significance'] = np.where(kmer_comparison_df['adjusted_p_value'] < 0.05, True, False)

    # TO DO: investigate repeat regions in significantly different k-mers
    significance_ranking_20 = kmer_comparison_df.sort_values(by='adjusted_p_value', ascending=True).head(20)
    significance_labels = significance_ranking_20[significance_ranking_20['adjusted_p_value'] < 0.05]['kmer']
    kmer_comparison_df['label'] = kmer_comparison_df['kmer']
    kmer_comparison_df['label'] = np.where(kmer_comparison_df['kmer'].isin(significance_labels), kmer_comparison_df['kmer'], "")

    # Manually adding SLG k-mer label (temporary)
    if 'soNNia' in [name1, name2]:
        kmer_comparison_df['label'] = np.where(kmer_comparison_df['kmer'].isin(['SLG']), "SLG", kmer_comparison_df['label'])

    significant_count = kmer_comparison_df['significance'].value_counts().get(True, 0)

    fig = px.scatter(
        kmer_comparison_df,
        x=f"pseudo_{name1}_freq",
        y=f"pseudo_{name2}_freq",
        hover_name="kmer",
        color="significance",
        color_discrete_map={True: "red", False: "blue"},
        labels={f"pseudo_{name1}_freq": f"Pseudo-log {name1} frequency",
                f"pseudo_{name2}_freq": f"Pseudo-log {name2} frequency"},
        title=f"Scatter plot of {name1} vs {name2} {k}-mer frequencies ({significant_count} significantly different {k}-mers, {len(kmer_comparison_df)} tests)",
        width=1000,
        height=1000,
        opacity=0.3,
        text="label"
    )

    fig.update_layout(
        xaxis_title=f"{name1} frequency (pseudo-log scale)",
        yaxis_title=f"{name2} frequency (pseudo-log scale)"
    )
    fig.update_traces(textposition="bottom right")

    fig.write_html(output_dir + "/kmer_comparison.html")


def count_amino_acid_repeats(kmer_counts):
    repeat_count = 0
    for kmer, count in kmer_counts.items():
        for i in range(len(kmer) - 1):
            if kmer[i] == kmer[i + 1]:
                repeat_count += count
                break
    return repeat_count


def compute_kmer_repeats(dataset_sequences, overrepresented_kmers, k):
    kmer_counts = get_kmer_counts(dataset_sequences, k=k)
    repeats_all = count_amino_acid_repeats(kmer_counts)
    overrepresented_kmer_counts = {kmer: kmer_counts[kmer] for kmer in overrepresented_kmers['kmer']}
    repeats_overrepresented = count_amino_acid_repeats(overrepresented_kmer_counts)
    return repeats_all, repeats_overrepresented


def kmer_repeat_regions_analysis(dataset1_sequences, name1, dataset2_sequences, name2, k, significant_kmers, output_dir):
    overrepresented_kmers_dataset1 = significant_kmers[
        significant_kmers[f'{name1}_freq'] > significant_kmers[f'{name2}_freq']]
    overrepresented_kmers_dataset2 = significant_kmers[
        significant_kmers[f'{name2}_freq'] > significant_kmers[f'{name1}_freq']]

    repeats_all_dataset1, repeats_overrepresented_dataset1 = compute_kmer_repeats(dataset1_sequences,
                                                                                  overrepresented_kmers_dataset1, k)
    repeats_all_dataset2, repeats_overrepresented_dataset2 = compute_kmer_repeats(dataset2_sequences,
                                                                                  overrepresented_kmers_dataset2, k)

    repeat_counts = pd.DataFrame({
        "repeats_all_kmers": [repeats_all_dataset1, repeats_all_dataset2],
        "repeats_overrepresented_kmers": [repeats_overrepresented_dataset1, repeats_overrepresented_dataset2]
    }, index=[name1, name2])

    repeat_counts.to_csv(f"{output_dir}/repeat_counts.tsv", sep="\t")

    # run fisher's exact test on repeat counts
    contingency_table = [[repeats_overrepresented_dataset1, repeats_all_dataset1],
                         [repeats_overrepresented_dataset2, repeats_all_dataset2]]
    _, p_value = fisher_exact(contingency_table)
    with open(output_dir + "/repeat_counts_fisher_exact_test_pval.txt", "w") as f:
        f.write(f"Contingency table:\n{contingency_table}\n")
        f.write(f"P-value: {p_value}\n")


def run_kmer_analysis(dataset1, name1, dataset2, name2, output_dir, k=3, kmer_count_threshold=5):

    os.makedirs(str(output_dir), exist_ok=True)

    dataset1_df = pd.read_csv(dataset1, sep="\t")
    dataset2_df = pd.read_csv(dataset2, sep="\t")

    shared_region_type = get_shared_region_type(dataset1_df, dataset2_df)
    dataset1_sequences = dataset1_df[shared_region_type].tolist()
    dataset2_sequences = dataset2_df[shared_region_type].tolist()

    kmer_comparison_df, significant_kmers = find_significantly_different_kmers(dataset1_sequences, name1,
                                                                               dataset2_sequences, name2,
                                                                               k, kmer_count_threshold)
    kmer_repeat_regions_analysis(dataset1_sequences, name1, dataset2_sequences, name2, k, significant_kmers, str(output_dir))

    plot_kmers_distribution(kmer_comparison_df, name1, name2, str(output_dir), k)


def main():
    dataset1 = "../results/dataset1/models/VAE/VAE_dataset1_0/gen_model/exported_gen_dataset/synthetic_my_vae_model_dataset.tsv"
    dataset2 = "../results/dataset1/simulations/train/simulation_0/dataset/simulated_dataset.tsv"
    output_dir = "output/"
    name1 = "VAE"
    name2 = "simulated"
    k = 3
    kmer_count_threshold = 5

    run_kmer_analysis(dataset1, name1, dataset2, name2, output_dir, k, kmer_count_threshold)

if __name__ == "__main__":
    main()
