import glob
import os
from pathlib import Path
import pandas as pd

from gen_airr_bm.core.tuning_config import TuningConfig
from gen_airr_bm.utils.tuning_utils import validate_analyses_data, save_and_plot_tuning_results


def run_reduced_dim_tuning(tuning_config: TuningConfig) -> None:
    """ Runs parameter tuning by similarity in reduced dimensionality.
        Args:
            tuning_config: Configuration for the tuning, including paths and model names.
        Returns:
            None
    """
    print("Tuning model hyperparameters based on reduced dimensionality metrics...")

    validated_analyses_paths = validate_analyses_data(tuning_config, required_analyses=['reduced_dimensionality'])
    print(f"Validated analyses for tuning: {validated_analyses_paths}")

    os.makedirs(tuning_config.tuning_output_dir, exist_ok=True)
    summary_dfs = collect_analyses_results(tuning_config)
    summary_names = ['aminoacid', 'kmer', 'length']

    for analysis_name, summary_df in zip(summary_names, summary_dfs):
        save_and_plot_tuning_results(tuning_config, analysis_name, summary_df, tuning_config.tuning_output_dir,
                                     plot_title=f"JSD scores between reference and generated {analysis_name} distributions")


def collect_analyses_results(tuning_config: TuningConfig) -> tuple:
    """ Collects results from analyses for tuning purposes.
        Args:
            tuning_config: Configuration for the tuning, including paths and model names.
        Returns:
            tuple: Summary dataframes of the analyses results for sequence length, kmer frequency and amino acid
            frequency distributions.
    """
    root_output_dir = tuning_config.root_output_dir
    reference_data = tuning_config.reference_data
    analyses_dir = (Path(root_output_dir) / "analyses/reduced_dimensionality" / '_'.join(tuning_config.subfolder_name.split()) /
                    '_'.join(reference_data))

    aminoacid_files = glob.glob(os.path.join(analyses_dir, "aminoacid*.tsv"))
    amino_acid_dfs = []
    for f in aminoacid_files:
        if not f.endswith("_ref.tsv"):
            df = pd.read_csv(f, sep="\t")
            amino_acid_dfs.append(df)

    aa_data = pd.concat(amino_acid_dfs, ignore_index=True)
    aa_summary = aa_data.groupby(["Reference", "Model"], as_index=False)["Mean_Score"].mean()

    kmer_data = pd.read_csv(os.path.join(analyses_dir, "kmer_grouped.tsv"), sep="\t")
    kmer_summary = kmer_data.groupby(["Reference", "Model"], as_index=False)["Mean_Score"].mean()

    seq_len_data = pd.read_csv(os.path.join(analyses_dir, "length_grouped.tsv"), sep="\t")
    length_summary = seq_len_data.groupby(["Reference", "Model"], as_index=False)["Mean_Score"].mean()

    for df in [aa_summary, kmer_summary, length_summary]:
        df.rename(columns={"Mean_Score": "Score"}, inplace=True)

    return aa_summary, kmer_summary, length_summary
