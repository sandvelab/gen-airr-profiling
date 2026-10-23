import os
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from gen_airr_bm.analysis.analyse_reduced_dimensionality import (compute_divergence_scores,
                                                                compute_train_test_reference_scores,
                                                                run_reduced_dimensionality_analysis)
from gen_airr_bm.analysis.distribution.distribution_factory import get_distribution_strategy
from gen_airr_bm.constants.distribution_type import DistributionType
from gen_airr_bm.core.analysis_config import AnalysisConfig

REFERENCE_SEQUENCES = ["CASSLGQGF", "CASSPGTEAFF", "CASSQETQYF", "CASRDRGNTIYF", "CASSLAPGATNEKLFF"]
# Each model generates sequences that differ per subset, so that the subset scores within a repertoire differ
GENERATED_SEQUENCES = {
    "pwm": [["CASSLGQGF", "CASSLGQGF", "CASSPGTEAFF"], ["CASSQETQYF", "CASSLAPGATNEKLFF", "CASSLGQGF"]],
    "vae": [["CASRDRGNTIYF", "CASSPGTEAFF", "CASSQETQYF"], ["CASSLGQGF", "CASSPGTEAFF", "CASSQETQYF"]],
}


def write_sequence_file(path, sequences):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"junction_aa": sequences,
                  "sequence_id": [f"sequence_{i + 1}" for i in range(len(sequences))]}).to_csv(path, sep="\t",
                                                                                               index=False)


@pytest.fixture
def output_tree(tmp_path):
    """ Mocks the output of post-processing: two repertoires, two generated subsets per repertoire and two models. """
    root = str(tmp_path / "results")
    for dataset in ["rep_1", "rep_2"]:
        for split in ["train", "test"]:
            write_sequence_file(f"{root}/{split}_sequences/{dataset}.tsv", REFERENCE_SEQUENCES)
            write_sequence_file(f"{root}/{split}_compairr_sequences/{dataset}.tsv", REFERENCE_SEQUENCES)
        for model, subsets in GENERATED_SEQUENCES.items():
            for subset, sequences in enumerate(subsets):
                write_sequence_file(f"{root}/novel_generated_compairr_sequences_split/{model}/{dataset}_{subset}.tsv",
                                    sequences)
    return root


@pytest.fixture
def analysis_config(output_tree):
    return AnalysisConfig(
        analysis="reduced_dimensionality",
        model_names=["pwm", "vae"],
        analysis_output_dir=f"{output_tree}/analyses/reduced_dimensionality/all_models",
        root_output_dir=output_tree,
        default_model_name="humanTRB",
        reference_data=["train", "test"],
        n_subsets=2,
        subfolder_name="all models",
        receptor_type="TCR"
    )


def test_compute_divergence_scores_one_row_per_subset(analysis_config):
    strategy = get_distribution_strategy(DistributionType.KMER)

    scores_df = compute_divergence_scores(analysis_config, strategy, defaultdict(lambda: defaultdict(list)))

    # 2 models x 2 references x 2 repertoires x 2 subsets
    assert len(scores_df) == 16
    assert set(scores_df["metric"]) == {"all"}
    assert set(scores_df["dataset"]) == {"rep_1", "rep_2"}
    assert set(scores_df["subset"]) == {0, 1}


def test_compute_divergence_scores_aa_one_metric_per_length(analysis_config):
    strategy = get_distribution_strategy(DistributionType.AA)

    scores_df = compute_divergence_scores(analysis_config, strategy, defaultdict(lambda: defaultdict(list)))

    assert set(scores_df["metric"]) == set(range(10, 21))
    assert len(scores_df) == 16 * len(range(10, 21))


def test_compute_train_test_reference_scores(analysis_config):
    strategy = get_distribution_strategy(DistributionType.LENGTH)

    reference_scores = compute_train_test_reference_scores(analysis_config, strategy)

    # Train and test files are identical in the fixture, so every repertoire scores 0
    assert reference_scores["repertoire"].tolist() == ["rep_1", "rep_2"]
    assert reference_scores["score"].tolist() == pytest.approx([0.0, 0.0], abs=1e-12)


def test_compute_train_test_reference_scores_without_both_references(analysis_config):
    analysis_config.reference_data = ["test"]
    strategy = get_distribution_strategy(DistributionType.LENGTH)

    assert compute_train_test_reference_scores(analysis_config, strategy) is None


@pytest.mark.parametrize("distribution_type, file_stems", [
    (DistributionType.LENGTH, ["length"]),
    (DistributionType.KMER, ["kmer"]),
    (DistributionType.AA, [f"aminoacid_{length}" for length in range(10, 21)]),
])
def test_run_reduced_dimensionality_analysis_writes_outputs(analysis_config, distribution_type, file_stems):
    run_reduced_dimensionality_analysis(analysis_config, distribution_type)

    output_dir = analysis_config.analysis_output_dir
    assert os.path.exists(f"{output_dir}/{distribution_type.value.replace(' ', '')}_jsd_scores.tsv")
    for file_stem in file_stems:
        assert os.path.exists(f"{output_dir}/train_test/{file_stem}_grouped.png")
        for table in ["repertoire_scores", "summary", "vs_train_test", "all_pairs"]:
            assert os.path.exists(f"{output_dir}/statistics/{file_stem}_{table}.tsv")
        assert os.path.exists(f"{output_dir}/statistics/{file_stem}_vs_train_test.png")
    vs_train_test = pd.read_csv(f"{output_dir}/statistics/{file_stems[0]}_vs_train_test.tsv", sep="\t")
    assert "p_value" not in vs_train_test.columns


def test_run_reduced_dimensionality_analysis_averages_subsets_first(analysis_config):
    run_reduced_dimensionality_analysis(analysis_config, DistributionType.KMER)

    output_dir = analysis_config.analysis_output_dir
    scores_df = pd.read_csv(f"{output_dir}/kmer_jsd_scores.tsv", sep="\t")
    plotted = pd.read_csv(f"{output_dir}/train_test/kmer_grouped.tsv", sep="\t")
    for model in ["pwm", "vae"]:
        model_scores = scores_df[(scores_df["model"] == model) & (scores_df["reference"] == "test")]
        repertoire_means = model_scores.groupby("dataset")["jsd"].mean()
        row = plotted[(plotted["Model"] == model) & (plotted["Reference"] == "test")].iloc[0]
        assert row["Mean_Score"] == pytest.approx(repertoire_means.mean())
        # The error bars are the SD across the repertoire means, not across all subsets
        assert row["Std_Dev"] == pytest.approx(np.std(repertoire_means, ddof=1))


def test_run_reduced_dimensionality_analysis_with_statistical_tests(analysis_config):
    analysis_config.statistical_tests = True
    analysis_config.donor_pattern = "^(rep_\\d+)$"

    run_reduced_dimensionality_analysis(analysis_config, DistributionType.LENGTH)

    vs_train_test = pd.read_csv(f"{analysis_config.analysis_output_dir}/statistics/length_vs_train_test.tsv",
                                sep="\t")
    assert vs_train_test["first"].tolist() == ["pwm", "vae"]
    assert {"ci_low", "ci_high", "p_value", "p_adjusted"} <= set(vs_train_test.columns)


def test_run_reduced_dimensionality_analysis_with_tests_requires_one_repertoire_per_donor(analysis_config):
    analysis_config.statistical_tests = True
    analysis_config.donor_pattern = "^(rep)_"

    with pytest.raises(ValueError, match="one repertoire per donor"):
        run_reduced_dimensionality_analysis(analysis_config, DistributionType.LENGTH)


def test_run_reduced_dimensionality_analysis_with_contrasts(analysis_config):
    analysis_config.contrasts = [["vae", "pwm"]]

    run_reduced_dimensionality_analysis(analysis_config, DistributionType.KMER)

    prefix = f"{analysis_config.analysis_output_dir}/statistics/kmer"
    contrasts = pd.read_csv(f"{prefix}_contrasts.tsv", sep="\t")
    assert contrasts[["first", "second", "n_repertoires"]].values.tolist() == [["vae", "pwm", 2]]
    assert os.path.exists(f"{prefix}_contrasts.png")


def test_run_reduced_dimensionality_analysis_corrects_aa_across_lengths(analysis_config):
    analysis_config.statistical_tests = True
    analysis_config.donor_pattern = "^(rep_\\d+)$"

    run_reduced_dimensionality_analysis(analysis_config, DistributionType.AA)

    vs_train_test = pd.concat([pd.read_csv(f"{analysis_config.analysis_output_dir}/statistics/aminoacid_{length}_"
                                           f"vs_train_test.tsv", sep="\t") for length in range(10, 21)])
    tested = vs_train_test.dropna(subset=["p_value"])
    assert len(tested) > 0
    # Bonferroni over the comparisons of all lengths together, not over the comparisons of each length
    assert tested["p_adjusted"].to_numpy() == pytest.approx(np.minimum(tested["p_value"].to_numpy() * len(tested), 1))
