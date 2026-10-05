import os

import numpy as np
import pandas as pd
import pytest

from gen_airr_bm.analysis.analyse_gene_usage import (compute_gene_usage_frequencies, compute_gene_usage_scores,
                                                    compute_jsd, compute_train_test_reference_score,
                                                    compute_usage_distribution, get_gene_family,
                                                    model_generates_gene_calls, normalise_gene_call, read_gene_calls,
                                                    run_gene_usage_analysis, aggregate_scores_by_reference)
from gen_airr_bm.core.analysis_config import AnalysisConfig

# The models generate gene names without the allele
V_GENERATED = ["TRBV6-1", "TRBV6-5", "TRBV19", "TRBV20"]
J_GENERATED = ["TRBJ1-1", "TRBJ2-3"]
# The experimental data keeps the allele on most calls
V_REFERENCE = ["TRBV6-1*01", "TRBV6-5", "TRBV19*01", "TRBV20*02"]
J_REFERENCE = ["TRBJ1-1*01", "TRBJ2-3*01"]


def write_sequence_file(path, v_calls, j_calls, with_gene_calls=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for i in range(12):
        rows.append({"locus": "TRB",
                     "v_call": v_calls[i % len(v_calls)] if with_gene_calls else "",
                     "j_call": j_calls[i % len(j_calls)] if with_gene_calls else "",
                     "junction_aa": f"CASSX{i}F",
                     "duplicate_count": 1,
                     "sequence_id": f"sequence_{i + 1}"})
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


@pytest.fixture
def output_tree(tmp_path):
    """ Mocks the output of post-processing: two repertoires, two generated subsets per repertoire and three models.
    The references use Adaptive gene names, sonnia uses IMGT names, vae uses Adaptive names and pwm has no gene calls.
    """
    root = str(tmp_path / "results")
    for dataset in ["rep_1", "rep_2"]:
        for split in ["train", "test"]:
            write_sequence_file(f"{root}/{split}_compairr_sequences/{dataset}.tsv", V_REFERENCE, J_REFERENCE)
    for model, (v_calls, j_calls, with_gene_calls) in {"sonnia": (V_GENERATED, J_GENERATED, True),
                                                      "vae": (V_GENERATED, J_GENERATED, True),
                                                      "pwm": (V_GENERATED, J_GENERATED, False)}.items():
        for dataset in ["rep_1", "rep_2"]:
            for subset in range(2):
                write_sequence_file(
                    f"{root}/novel_generated_compairr_sequences_split/{model}/{dataset}_{subset}.tsv",
                    v_calls, j_calls, with_gene_calls)
    return root


@pytest.fixture
def analysis_config(output_tree):
    return AnalysisConfig(
        analysis="gene_usage",
        model_names=["pwm", "vae", "sonnia"],
        analysis_output_dir=f"{output_tree}/analyses/gene_usage/all_models",
        root_output_dir=output_tree,
        default_model_name="humanTRB",
        reference_data=["train", "test"],
        n_subsets=2,
        subfolder_name="all models",
        receptor_type="TCR"
    )


@pytest.mark.parametrize("gene_call, expected", [
    ("TRBV5-5*01", "TRBV5-5"),
    ("TRBV6-1*02", "TRBV6-1"),
    ("TRBV20", "TRBV20"),
    ("TRBV6-1,TRBV6-5", "TRBV6-1"),
    ("TRBV6-1*01,TRBV6-5*01", "TRBV6-1"),
    ("", None),
    ("   ", None),
    (np.nan, None),
])
def test_normalise_gene_call(gene_call, expected):
    assert normalise_gene_call(gene_call) == expected


@pytest.mark.parametrize("gene_name, expected", [
    ("TRBV6-1", "TRBV6"),
    ("TRBV20", "TRBV20"),
    ("TRBV20/OR9-2", "TRBV20"),
    ("TRBJ2-3", "TRBJ2"),
])
def test_get_gene_family(gene_name, expected):
    assert get_gene_family(gene_name) == expected


def test_read_gene_calls_drops_alleles(output_tree):
    gene_calls = read_gene_calls(f"{output_tree}/train_compairr_sequences/rep_1.tsv")

    assert set(gene_calls["v_call"]) == set(V_GENERATED)
    assert set(gene_calls["j_call"]) == set(J_GENERATED)


def test_read_gene_calls_drops_sequences_without_calls(output_tree):
    gene_calls = read_gene_calls(f"{output_tree}/novel_generated_compairr_sequences_split/pwm/rep_1_0.tsv")

    assert len(gene_calls) == 0


def test_read_gene_calls_without_gene_columns(tmp_path):
    file_path = str(tmp_path / "no_genes.tsv")
    pd.DataFrame({"junction_aa": ["CASSF"]}).to_csv(file_path, sep="\t", index=False)

    assert len(read_gene_calls(file_path)) == 0


def test_compute_usage_distribution_gene_and_family():
    gene_calls = pd.DataFrame({"v_call": ["TRBV6-1", "TRBV6-5", "TRBV19", "TRBV19"],
                               "j_call": ["TRBJ1-1", "TRBJ1-1", "TRBJ2-3", "TRBJ2-3"]})

    gene_distribution = compute_usage_distribution(gene_calls, ["v_call"], "gene")
    family_distribution = compute_usage_distribution(gene_calls, ["v_call"], "family")
    pairing_distribution = compute_usage_distribution(gene_calls, ["v_call", "j_call"], "gene")

    assert gene_distribution == {"TRBV6-1": 0.25, "TRBV6-5": 0.25, "TRBV19": 0.5}
    assert family_distribution == {"TRBV6": 0.5, "TRBV19": 0.5}
    assert pairing_distribution == {"TRBV6-1|TRBJ1-1": 0.25, "TRBV6-5|TRBJ1-1": 0.25, "TRBV19|TRBJ2-3": 0.5}


def test_compute_usage_distribution_empty():
    assert compute_usage_distribution(pd.DataFrame(columns=["v_call", "j_call"]), ["v_call"], "gene") == {}


def test_compute_jsd_identical_distributions():
    distribution = {"TRBV6-1": 0.4, "TRBV19": 0.6}

    assert compute_jsd(distribution, distribution) == pytest.approx(0.0, abs=1e-12)


def test_compute_jsd_disjoint_distributions():
    assert compute_jsd({"TRBV6-1": 1.0}, {"TRBV19": 1.0}) == pytest.approx(1.0, abs=1e-6)


def test_compute_jsd_empty_distribution():
    assert np.isnan(compute_jsd({}, {"TRBV19": 1.0}))


def test_model_generates_gene_calls(analysis_config):
    assert model_generates_gene_calls(analysis_config, "sonnia")
    assert model_generates_gene_calls(analysis_config, "vae")
    assert not model_generates_gene_calls(analysis_config, "pwm")


def test_compute_gene_usage_scores_matches_gene_names_across_allele_resolutions(analysis_config):
    scores_df = compute_gene_usage_scores(analysis_config, ["sonnia", "vae"], {})

    # 2 models x 2 references x 2 repertoires x 2 subsets x 5 metrics
    assert len(scores_df) == 80
    assert set(scores_df["subset"]) == {0, 1}
    # The generated gene usage is identical to the reference usage, only written without the allele
    assert scores_df["jsd"].max() == pytest.approx(0.0, abs=1e-12)


def test_aggregate_scores_by_reference():
    scores_df = pd.DataFrame([
        {"metric": "v gene", "model": "vae", "reference": "train", "dataset": "rep_1", "subset": 0, "jsd": 0.1},
        {"metric": "v gene", "model": "vae", "reference": "train", "dataset": "rep_1", "subset": 1, "jsd": 0.3},
        {"metric": "v gene", "model": "vae", "reference": "train", "dataset": "rep_2", "subset": 0, "jsd": 0.6},
        {"metric": "j gene", "model": "vae", "reference": "train", "dataset": "rep_1", "subset": 0, "jsd": 0.9},
    ])

    mean_scores_by_ref, std_scores_by_ref = aggregate_scores_by_reference(scores_df, "v gene")

    # The two subsets of rep_1 are averaged first, so the repertoire scores are 0.2 and 0.6
    assert mean_scores_by_ref == {"train": {"vae": pytest.approx(0.4)}}
    assert std_scores_by_ref == {"train": {"vae": pytest.approx(0.2)}}


def test_compute_train_test_reference_score(analysis_config):
    reference_score = compute_train_test_reference_score(analysis_config, "v gene", {})

    assert reference_score == pytest.approx(0.0, abs=1e-12)


def test_compute_train_test_reference_score_without_both_references(analysis_config):
    analysis_config.reference_data = ["test"]

    assert compute_train_test_reference_score(analysis_config, "v gene", {}) is None


def test_compute_gene_usage_frequencies(analysis_config):
    frequencies_df = compute_gene_usage_frequencies(analysis_config, ["sonnia"], {})

    v_gene_frequencies = frequencies_df[(frequencies_df["metric"] == "v gene")
                                       & (frequencies_df["dataset"] == "rep_1")]
    assert set(v_gene_frequencies["source"]) == {"sonnia", "train", "test"}
    assert set(v_gene_frequencies[v_gene_frequencies["source"] == "sonnia"]["gene"]) == set(V_GENERATED)
    for source in ["sonnia", "train", "test"]:
        assert v_gene_frequencies[v_gene_frequencies["source"] == source]["frequency"].sum() == pytest.approx(1.0)


def test_run_gene_usage_analysis_writes_outputs(analysis_config):
    run_gene_usage_analysis(analysis_config)

    output_dir = analysis_config.analysis_output_dir
    assert os.path.exists(f"{output_dir}/gene_usage_jsd_scores.tsv")
    assert os.path.exists(f"{output_dir}/gene_usage_frequencies.tsv")
    assert os.path.exists(f"{output_dir}/skipped_models.txt")
    for metric_name in ["v_gene", "j_gene", "vj_pairing", "v_family", "j_family"]:
        assert os.path.exists(f"{output_dir}/train_test/gene_usage_{metric_name}_grouped.tsv")
        assert os.path.exists(f"{output_dir}/train_test/gene_usage_{metric_name}_grouped.png")
        assert os.path.exists(f"{output_dir}/gene_usage_frequencies_{metric_name}.png")

    # pwm does not generate gene calls, so it is skipped and not part of the scores
    with open(f"{output_dir}/skipped_models.txt") as file:
        assert "pwm" in file.read()
    scores_df = pd.read_csv(f"{output_dir}/gene_usage_jsd_scores.tsv", sep="\t")
    assert set(scores_df["model"]) == {"vae", "sonnia"}


def test_run_gene_usage_analysis_without_gene_generating_models(analysis_config):
    analysis_config.model_names = ["pwm"]

    run_gene_usage_analysis(analysis_config)

    output_dir = analysis_config.analysis_output_dir
    assert os.path.exists(f"{output_dir}/skipped_models.txt")
    assert not os.path.exists(f"{output_dir}/gene_usage_jsd_scores.tsv")
