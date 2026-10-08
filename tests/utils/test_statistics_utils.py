import itertools
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from gen_airr_bm.core.main_config import MainConfig
from gen_airr_bm.utils import statistics_utils
from gen_airr_bm.utils.statistics_utils import (DESCRIPTIVE_COMPARISON_COLUMNS, TRAIN_TEST_REFERENCE,
                                                aggregate_repertoire_scores, check_one_repertoire_per_donor,
                                                compare_paired, describe_error_bars, get_donor, is_underpowered,
                                                min_possible_p, run_repertoire_statistics, sign_flip_test,
                                                summarise_repertoire_scores, t_confidence_interval,
                                                validate_contrasts)


def make_repertoire_scores(scores_by_source: dict, donors: list = None) -> pd.DataFrame:
    """ One repertoire per score, rep_0, rep_1, ...; by default every repertoire is its own donor. """
    return pd.DataFrame([{"source": source, "repertoire": f"rep_{i}",
                          "donor": donors[i] if donors else f"rep_{i}", "score": score}
                         for source, scores in scores_by_source.items() for i, score in enumerate(scores)])


@pytest.mark.parametrize("repertoire, donor_pattern, expected", [
    ("6271_pLN_CD8", "^([^_]+)_", "6271"),
    ("P00021_male_neg", "^([^_]+)_", "P00021"),
    ("6271_pLN_CD8", "^\\d+", "6271"),
    ("6271_pLN_CD8", None, "6271_pLN_CD8"),
])
def test_get_donor(repertoire, donor_pattern, expected):
    assert get_donor(repertoire, donor_pattern) == expected


def test_get_donor_without_match():
    with pytest.raises(ValueError, match="does not match"):
        get_donor("rep_1", "^(\\d+)_")


def test_aggregate_repertoire_scores_averages_subsets():
    scores_df = pd.DataFrame([
        {"model": "vae", "dataset": "6271_pLN_CD8", "subset": 0, "jsd": 0.1},
        {"model": "vae", "dataset": "6271_pLN_CD8", "subset": 1, "jsd": 0.3},
        {"model": "vae", "dataset": "6271_pLN_Treg", "subset": 0, "jsd": 0.6},
    ])

    repertoire_scores = aggregate_repertoire_scores(scores_df, "jsd", "^([^_]+)_")

    assert repertoire_scores.to_dict("records") == [
        {"source": "vae", "repertoire": "6271_pLN_CD8", "donor": "6271", "score": pytest.approx(0.2), "n_subsets": 2},
        {"source": "vae", "repertoire": "6271_pLN_Treg", "donor": "6271", "score": pytest.approx(0.6),
         "n_subsets": 1}]


def test_check_one_repertoire_per_donor():
    check_one_repertoire_per_donor(make_repertoire_scores({"a": [0.1, 0.2]}))
    with pytest.raises(ValueError, match="one repertoire per donor"):
        check_one_repertoire_per_donor(make_repertoire_scores({"a": [0.1, 0.2]}, donors=["d1", "d1"]))


def test_t_confidence_interval():
    ci_low, ci_high = t_confidence_interval([0.10, 0.14, 0.18, 0.22, 0.26, 0.30])

    assert (ci_low, ci_high) == (pytest.approx(0.1214, abs=1e-4), pytest.approx(0.2786, abs=1e-4))


@pytest.mark.parametrize("values", [[], [0.3], [np.nan, 0.3]])
def test_t_confidence_interval_needs_two_values(values):
    assert np.isnan(t_confidence_interval(values)).all()


@pytest.mark.parametrize("statistical_tests", [False, True])
def test_summarise_repertoire_scores(statistical_tests):
    repertoire_scores = make_repertoire_scores({"vae": [0.2, 0.6, 0.4]}, donors=["d1", "d1", "d2"])

    summary = summarise_repertoire_scores(repertoire_scores, statistical_tests)

    vae = summary.iloc[0]
    assert (vae["n_repertoires"], vae["n_donors"]) == (3, 2)
    assert (vae["mean"], vae["sd"]) == (pytest.approx(0.4), pytest.approx(0.2))
    assert ("ci_low" in summary.columns) == statistical_tests


@pytest.mark.parametrize("differences, expected_p", [
    # All six repertoires in the same direction: only "no flips" and "all flipped" are as extreme, 2 / 2^6
    ([0.02, 0.03, 0.01, 0.04, 0.02, 0.03], 2 / 64),
    # One repertoire in the other direction
    ([0.02, 0.03, -0.01, 0.04, 0.02, 0.03], 0.0625),
    ([0.0, 0.0, 0.0], 1.0),
])
def test_sign_flip_test_exact(differences, expected_p):
    p_value, test = sign_flip_test(differences)

    assert p_value == pytest.approx(expected_p)
    assert test == "exact sign-flip"


def test_sign_flip_test_exact_matches_brute_force():
    differences = np.random.default_rng(0).normal(0.01, 0.02, size=9)
    observed = abs(differences.mean())
    flipped_means = [abs((differences * np.array(signs)).mean())
                     for signs in itertools.product([1, -1], repeat=len(differences))]
    expected_p = np.mean([mean >= observed - 1e-12 for mean in flipped_means])

    assert sign_flip_test(differences)[0] == pytest.approx(expected_p)


def test_sign_flip_test_monte_carlo(monkeypatch):
    monkeypatch.setattr(statistics_utils, "N_PERMUTATIONS", 2000)
    differences = np.full(30, 0.01)

    p_value, test = sign_flip_test(differences)

    # No random sign combination is as extreme as all positive, so p is the smallest possible Monte Carlo p-value
    assert p_value == pytest.approx(1 / 2001)
    assert test.startswith("Monte Carlo")


def test_sign_flip_test_without_differences():
    assert np.isnan(sign_flip_test([])[0])


def test_min_possible_p():
    assert min_possible_p(6) == pytest.approx(0.03125)
    assert min_possible_p(11) == pytest.approx(2 / 2048)
    assert min_possible_p(100) == pytest.approx(1 / (statistics_utils.N_PERMUTATIONS + 1))


def test_compare_paired_descriptive_counts_repertoires_and_donors():
    # rep_3 has no score for b, so the differences are 0.1 (d1), 0.1 (d1) and -0.3 (d2)
    repertoire_scores = make_repertoire_scores({"a": [0.2, 0.3, 0.2, 0.5], "b": [0.1, 0.2, 0.5]},
                                               donors=["d1", "d1", "d2", "d3"])

    comparisons, differences = compare_paired(repertoire_scores, [("a", "b")], statistical_tests=False)

    assert list(comparisons.columns) == DESCRIPTIVE_COMPARISON_COLUMNS
    row = comparisons.iloc[0]
    assert (row["n_repertoires"], row["n_donors"]) == (3, 2)
    assert row["mean_difference"] == pytest.approx(-0.1 / 3)
    assert (row["n_first_higher"], row["n_first_lower"]) == (2, 1)
    assert (row["n_donors_first_higher"], row["n_donors_first_lower"]) == (1, 1)
    assert differences["difference"].tolist() == pytest.approx([0.1, 0.1, -0.3])
    assert differences["donor"].tolist() == ["d1", "d1", "d2"]


def test_compare_paired_with_tests_holm_correction():
    repertoire_scores = make_repertoire_scores({"a": [0.1] * 6, "b": [0.2] * 6, "c": [0.3] * 6})

    comparisons, _ = compare_paired(repertoire_scores, [("a", "b"), ("a", "c")], statistical_tests=True)

    # Both raw p-values are at the floor of 2 / 2^6, and Holm multiplies the smallest by the number of tests
    assert comparisons["p_value"].tolist() == pytest.approx([0.03125, 0.03125])
    assert comparisons["p_adjusted"].tolist() == pytest.approx([0.0625, 0.0625])
    assert is_underpowered(comparisons)


def test_is_underpowered():
    repertoire_scores = make_repertoire_scores({"a": [0.1] * 11, "b": [0.2] * 11})

    assert not is_underpowered(compare_paired(repertoire_scores, [("a", "b")], statistical_tests=True)[0])
    assert not is_underpowered(compare_paired(repertoire_scores, [("a", "b")], statistical_tests=False)[0])


def test_validate_contrasts():
    assert validate_contrasts([["a", "b"]], ["a", "b"]) == [("a", "b")]
    with pytest.raises(ValueError, match="Unknown models"):
        validate_contrasts([["a", "x"]], ["a", "b"])
    with pytest.raises(ValueError, match="pairs"):
        validate_contrasts([["a", "b", "c"]], ["a", "b", "c"])


@pytest.mark.parametrize("statistical_tests, n_repertoires, n_donors, expected", [
    (True, 20, 20, "Error bars: 95% CI across repertoires (n=20); dots: repertoires"),
    (False, 11, 5, "Error bars: SD across repertoires (n=11 from 5 donors); dots: repertoires"),
])
def test_describe_error_bars(statistical_tests, n_repertoires, n_donors, expected):
    assert describe_error_bars(statistical_tests, n_repertoires, n_donors) == expected


@pytest.mark.parametrize("statistical_tests", [False, True])
def test_run_repertoire_statistics(tmp_path, statistical_tests):
    repertoire_scores = make_repertoire_scores({"a": [0.2, 0.3, 0.4], "b": [0.3, 0.4, 0.6], "c": [0.5, 0.5, 0.5]})
    reference_scores = pd.DataFrame({"repertoire": ["rep_0", "rep_1", "rep_2"], "score": [0.05, 0.05, 0.05]})
    output_prefix = str(tmp_path / "statistics" / "metric")

    results = run_repertoire_statistics(repertoire_scores, ["a", "b", "c"], output_prefix, statistical_tests,
                                        reference_scores, [["a", "b"]])

    assert results["vs_train_test"][["first", "second"]].values.tolist() == [
        ["a", TRAIN_TEST_REFERENCE], ["b", TRAIN_TEST_REFERENCE], ["c", TRAIN_TEST_REFERENCE]]
    assert results["contrasts"][["first", "second"]].values.tolist() == [["a", "b"]]
    assert len(results["all_pairs"]) == 3
    assert ("p_adjusted" in results["all_pairs"].columns) == statistical_tests
    if statistical_tests:
        for name in ["vs_train_test", "contrasts", "all_pairs"]:
            assert set(results[name]["correction"]) == {"holm"}
    assert TRAIN_TEST_REFERENCE in set(results["summary"]["source"])
    for table in ["repertoire_scores", "summary", "vs_train_test", "vs_train_test_differences", "contrasts",
                  "contrasts_differences", "all_pairs", "all_pairs_differences"]:
        assert os.path.exists(f"{output_prefix}_{table}.tsv")


def test_run_repertoire_statistics_with_tests_requires_one_repertoire_per_donor(tmp_path):
    repertoire_scores = make_repertoire_scores({"a": [0.2, 0.3], "b": [0.3, 0.4]}, donors=["d1", "d1"])

    with pytest.raises(ValueError, match="one repertoire per donor"):
        run_repertoire_statistics(repertoire_scores, ["a", "b"], str(tmp_path / "metric"), statistical_tests=True)


def test_run_repertoire_statistics_assigns_donors_to_reference_scores(tmp_path):
    repertoire_scores = make_repertoire_scores({"a": [0.2, 0.3]}, donors=["6271", "6271"])
    repertoire_scores["repertoire"] = ["6271_pLN_CD8", "6271_pLN_Treg"]
    reference_scores = pd.DataFrame({"repertoire": ["6271_pLN_CD8", "6271_pLN_Treg"], "score": [0.05, 0.05]})

    results = run_repertoire_statistics(repertoire_scores, ["a"], str(tmp_path / "metric"), False, reference_scores,
                                        donor_pattern="^([^_]+)_")

    assert results["vs_train_test"][["n_repertoires", "n_donors"]].values.tolist() == [[2, 1]]


@pytest.mark.parametrize("analysis_fields, expected", [
    ({}, (None, False, None)),
    ({"contrasts": [["a", "b"]], "statistical_tests": True, "donor_pattern": "^([^_]+)_"},
     ([["a", "b"]], True, "^([^_]+)_")),
])
def test_main_config_reads_statistics_fields(tmp_path, analysis_fields, expected):
    analysis = {"name": "gene_usage", "model_names": ["a", "b"], "default_model_name": "humanTRB",
                "subfolder_name": "subfolder", "receptor_type": "TCR", **analysis_fields}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({"n_experiments": 1, "output_dir": str(tmp_path / "out"),
                                      "input_dir": str(tmp_path), "seed": 42, "analyses": [analysis]}))

    config = MainConfig(str(config_path)).analysis_configs[0]

    assert (config.contrasts, config.statistical_tests, config.donor_pattern) == expected
