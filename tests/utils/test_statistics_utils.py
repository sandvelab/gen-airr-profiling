import os

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats

from gen_airr_bm.core.main_config import MainConfig
from gen_airr_bm.utils.statistics_utils import (DESCRIPTIVE_COMPARISON_COLUMNS, TRAIN_TEST_REFERENCE,
                                                aggregate_repertoire_scores, check_one_repertoire_per_donor,
                                                compare_paired, describe_error_bars, get_donor, paired_t_test,
                                                correct_across_metrics, run_repertoire_statistics,
                                                summarise_repertoire_scores, t_confidence_interval,
                                                validate_contrasts, write_repertoire_statistics)


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
    check_one_repertoire_per_donor(make_repertoire_scores({"a": [0.1, 0.2]}), "^(rep_\\d+)$")
    with pytest.raises(ValueError, match="one repertoire per donor"):
        check_one_repertoire_per_donor(make_repertoire_scores({"a": [0.1, 0.2]}, donors=["d1", "d1"]), "^(rep)_")


def test_check_one_repertoire_per_donor_requires_donor_pattern():
    with pytest.raises(ValueError, match="require a donor_pattern"):
        check_one_repertoire_per_donor(make_repertoire_scores({"a": [0.1, 0.2]}), None)


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


def test_paired_t_test_matches_scipy():
    first, second = np.array([0.21, 0.35, 0.28, 0.40, 0.31]), np.array([0.25, 0.36, 0.33, 0.41, 0.30])

    p_value, test = paired_t_test(first - second)

    assert p_value == pytest.approx(stats.ttest_rel(first, second).pvalue)
    assert test == "paired t-test"


@pytest.mark.parametrize("differences", [[], [0.1], [np.nan, 0.1], [0.0, 0.0, 0.0]])
def test_paired_t_test_without_enough_differences(differences):
    assert np.isnan(paired_t_test(differences)[0])


def test_paired_t_test_agrees_with_confidence_interval():
    # The p-value is below 0.05 exactly when the 95% confidence interval of the mean difference excludes zero
    rng = np.random.default_rng(0)
    for _ in range(50):
        differences = rng.normal(0.01, 0.02, size=8)
        ci_low, ci_high = t_confidence_interval(differences)
        assert (paired_t_test(differences)[0] < 0.05) == (ci_low > 0 or ci_high < 0)


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


def test_compare_paired_with_tests_bonferroni_correction():
    # a - b and a - c have different raw p-values. Unlike Holm, Bonferroni multiplies every p-value, not only the
    # smallest, by the number of tests.
    repertoire_scores = make_repertoire_scores({"a": [0.10, 0.12, 0.11, 0.13, 0.10, 0.12],
                                                "b": [0.20, 0.21, 0.19, 0.22, 0.20, 0.23],
                                                "c": [0.30, 0.00, 0.30, 0.00, 0.30, 0.05]})

    comparisons, _ = compare_paired(repertoire_scores, [("a", "b"), ("a", "c")], statistical_tests=True)

    p_values = comparisons["p_value"].to_numpy()
    assert p_values[0] < 0.001 < 0.5 < p_values[1]
    # Adjusted p-values are capped at 1
    assert comparisons["p_adjusted"].tolist() == pytest.approx([2 * p_values[0], 1.0])
    assert set(comparisons["test"]) == {"paired t-test"}


@pytest.mark.parametrize("n_repertoires, n_donors, expected", [
    (20, 20, "Error bars: SD across repertoires (n=20); dots: repertoires"),
    (11, 5, "Error bars: SD across repertoires (n=11 from 5 donors); dots: repertoires"),
])
def test_describe_error_bars(n_repertoires, n_donors, expected):
    assert describe_error_bars(n_repertoires, n_donors) == expected


@pytest.mark.parametrize("statistical_tests", [False, True])
def test_run_repertoire_statistics(tmp_path, statistical_tests):
    repertoire_scores = make_repertoire_scores({"a": [0.2, 0.3, 0.4], "b": [0.3, 0.4, 0.6], "c": [0.5, 0.5, 0.5]})
    reference_scores = pd.DataFrame({"repertoire": ["rep_0", "rep_1", "rep_2"], "score": [0.05, 0.05, 0.05]})
    output_prefix = str(tmp_path / "statistics" / "metric")

    results = run_repertoire_statistics(repertoire_scores, ["a", "b", "c"], output_prefix, statistical_tests,
                                        reference_scores, [["a", "b"]], donor_pattern="^(rep_\\d+)$")

    assert results["vs_train_test"][["first", "second"]].values.tolist() == [
        ["a", TRAIN_TEST_REFERENCE], ["b", TRAIN_TEST_REFERENCE], ["c", TRAIN_TEST_REFERENCE]]
    assert results["contrasts"][["first", "second"]].values.tolist() == [["a", "b"]]
    assert len(results["all_pairs"]) == 3
    assert ("p_adjusted" in results["all_pairs"].columns) == statistical_tests
    if statistical_tests:
        for name in ["vs_train_test", "contrasts", "all_pairs"]:
            assert set(results[name]["correction"]) == {"bonferroni"}
    assert TRAIN_TEST_REFERENCE in set(results["summary"]["source"])
    for table in ["repertoire_scores", "summary", "vs_train_test", "vs_train_test_differences", "contrasts",
                  "contrasts_differences", "all_pairs", "all_pairs_differences"]:
        assert os.path.exists(f"{output_prefix}_{table}.tsv")


def test_run_repertoire_statistics_with_tests_requires_one_repertoire_per_donor(tmp_path):
    repertoire_scores = make_repertoire_scores({"a": [0.2, 0.3], "b": [0.3, 0.4]}, donors=["d1", "d1"])

    with pytest.raises(ValueError, match="one repertoire per donor"):
        run_repertoire_statistics(repertoire_scores, ["a", "b"], str(tmp_path / "metric"), statistical_tests=True,
                                  donor_pattern="^(rep)_")


def test_run_repertoire_statistics_with_tests_requires_donor_pattern(tmp_path):
    repertoire_scores = make_repertoire_scores({"a": [0.2, 0.3], "b": [0.3, 0.4]})

    with pytest.raises(ValueError, match="require a donor_pattern"):
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


def test_run_repertoire_statistics_without_writing(tmp_path):
    repertoire_scores = make_repertoire_scores({"a": [0.2, 0.3, 0.4], "b": [0.3, 0.4, 0.6]})
    output_prefix = str(tmp_path / "statistics" / "metric")

    results = run_repertoire_statistics(repertoire_scores, ["a", "b"], output_prefix, False, write=False)

    assert not os.path.exists(tmp_path / "statistics")
    write_repertoire_statistics(results, output_prefix)
    for table in ["repertoire_scores", "summary", "all_pairs", "all_pairs_differences"]:
        assert os.path.exists(f"{output_prefix}_{table}.tsv")


def test_correct_across_metrics_makes_one_family_per_comparison_set(tmp_path):
    results_by_metric = {
        metric: run_repertoire_statistics(make_repertoire_scores(scores), ["a", "b", "c"],
                                          str(tmp_path / str(metric)), True, donor_pattern="^(rep_\\d+)$",
                                          write=False)
        for metric, scores in {10: {"a": [0.2, 0.3, 0.4], "b": [0.3, 0.5, 0.6], "c": [0.5, 0.5, 0.7]},
                               11: {"a": [0.1, 0.2, 0.2], "b": [0.4, 0.4, 0.5], "c": [0.3, 0.6, 0.6]}}.items()}
    p_values = {metric: results["all_pairs"]["p_value"].to_numpy() for metric, results in results_by_metric.items()}

    correct_across_metrics(results_by_metric)

    # Bonferroni over the 3 pairs of both metrics, instead of over the 3 pairs of each metric
    for metric, results in results_by_metric.items():
        assert results["all_pairs"]["p_adjusted"].to_numpy() == pytest.approx(np.minimum(p_values[metric] * 6, 1))


def test_correct_across_metrics_leaves_descriptive_results(tmp_path):
    results = run_repertoire_statistics(make_repertoire_scores({"a": [0.2, 0.3], "b": [0.3, 0.4]}), ["a", "b"],
                                        str(tmp_path / "metric"), False, write=False)

    correct_across_metrics({10: results, 11: results})

    assert "p_adjusted" not in results["all_pairs"].columns
