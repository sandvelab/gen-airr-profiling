"""Aggregate MAP metrics from all subdirectories into a single CSV, and test them with a permutation test.

Permutation test: the Jaccard similarity matrix saved next to each map_metrics.tsv is kept fixed, the
phenotype (or subject) labels are shuffled across repertoires, and MAP is recomputed. The p-value is the share
of shuffles whose MAP is at least the observed one. Shuffling keeps the label counts, so singleton labels
(skipped by compute_map) stay singletons and the null matches the observed MAP.

Every ordering of the labels is equally likely under the null, and orderings that give the same labeling (e.g.
swapping two CD8 repertoires) give the same MAP, so the exact null only needs the distinct labelings, each
counted once (11 TCR repertoires: 11,550 phenotype and 277,200 subject labelings instead of 11! orderings).
With up to EXACT_MAX_LABELINGS distinct labelings the null is exact; above that, random shuffles are used with
p = (1 + #null >= observed) / (1 + n_permutations). The labelings (or shuffles) depend only on the labels, so
they are made once and reused for every model; the null MAP values are computed per model, since their spread
depends on the model's similarity matrix (only their mean is the same for all models).

Multiple testing: p-values are corrected within each family, where a family is one receptor type x one label
(phenotype or subject) over the tested models. One test is run per model configuration (e.g. ProGen_kmer and
ProGen_overlap are two tests). The family is therefore set by --models and --labels: choose it before looking
at the p-values. "train" is excluded unless listed in --models, since it is the reference, not a model being
evaluated. By default three corrections are reported side by side, since the tests are not independent (all
models in a family share the same repertoires and labels):
    p_bh          Benjamini-Hochberg: false discovery rate, valid for independent or positively dependent tests
    p_by          Benjamini-Yekutieli: false discovery rate under any dependence (more conservative)
    p_bonferroni  Bonferroni: family-wise error rate under any dependence (most conservative)

Outputs in --output_dir:
    all_map_metrics.csv        MAP scores of every map_metrics.tsv found
    map_permutation_tests.csv  one row per tested model x label: MAP, p, corrected p-values, family size,
                               null summary
    map_metrics_<receptor>.png bar chart per receptor type

Usage:
    python adhoc_analysis/aggregate_map.py --root_dir <results>/BCR/phenotype --output_dir <out> \\
        --labels subject --models ProGen_kmer ProGen_overlap TCRpeg_kmer TCRpeg_overlap
"""

import argparse
import glob
import itertools
import math
import os
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.stats.multitest import multipletests

from gen_airr_bm.analysis.analyse_phenotype import compute_map, extract_phenotype, extract_subject
from gen_airr_bm.utils.plotting_utils import get_collection_specification_for_title

EXACT_MAX_LABELINGS = 1_000_000  # distinct labelings scored at once; about 5 s and 100 MB per model at 11 repertoires
TOLERANCE = 1e-12  # treat floating-point ties with the observed MAP as ">="
LABEL_EXTRACTORS = {"phenotype": extract_phenotype, "subject": extract_subject}
# Column suffix -> statsmodels multipletests method
CORRECTIONS = {"bh": "fdr_bh", "by": "fdr_by", "bonferroni": "bonferroni", "holm": "holm"}
DEFAULT_CORRECTIONS = ("bh", "by", "bonferroni")


def count_labelings(labels):
    """Number of distinct labelings of the repertoires with the same label counts."""
    counts = pd.Series(labels).value_counts()
    return math.factorial(len(labels)) // math.prod(math.factorial(c) for c in counts)


@lru_cache(maxsize=None)
def null_labelings(labels, n_permutations, seed):
    """Labelings to score under the null, as label codes (one row per labeling). Cached, so every model with the
    same labels reuses them.

    Args:
        labels: tuple of labels, one per repertoire
        n_permutations: number of random shuffles when the distinct labelings are too many to enumerate
        seed: random seed for the random shuffles
    Returns:
        (np.ndarray of shape (n_labelings, n_repertoires), method "exact" or "random")
    """
    codes = pd.factorize(pd.Series(labels))[0]
    n = len(codes)
    if count_labelings(labels) > EXACT_MAX_LABELINGS:
        rng = np.random.default_rng(seed)
        return rng.permuted(np.tile(codes, (n_permutations, 1)), axis=1), "random"

    # Place the labels one after another: choose the positions of the first label among all positions, then of the
    # second among the remaining ones, and so on.
    labelings = [np.full(n, -1)]
    for code, count in enumerate(np.bincount(codes)):
        extended = []
        for labeling in labelings:
            free = np.flatnonzero(labeling < 0)
            for positions in itertools.combinations(free, count):
                new = labeling.copy()
                new[list(positions)] = code
                extended.append(new)
        labelings = extended
    return np.array(labelings), "exact"


def compute_map_for_labelings(similarities_df, labelings):
    """MAP of one similarity matrix for many labelings at once; the same result as compute_map per labeling.

    Args:
        similarities_df: pd.DataFrame, square similarity matrix
        labelings: np.ndarray of label codes, one row per labeling, columns in the order of the matrix rows
    Returns:
        np.ndarray with one MAP per labeling
    """
    sim = similarities_df.values.copy().astype(float)
    np.fill_diagonal(sim, -np.inf)
    n = sim.shape[0]
    ap_sum = np.zeros(len(labelings))
    n_queries = np.zeros(len(labelings))
    for i in range(n):
        ranking = np.argsort(-sim[i])  # the same ranking as compute_map
        ranking = ranking[ranking != i]
        relevant = labelings[:, ranking] == labelings[:, [i]]
        n_relevant = relevant.sum(axis=1)
        precisions = np.cumsum(relevant, axis=1) / np.arange(1, n)
        has_peers = n_relevant > 0  # repertoires without peers are skipped, as in compute_map
        ap_sum += np.where(has_peers, (precisions * relevant).sum(axis=1) / np.maximum(n_relevant, 1), 0.0)
        n_queries += has_peers
    return np.divide(ap_sum, n_queries, out=np.zeros(len(labelings)), where=n_queries > 0)


def map_permutation_test(similarities_df, labels, n_permutations=9999, seed=0):
    """Permutation test for MAP: shuffle labels over a fixed similarity matrix and recompute MAP.

    Args:
        similarities_df: pd.DataFrame, square similarity matrix (rows/columns in the same order as labels)
        labels: list of labels, one per repertoire
        n_permutations: number of random shuffles when the test is not exact
        seed: random seed for the random shuffles
    Returns:
        dict with the observed MAP, p-value, null mean, null 95th percentile, method ("exact" or "random") and
        the number of null labelings
    """
    observed = compute_map(similarities_df, labels)
    labelings, method = null_labelings(tuple(labels), n_permutations, seed)
    null = compute_map_for_labelings(similarities_df, labelings)

    if method == "exact":
        p_value = float(np.mean(null >= observed - TOLERANCE))
    else:
        p_value = float((1 + np.sum(null >= observed - TOLERANCE)) / (1 + n_permutations))

    return {"map": observed, "p": p_value, "null_mean": float(null.mean()),
            "null_95": float(np.quantile(null, 0.95)), "method": method, "n_null": len(null)}


def run_permutation_tests(df, models=None, labels=("phenotype", "subject"), n_permutations=9999, seed=0):
    """Run one MAP permutation test per model and label.

    Reads similarities_matrix.tsv from the same directory as each row's map_metrics.tsv (column source_path).

    Args:
        df: aggregated MAP metrics (output of aggregate_map_metrics)
        models: model names to test; None tests every model except "train"
        labels: labels to test, from "phenotype" and "subject"
        n_permutations: number of random shuffles when the test is not exact
        seed: random seed for the random shuffles
    Returns:
        pd.DataFrame with one row per tested model x label
    """
    if models is None:
        df = df[df["model"] != "train"]
    else:
        missing = sorted(set(models) - set(df["model"]))
        if missing:
            print(f"Warning: no map_metrics.tsv found for models {missing}")
        df = df[df["model"].isin(models)]

    rows = []
    for _, row in df.iterrows():
        matrix_path = os.path.join(os.path.dirname(row["source_path"]), "similarities_matrix.tsv")
        if not os.path.exists(matrix_path):
            print(f"No similarities_matrix.tsv next to {row['source_path']}; skipping permutation test")
            continue

        similarities_df = pd.read_csv(matrix_path, sep="\t", index_col=0)
        names = list(similarities_df.index)
        label_config = SimpleNamespace(receptor_type=row["receptor_type"])

        for label_name in labels:
            label_values = [LABEL_EXTRACTORS[label_name](label_config, name) for name in names]
            test = map_permutation_test(similarities_df, label_values, n_permutations, seed)
            if not np.isclose(test["map"], row[f"map_{label_name}"]):
                print(f"Warning: recomputed MAP {label_name} ({test['map']:.4f}) differs from "
                      f"{row['source_path']} ({row[f'map_{label_name}']:.4f})")
            rows.append({"receptor_type": row["receptor_type"], "model": row["model"], "label": label_name,
                         "n_repertoires": len(names), "map": test["map"], "p": test["p"],
                         "null_mean": test["null_mean"], "null_95": test["null_95"],
                         "permutation_method": test["method"], "n_null": test["n_null"],
                         "source_path": row["source_path"]})

    return pd.DataFrame(rows)


def correct_for_multiple_testing(tests_df, corrections=DEFAULT_CORRECTIONS):
    """Correct p-values within each family (receptor type x label) over the tested models.

    Args:
        tests_df: output of run_permutation_tests
        corrections: keys of CORRECTIONS, e.g. ("bh", "by", "bonferroni")
    Returns:
        tests_df with column n_tests (family size) and one column p_<correction> per correction
    """
    tests_df = tests_df.copy()
    families = tests_df.groupby(["receptor_type", "label"])["p"]
    tests_df["n_tests"] = families.transform("size")
    for correction in corrections:
        method = CORRECTIONS[correction]
        tests_df[f"p_{correction}"] = families.transform(lambda p: multipletests(p, method=method)[1])
    return tests_df.sort_values(["receptor_type", "label", "p", "model"]).reset_index(drop=True)


def aggregate_map_metrics(root_dir=".", output_path="all_map_metrics.csv"):
    """Find all map_metrics.tsv files under root_dir and combine them into one CSV.

    Args:
        root_dir: Directory to search from (default: current directory)
        output_path: Where to save the combined CSV
    """
    pattern = os.path.join(root_dir, "**", "map_metrics.tsv")
    paths = glob.glob(pattern, recursive=True)

    if not paths:
        print(f"No map_metrics.tsv files found under {root_dir}")
        return None

    print(f"Found {len(paths)} files:")
    for p in paths:
        print(f"  {p}")

    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep='\t')
        df['source_path'] = path  # track where each row came from
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(['receptor_type', 'map_phenotype'], ascending=[True, False])
    combined.to_csv(output_path, index=False)

    print(f"\nAggregated {len(combined)} rows into {output_path}")
    print("\nSummary:")
    print(combined.to_string(index=False))

    return combined


def plot_map_metrics_per_receptor(df, output_dir="."):
    """Save one grouped bar chart (MAP phenotype + MAP subject per model) per receptor type."""
    os.makedirs(output_dir, exist_ok=True)

    df = df[df["model"] != "train"]

    for receptor_type, sub in df.groupby("receptor_type"):
        sub = sub.sort_values("map_phenotype", ascending=False)

        fig = go.Figure(
            data=[
                go.Bar(
                    name="MAP Phenotype",
                    x=sub["model"],
                    y=sub["map_phenotype"]
                ),
                go.Bar(
                    name="MAP Subject",
                    x=sub["model"],
                    y=sub["map_subject"]
                ),
            ]
        )
        color_palette = px.colors.qualitative.Safe
        collection_specification = get_collection_specification_for_title(receptor_type.upper())
        title_text = (f"Mean Average Precision for Jaccard-Based Ranking <br>of Repertoire Labels for "
                      f"{collection_specification} Repertoires")
        fig.update_layout(
            title={'text': title_text,
                   'font': {'size': 20}},
            xaxis_title={'text': "Model", 'font': {'size': 20}},
            yaxis_title={'text': "MAP Score", 'font': {'size': 20}},
            xaxis=dict(tickangle=-45, tickfont=dict(size=18)),
            yaxis=dict(tickfont=dict(size=18)),
            barmode="group",
            template="plotly_white",
            colorway=color_palette
        )

        out_path = os.path.join(output_dir, f"map_metrics_{receptor_type}.png")
        fig.write_image(out_path, scale=3)
        print(f"Plot saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_dir", default=".", help="Directory searched recursively for map_metrics.tsv")
    parser.add_argument("--output_dir", default=".", help="Where the CSVs and plots are written")
    parser.add_argument("--no_permutation_test", action="store_true", help="Skip the permutation test")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to test, e.g. ProGen_kmer ProGen_overlap (default: all except train). "
                             "Together with --labels this sets the multiple-testing family.")
    parser.add_argument("--labels", nargs="+", choices=list(LABEL_EXTRACTORS), default=list(LABEL_EXTRACTORS),
                        help="Labels to test (default: phenotype subject)")
    parser.add_argument("--corrections", nargs="+", choices=list(CORRECTIONS), default=list(DEFAULT_CORRECTIONS),
                        help="Multiple-testing corrections to report, one p_<name> column each "
                             "(default: bh by bonferroni; holm is also available)")
    parser.add_argument("--n_permutations", type=int, default=9999,
                        help=f"Random shuffles when there are more than {EXACT_MAX_LABELINGS:,} distinct labelings")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    combined = aggregate_map_metrics(args.root_dir, os.path.join(args.output_dir, "all_map_metrics.csv"))
    if combined is not None:
        if not args.no_permutation_test:
            tests = run_permutation_tests(combined, args.models, args.labels, args.n_permutations, args.seed)
            if tests.empty:
                print("No permutation tests were run")
            else:
                tests = correct_for_multiple_testing(tests, args.corrections)
                tests_path = os.path.join(args.output_dir, "map_permutation_tests.csv")
                tests.to_csv(tests_path, index=False)
                print(f"\nPermutation tests saved: {tests_path}")
                print(tests.drop(columns="source_path").to_string(index=False))
        plot_map_metrics_per_receptor(combined, args.output_dir)
