"""Aggregate MAP metrics from all subdirectories into a single CSV, and test them with a permutation test.

Permutation test: the Jaccard similarity matrix saved next to each map_metrics.tsv is kept fixed, the
phenotype (or subject) labels are shuffled across repertoires, and MAP is recomputed. The p-value is the share
of shuffles whose MAP is at least the observed one. Shuffling keeps the label counts, so singleton labels
(skipped by compute_map) stay singletons and the null matches the observed MAP. With up to
EXACT_MAX_REPERTOIRES repertoires all orderings are enumerated (exact p-value); above that, random shuffles
are used with p = (1 + #null >= observed) / (1 + n_permutations).

Multiple testing: p-values are corrected (Benjamini-Hochberg by default) within each family, where a family
is one receptor type x one label (phenotype or subject) over the tested models. One test is run per model
configuration (e.g. ProGen_kmer and ProGen_overlap are two tests). The family is therefore set by --models and
--labels: choose it before looking at the p-values. "train" is excluded unless listed in --models, since it is
the reference, not a model being evaluated.

Outputs in --output_dir:
    all_map_metrics.csv        MAP scores of every map_metrics.tsv found
    map_permutation_tests.csv  one row per tested model x label: MAP, p, corrected p, family size, null summary
    map_metrics_<receptor>.png bar chart per receptor type

Usage:
    python adhoc_analysis/aggregate_map.py --root_dir <results>/BCR/phenotype --output_dir <out> \\
        --labels subject --models ProGen_kmer ProGen_overlap TCRpeg_kmer TCRpeg_overlap
"""

import argparse
import glob
import itertools
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.stats.multitest import multipletests

from gen_airr_bm.analysis.analyse_phenotype import compute_map, extract_phenotype, extract_subject
from gen_airr_bm.utils.plotting_utils import get_collection_specification_for_title

EXACT_MAX_REPERTOIRES = 8  # 8! = 40,320 orderings; enumerating more gets slow
TOLERANCE = 1e-12  # treat floating-point ties with the observed MAP as ">="
LABEL_EXTRACTORS = {"phenotype": extract_phenotype, "subject": extract_subject}


def map_permutation_test(similarities_df, labels, n_permutations=9999, seed=0):
    """Permutation test for MAP: shuffle labels over a fixed similarity matrix and recompute MAP.

    Args:
        similarities_df: pd.DataFrame, square similarity matrix (rows/columns in the same order as labels)
        labels: list of labels, one per repertoire
        n_permutations: number of random shuffles when the test is not exact
        seed: random seed for the random shuffles
    Returns:
        dict with the observed MAP, p-value, null mean, null 95th percentile, and method ("exact" or "random")
    """
    observed = compute_map(similarities_df, labels)
    n = len(labels)

    if n <= EXACT_MAX_REPERTOIRES:
        # Every ordering of the labels; orderings that give the same grouping are counted equally often.
        null = np.array([compute_map(similarities_df, [labels[i] for i in order])
                         for order in itertools.permutations(range(n))])
        p_value = float(np.mean(null >= observed - TOLERANCE))
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        null = np.array([compute_map(similarities_df, list(rng.permutation(labels)))
                         for _ in range(n_permutations)])
        p_value = float((1 + np.sum(null >= observed - TOLERANCE)) / (1 + n_permutations))
        method = "random"

    return {"map": observed, "p": p_value, "null_mean": float(null.mean()),
            "null_95": float(np.quantile(null, 0.95)), "method": method}


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
                         "permutation_method": test["method"], "source_path": row["source_path"]})

    return pd.DataFrame(rows)


def correct_for_multiple_testing(tests_df, method="fdr_bh"):
    """Correct p-values within each family (receptor type x label) over the tested models.

    Args:
        tests_df: output of run_permutation_tests
        method: statsmodels multipletests method, e.g. "fdr_bh" (Benjamini-Hochberg) or "holm"
    Returns:
        tests_df with columns n_tests (family size), p_adjusted and correction_method
    """
    tests_df = tests_df.copy()
    tests_df["n_tests"] = tests_df.groupby(["receptor_type", "label"])["p"].transform("size")
    tests_df["p_adjusted"] = tests_df.groupby(["receptor_type", "label"])["p"].transform(
        lambda p: multipletests(p, method=method)[1])
    tests_df["correction_method"] = method
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
    parser.add_argument("--correction", default="fdr_bh",
                        help="statsmodels multipletests method: fdr_bh (default), holm, bonferroni, fdr_by, ...")
    parser.add_argument("--n_permutations", type=int, default=9999,
                        help=f"Random shuffles when there are more than {EXACT_MAX_REPERTOIRES} repertoires")
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
                tests = correct_for_multiple_testing(tests, args.correction)
                tests_path = os.path.join(args.output_dir, "map_permutation_tests.csv")
                tests.to_csv(tests_path, index=False)
                print(f"\nPermutation tests saved: {tests_path}")
                print(tests.drop(columns="source_path").to_string(index=False))
        plot_map_metrics_per_receptor(combined, args.output_dir)
