"""Redraw only the UMI clone frequency summary plots (train_test bar chart and statistics), without the scatter plots.

Reads the saved model scores (clone_frequency_jsd_scores.tsv) and recomputes only the train vs. test reference JSDs,
which are not saved. Then runs the same summary and statistics steps as analyse_clone_frequencies_umi.

Usage: PYTHONPATH=. python3 adhoc_analysis/replot_clone_frequency_summary.py configs/pipelines/experimental_UMI_analysis_clone_frequencies.yaml
"""
import sys

import pandas as pd
from scipy.spatial import distance

from gen_airr_bm.analysis.analyse_clone_frequencies_umi import (REFERENCE_PREFIX, collect_clone_frequencies_reference,
                                                                run_clone_frequency_statistics, summarize_and_plot_all)
from gen_airr_bm.core.main_config import MainConfig


def main(config_path: str) -> None:
    configs = [c for c in MainConfig(config_path).analysis_configs if c.analysis == "clone_frequencies"]
    for analysis_config in configs:
        scores_df = pd.read_csv(f"{analysis_config.analysis_output_dir}/clone_frequency_jsd_scores.tsv", sep="\t")

        ref_frequencies = collect_clone_frequencies_reference(analysis_config)
        reference_scores = pd.DataFrame(
            [{"repertoire": name.removeprefix(REFERENCE_PREFIX),
              "score": distance.jensenshannon(df["freq_train"], df["freq_test"], base=2) ** 2}
             for name, df in sorted(ref_frequencies.items()) if "_all" not in name],
            columns=["repertoire", "score"])
        reference_score = float(reference_scores["score"].mean()) if len(reference_scores) else None

        summarize_and_plot_all(analysis_config, scores_df, reference_score)
        run_clone_frequency_statistics(analysis_config, scores_df, reference_scores)
        print(f"Redrew the summary plots in {analysis_config.analysis_output_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
