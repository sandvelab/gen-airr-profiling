"""Aggregate MAP metrics from all subdirectories into a single CSV."""

import glob
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from gen_airr_bm.utils.plotting_utils import get_collection_specification_for_title


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
        fig.write_image(out_path, scale=2)
        print(f"Plot saved: {out_path}")


if __name__ == "__main__":
    combined = aggregate_map_metrics()
    if combined is not None:
        plot_map_metrics_per_receptor(combined)
