from collections import Counter
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import jensenshannon

from gen_airr_bm.analysis.distribution.base_distribution_strategy import BaseDistributionStrategy
from gen_airr_bm.utils.plotting_utils import wrap_title


class LengthDistributionStrategy(BaseDistributionStrategy):
    def compute_divergence(self, seqs1, seqs2):
        lengths1 = [len(seq) for seq in seqs1]
        lengths2 = [len(seq) for seq in seqs2]
        dist1 = Counter(lengths1)
        dist2 = Counter(lengths2)
        return [compute_jsd_length(dist1, dist2)]

    def plot_distributions_per_dataset(
            self,
            analysis_config,
            dataset_label,
            gen_seqs,
            ref_seqs,
            gen_label,
            ref_label
    ):
        ref_lengths = [len(seq) for seq in ref_seqs[0]]
        dist_ref = Counter(ref_lengths)

        dist_gens = [Counter([len(seq) for seq in run]) for run in gen_seqs]

        all_lengths = sorted(
            set().union(*(d.keys() for d in dist_gens), dist_ref.keys())
        )

        gen_matrix = np.array([[d.get(k, 0) for k in all_lengths] for d in dist_gens])
        gen_mean = gen_matrix.mean(axis=0)
        gen_std = gen_matrix.std(axis=0)

        ref_values = [dist_ref.get(k, 0) for k in all_lengths]

        color_palette = px.colors.qualitative.Safe

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=all_lengths,
            y=gen_mean,
            name=f"{gen_label}",
            error_y=dict(type="data", array=gen_std, visible=True)
        ))

        fig.add_trace(go.Bar(
            x=all_lengths,
            y=ref_values,
            name=f"{ref_label}"
        ))

        title_text = (f"Length Distribution: Generated vs. {ref_label.capitalize()} {analysis_config.receptor_type} "
                      f"Sets (Dataset {dataset_label})")
        fig.update_layout(
            barmode="group",
            title={
                'text': wrap_title(title_text),
                'font': {'size': 18}
            },
            xaxis_title="Sequence Length",
            yaxis_title="Count",
            template="plotly_white",
            colorway=color_palette,
            showlegend=True,
            xaxis=dict(
                tickvals=list(range(min(all_lengths), max(all_lengths) + 1))
            )
        )

        plot_path = Path(analysis_config.analysis_output_dir) / f"length_dist_{dataset_label}_{gen_label}_{ref_label}.png"
        fig.write_image(plot_path)
        print(f"Plot saved as PNG at: {plot_path}")


def compute_jsd_length(dist1, dist2):
    all_lengths = set(dist1.keys()).union(set(dist2.keys()))
    p = [dist1.get(k, 0) for k in all_lengths]
    q = [dist2.get(k, 0) for k in all_lengths]
    return jensenshannon(p, q, base=2)
