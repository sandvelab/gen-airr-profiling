import os

import pandas as pd
import plotly.graph_objects as go


def plot_jsd_scores(mean_divergence_scores_dict, std_divergence_scores_dict, output_dir, reference_data, file_name,
                    distribution_type):
    fig_dir = os.path.join(output_dir, reference_data)
    os.makedirs(fig_dir, exist_ok=True)

    models, scores = zip(*sorted(mean_divergence_scores_dict.items(), key=lambda x: x[1]))
    errors = [std_divergence_scores_dict[model] for model in models]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=models,
        y=scores,
        error_y=dict(type='data', array=errors, visible=True),
        marker=dict(color='skyblue'),
    ))

    fig.update_layout(
        title=f"JSD Scores Comparing {distribution_type.capitalize()} Distributions Across Models and "
              f"{reference_data.capitalize()} Data",
        xaxis_title="Models",
        yaxis_title=f"Mean JSD for {distribution_type.capitalize()} Distributions",
        xaxis_tickangle=-45,
        template="plotly_white"
    )

    png_path = os.path.join(fig_dir, file_name)
    fig.write_image(png_path)

    print(f"Plot saved as PNG at: {png_path}")


def plot_degree_distribution(gen_node_degree_distribution, ref_node_degree_distribution, output_dir, model_name, reference_data,
                    dataset_name):
    """Plot histograms of the two node degree distributions in one plot."""
    fig_dir = os.path.join(output_dir, reference_data)
    os.makedirs(fig_dir, exist_ok=True)

    merged_df = pd.merge(gen_node_degree_distribution, ref_node_degree_distribution, how='outer',
                         suffixes=('_gen', '_ref'),
                         left_index=True, right_index=True).fillna(0)

    merged_df["freq_gen"] = merged_df["count_gen"] / merged_df["count_gen"].sum()
    merged_df["freq_ref"] = merged_df["count_ref"] / merged_df["count_ref"].sum()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=merged_df.index,
        y=merged_df["freq_gen"],
        name=model_name,
        marker=dict(color='skyblue')
    ))

    fig.add_trace(go.Bar(
        x=merged_df.index,
        y=merged_df["freq_ref"],
        name=reference_data,
        marker=dict(color='orange')
    ))

    fig.update_layout(
        title="Comparison of Connectivity Distributions",
        xaxis_title="Number of neighbors",
        yaxis_title="Frequency",
        barmode="group"
    )

    png_path = f"{fig_dir}/histogram_{dataset_name}_{model_name}_{reference_data}.png"
    fig.write_image(png_path)

    print(f"Plot saved as PNG at: {png_path}")


def plot_diversity_bar_chart(mean_diversity, std_diversity, output_path):
    labels = list(mean_diversity.keys())
    means = [mean_diversity[label] for label in labels]
    errors = [std_diversity.get(label, 0) for label in labels]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=means,
                error_y=dict(type='data', array=errors, visible=True),
                text=[f"{val:.2f}" for val in means],
                textposition='auto'
            )
        ]
    )

    fig.update_layout(
        title="Mean Shannon Diversity with Error Bars",
        xaxis_title="Dataset/Model",
        yaxis_title="Shannon Diversity",
        template="plotly_white"
    )

    fig.write_image(output_path)
