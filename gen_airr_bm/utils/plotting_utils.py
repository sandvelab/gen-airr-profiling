import os

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_jsd_scores(mean_divergence_scores_dict, std_divergence_scores_dict, output_dir, reference_data, file_name,
                    distribution_type):
    fig_dir = os.path.join(output_dir, reference_data)
    os.makedirs(fig_dir, exist_ok=True)

    models, scores = zip(*sorted(mean_divergence_scores_dict.items(), key=lambda x: x[0]))
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


def plot_degree_distribution(ref_node_degree_distribution, gen_node_degree_distributions, output_dir, model_name, reference_data,
                              dataset_name):
    """Plot histograms of the two node degree distributions in one plot (with error bars for generated)."""
    fig_dir = os.path.join(output_dir, reference_data)
    os.makedirs(fig_dir, exist_ok=True)

    # Normalize the reference distribution
    ref_freq = ref_node_degree_distribution / ref_node_degree_distribution.sum()
    ref_freq = ref_freq.rename("freq_ref").to_frame()

    # Normalize and collect all generated distributions
    freq_dfs = []
    for i, dist in enumerate(gen_node_degree_distributions):
        norm = dist / dist.sum()
        freq_dfs.append(norm.rename(f"freq_{i}").to_frame())

    gen_merged = pd.concat(freq_dfs, axis=1).fillna(0)
    gen_merged["freq_gen"] = gen_merged.mean(axis=1)
    gen_merged["std_gen"] = gen_merged.std(axis=1)

    merged_df = pd.merge(ref_freq, gen_merged[["freq_gen", "std_gen"]], left_index=True, right_index=True, how='outer').fillna(0)
    merged_df = merged_df.sort_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=merged_df.index,
        y=merged_df["freq_gen"],
        name=model_name,
        marker=dict(color='skyblue'),
        error_y=dict(type='data', array=merged_df["std_gen"], visible=True)
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
        yaxis_title="Frequency (log scale)",
        yaxis_type="log",
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


def plot_scatter_precision_recall(precision_scores_dict, recall_scores_dict, output_dir, reference_data, file_name,
                                  plot_mean=False):
    """ Plot scatter plot of precision vs recall, colored by dataset and shaped by model. """

    data = []
    for dataset in precision_scores_dict:
        for model in precision_scores_dict[dataset]:

            if plot_mean:
                precision_scores_dict[dataset][model] = [precision_scores_dict[dataset][model]]
                recall_scores_dict[dataset][model] = [recall_scores_dict[dataset][model]]

            for precision, recall in zip(precision_scores_dict[dataset][model], recall_scores_dict[dataset][model]):
                data.append({
                    'Dataset': dataset,
                    'Model': model,
                    'Precision': precision,
                    'Recall': recall
                })

    df = pd.DataFrame(data)
    df = df.sort_values(by=["Dataset", "Model"])

    fig = px.scatter(
        df,
        x="Recall",
        y="Precision",
        color="Dataset",
        symbol="Model",
        title=f"Mean Precision vs Recall: Colored by Dataset, Shaped by Model (Reference: {reference_data})" if
        plot_mean else f"Precision vs Recall: Colored by Dataset, Shaped by Model (Reference: {reference_data})"
    )

    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='dark gray')))
    fig.update_layout(
        width=1000,
        height=700,
        legend_title_text='Dataset, model'
    )

    fig.update_xaxes(range=[-0.02, 0.5], title_text="Recall")
    fig.update_yaxes(range=[-0.02, 0.5], title_text="Precision")

    png_path = os.path.join(output_dir, file_name)
    fig.write_image(png_path, scale=2)

    print(f"Plot saved as PNG at: {png_path}")
