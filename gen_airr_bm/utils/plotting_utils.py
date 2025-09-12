import os

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc


def plot_avg_scores(mean_scores_dict, std_scores_dict, output_dir, reference_data, file_name,
                    distribution_type, scoring_method="JSD"):
    fig_dir = os.path.join(output_dir, reference_data)
    os.makedirs(fig_dir, exist_ok=True)

    models, scores = zip(*sorted(mean_scores_dict.items(), key=lambda x: x[1], reverse=True))
    errors = [std_scores_dict[model] for model in models]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=models,
        y=scores,
        error_y=dict(type='data', array=errors, visible=True),
        marker=dict(color='skyblue'),
    ))

    fig.update_layout(
        title=f"Average {scoring_method} Scores Comparing {distribution_type.capitalize()} Distributions Across Models and "
              f"{reference_data.capitalize()} Data",
        xaxis_title="Models",
        yaxis_title=f"Mean score for {distribution_type.capitalize()} Distributions",
        xaxis_tickangle=-45,
        template="plotly_white"
    )

    png_path = os.path.join(fig_dir, file_name)
    fig.write_image(png_path)

    print(f"Plot saved as PNG at: {png_path}")


def plot_grouped_avg_scores(mean_scores_by_ref, std_scores_by_ref, output_dir, reference_data, file_name,
                            distribution_type, scoring_method="JSD"):
    """
    Plots grouped bar chart for mean scores across models and reference types.

    Args:
        mean_scores_by_ref: dict of {ref_label: {model: mean_score}}
        std_scores_by_ref: dict of {ref_label: {model: std_score}}
        output_dir: output directory
        reference_data: string or list, used for subfolder naming
        file_name: output file name
        distribution_type: e.g. "cdr3"
        scoring_method: e.g. "JSD"
    """
    if isinstance(reference_data, (list, tuple)):
        ref_folder = "_".join(reference_data)
    else:
        ref_folder = str(reference_data)

    fig_dir = os.path.join(output_dir, ref_folder)
    os.makedirs(fig_dir, exist_ok=True)

    all_models = sorted({model for ref_scores in mean_scores_by_ref.values() for model in ref_scores})
    all_refs = sorted(mean_scores_by_ref.keys())

    data = []
    for ref_label in all_refs:
        means = [mean_scores_by_ref.get(ref_label, {}).get(model, None) for model in all_models]
        stds = [std_scores_by_ref.get(ref_label, {}).get(model, 0) for model in all_models]
        data.append(go.Bar(
            name=ref_label.capitalize(),
            x=all_models,
            y=means,
            error_y=dict(type='data', array=stds, visible=True),
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='group',
        title={'text': f"Avg {scoring_method} Scores Comparing {distribution_type.capitalize()} Distributions Across "
                       f"Models and References",
               'font': {'size': 14}},
        xaxis_title="Models",
        yaxis_title=f"Mean score for {distribution_type.capitalize()} Distributions",
        xaxis_tickangle=-45,
        template="plotly_white",
        colorway=pc.qualitative.Set2
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
        title=f"Comparison of Connectivity Distributions for {dataset_name}",
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

    fig.update_xaxes(range=[-0.02, 1.02], title_text="Recall")
    fig.update_yaxes(range=[-0.02, 1.02], title_text="Precision")

    png_path = os.path.join(output_dir, file_name)
    fig.write_image(png_path, scale=2)

    print(f"Plot saved as PNG at: {png_path}")


def _plot_grouped_bar(df, value_col_mean, value_col_std, all_models, all_datasets, color_palette, title, yaxis_title,
                      output_path):
    """
    Helper function to plot grouped bar chart from a DataFrame.
    """
    bars = []
    for i, dataset in enumerate(all_datasets):
        means, stds = [], []
        for model in all_models:
            row = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
            if not row.empty:
                means.append(row[f'{value_col_mean}'].values[0])
                stds.append(row[f'{value_col_std}'].values[0] if not pd.isna(row[f'{value_col_std}'].values[0]) else 0)
            else:
                means.append(0)
                stds.append(0)
        bars.append(go.Bar(
            name=dataset,
            x=all_models,
            y=means,
            error_y=dict(type='data', array=stds, visible=True, thickness=1,),
            marker_color=color_palette[i % len(color_palette)]
        ))

    fig = go.Figure(data=bars)
    fig.update_layout(
        barmode='group',
        title=title,
        xaxis_title="Model",
        yaxis_title=yaxis_title,
        xaxis_tickangle=-45,
        template="plotly_white"
    )
    fig.write_image(output_path, scale=2)
    print(f"{yaxis_title} bar chart saved as PNG at: {output_path}")


def sort_names_ignore_prefix(names):
    """ Sorts names by ignoring the first 5 characters (e.g., '0001_') """
    # TODO: this solution works specifically for current dataset names but should find another way
    return sorted(names, key=lambda x: x[5:])  # skip first 5 chars: 4 digits + "_"


def plot_grouped_bar_precision_recall(precision_scores_dict, recall_scores_dict, output_dir, reference_data,
                                      precision_file_name="precision_grouped_bar.png",
                                      recall_file_name="recall_grouped_bar.png"):
    """
    Plots two grouped bar charts: one for precision and one for recall.
    Each chart is grouped by model, with bars for each dataset.

    Args:
        precision_scores_dict: dict of {dataset: {model: [precision_scores]}}
        recall_scores_dict: dict of {dataset: {model: [recall_scores]}}
        output_dir: output directory
        reference_data: string, used for subfolder naming and title
        precision_file_name: output file name for precision chart
        recall_file_name: output file name for recall chart
    """
    fig_dir = os.path.join(output_dir, reference_data)
    os.makedirs(fig_dir, exist_ok=True)

    data = []
    for dataset in precision_scores_dict:
        for model in precision_scores_dict[dataset]:
            prec_vals = precision_scores_dict[dataset][model]
            rec_vals = recall_scores_dict[dataset][model]
            for precision, recall in zip(prec_vals, rec_vals):
                data.append({'Dataset': dataset, 'Model': model, 'Precision': precision, 'Recall': recall})

    df = pd.DataFrame(data)
    df = df.sort_values(by=["Dataset", "Model"])

    # Compute mean and std for each (Dataset, Model) pair
    grouped = df.groupby(['Dataset', 'Model']).agg(
        Precision_mean=('Precision', 'mean'),
        Precision_std=('Precision', 'std'),
        Recall_mean=('Recall', 'mean'),
        Recall_std=('Recall', 'std')
    ).reset_index()

    # Calculate average precision for each model across all datasets
    model_avg_precision = df.groupby('Model')['Precision'].mean().sort_values(ascending=False)
    all_models = list(model_avg_precision.index)
    if "upper_reference" in all_models:
        all_models = ["upper_reference"] + [m for m in all_models if m != "upper_reference"]

    all_datasets = sort_names_ignore_prefix(df['Dataset'].unique())
    color_palette = px.colors.sequential.Blues_r

    # Plot precision
    _plot_grouped_bar(
        grouped,
        value_col_mean='Precision_mean',
        value_col_std='Precision_std',
        all_models=all_models,
        all_datasets=all_datasets,
        color_palette=color_palette,
        title=f"Realism score by Model and Dataset (Reference: {reference_data})",
        yaxis_title="Realism",
        output_path=os.path.join(fig_dir, precision_file_name)
    )

    # Plot recall
    _plot_grouped_bar(
        grouped,
        value_col_mean='Recall_mean',
        value_col_std='Recall_std',
        all_models=all_models,
        all_datasets=all_datasets,
        color_palette=color_palette,
        title=f"Coverage score by Model and Dataset (Reference: {reference_data})",
        yaxis_title="Coverage",
        output_path=os.path.join(fig_dir, recall_file_name)
    )