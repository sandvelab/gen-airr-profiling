import os
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc

from gen_airr_bm.core.analysis_config import AnalysisConfig


def plot_avg_scores(mean_scores_dict, std_scores_dict, output_dir, reference_data, file_name,
                    distribution_type, scoring_method="JSD"):
    """ Plots a bar chart for mean scores across models.
    Args:
        mean_scores_dict: dict of {model: mean_score}
        std_scores_dict: dict of {model: std_score}
        output_dir: output directory
        reference_data: string or list, used for subfolder naming
        file_name: output file name without extension
        distribution_type: used for titles. e.g. "connectivity"
        scoring_method: used for titles. e.g. "JSD"
    Returns:
        None
    """
    fig_dir = os.path.join(output_dir, reference_data)
    os.makedirs(fig_dir, exist_ok=True)
    png_path = os.path.join(fig_dir, file_name) + ".png"
    plotting_data_file = os.path.join(fig_dir, file_name) + ".tsv"

    plotting_df = pd.DataFrame({
        "Model": list(mean_scores_dict.keys()),
        "Mean_Score": list(mean_scores_dict.values()),
        "Std_Dev": [std_scores_dict.get(m, 0) for m in mean_scores_dict]
    })
    plotting_df = plotting_df.sort_values("Mean_Score", ascending=False)

    if not os.path.exists(plotting_data_file):
        plotting_df.to_csv(plotting_data_file, sep="\t", index=False)

    fig = go.Figure(
        go.Bar(
            x=plotting_df["Model"],
            y=plotting_df["Mean_Score"],
            error_y=dict(type="data", array=plotting_df["Std_Dev"], visible=True),
            marker=dict(color="skyblue"),
        )
    )

    fig.update_layout(
        title=f"Comparison of Model and Reference {distribution_type.capitalize()} Distributions",
        xaxis_title="Models",
        yaxis_title=f"Mean {scoring_method} score",
        xaxis_tickangle=-45,
        template="plotly_white"
    )

    fig.write_image(png_path)
    print(f"Plot saved as PNG at: {png_path}.png")


def plot_grouped_avg_scores(analysis_config: AnalysisConfig, mean_scores_by_ref, std_scores_by_ref,  file_name,
                            distribution_type, scoring_method="JSD", reference_score=None) -> None:
    """
    Plots grouped bar chart for mean scores across models and reference types.

    Args:
        analysis_config: AnalysisConfig object containing analysis settings
        mean_scores_by_ref: dict of {ref_label: {model: mean_score}}
        std_scores_by_ref: dict of {ref_label: {model: std_score}}
        file_name: output file name without extension
        distribution_type: used for titles. e.g. "connectivity"
        scoring_method: used for titles. e.g. "JSD"
        reference_score: optional float, to plot a reference line
    Returns:
        None
    """
    reference_data = analysis_config.reference_data
    output_dir = analysis_config.analysis_output_dir
    receptor_type = analysis_config.receptor_type
    if isinstance(reference_data, (list, tuple)):
        ref_folder = "_".join(reference_data)
    else:
        ref_folder = str(reference_data)

    fig_dir = os.path.join(output_dir, ref_folder)
    os.makedirs(fig_dir, exist_ok=True)
    png_path = os.path.join(fig_dir, file_name) + ".png"
    plotting_data_file = os.path.join(fig_dir, file_name) + ".tsv"

    all_models = sorted({model for ref_scores in mean_scores_by_ref.values() for model in ref_scores})
    all_refs = sorted(mean_scores_by_ref.keys())

    plotting_df = pd.DataFrame([{"Reference": ref,
                                 "Model": model,
                                 "Mean_Score": mean_scores_by_ref.get(ref, {}).get(model, np.nan),
                                 "Std_Dev": std_scores_by_ref.get(ref, {}).get(model, 0),
                                 "abs_diff_to_ref": (abs(mean_scores_by_ref.get(ref, {}).get(model, np.nan) -
                                                     reference_score)
                                                     if reference_score is not None else np.nan)}
                                for ref in all_refs
                                for model in all_models])

    if not os.path.exists(plotting_data_file):
        plotting_df.to_csv(plotting_data_file, sep="\t", index=False)

    data = [
        go.Bar(
            name=ref.capitalize(),
            x=plotting_df.loc[plotting_df["Reference"] == ref, "Model"],
            y=plotting_df.loc[plotting_df["Reference"] == ref, "Mean_Score"],
            error_y=dict(
                type="data",
                array=plotting_df.loc[plotting_df["Reference"] == ref, "Std_Dev"],
                visible=True,
            ),
        )
        for ref in all_refs
    ]

    fig = go.Figure(data=data)
    color_palette = px.colors.qualitative.Safe
    title_text = f"{distribution_type.title()} Distribution Comparison: Generated vs. Train and Test {receptor_type} Sets"
    fig.update_layout(
        barmode='group',
        title={'text': wrap_title(title_text),
               'font': {'size': 18}},
        xaxis_title="Model",
        yaxis_title=f"Mean {scoring_method}",
        xaxis_tickangle=-45,
        template="plotly_white",
        colorway=color_palette,
        showlegend=True,
    )

    if reference_score is not None:
        fig.add_hline(
            y=reference_score,
            line=dict(color="black", dash="dash"),
            annotation_text=f"Train vs. Test = {reference_score:.3f}",
            annotation_position="top right"
        )

    fig.write_image(png_path)
    print(f"Plot saved as PNG at: {png_path}")


def wrap_title(text, width=60):
    return "<br>".join(textwrap.wrap(text, width=width))


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


def _plot_grouped_bar(df, all_models, title, output_path):
    """
    Plot grouped bar chart per model with two bars: Precision and Recall.
    - df must contain columns: Model, Precision_mean, Precision_std, Recall_mean, Recall_std
    - Uses Plotly qualitative Safe colorway
    """
    precision_means, precision_stds = [], []
    recall_means, recall_stds = [], []

    for model in all_models:
        rows = df[df['Model'] == model]
        # Precision
        precision_means.append(rows['Precision_mean'].mean() if not rows.empty else 0)
        precision_stds.append(rows['Precision_std'].mean() if not rows.empty else 0)
        # Recall
        recall_means.append(rows['Recall_mean'].mean() if not rows.empty else 0)
        recall_stds.append(rows['Recall_std'].mean() if not rows.empty else 0)

    # Two traces: Precision and Recall
    trace_precision = go.Bar(
        x=all_models,
        y=precision_means,
        name='Precision',
        error_y=dict(type='data', array=precision_stds, visible=True, thickness=1)
    )
    trace_recall = go.Bar(
        x=all_models,
        y=recall_means,
        name='Recall',
        error_y=dict(type='data', array=recall_stds, visible=True, thickness=1)
    )

    fig = go.Figure(data=[trace_precision, trace_recall])
    fig.update_layout(
        barmode='group',
        title=title,
        xaxis_title="Model",
        yaxis_title="Score",
        xaxis_tickangle=-45,
        template="plotly_white",
        colorway=px.colors.qualitative.Safe,  # Safe palette for the bars
        legend_title_text="Metric"
    )

    fig.write_image(output_path, scale=2)
    print(f"Grouped Precision/Recall bar chart saved as PNG at: {output_path}")


def sort_names_ignore_prefix(names):
    """ Sorts names by ignoring the first 5 characters (e.g., '0001_') """
    # TODO: this solution works specifically for current dataset names but should find another way
    return sorted(names, key=lambda x: x[5:])  # skip first 5 chars: 4 digits + "_"


def plot_grouped_bar_precision_recall(precision_scores_dict, recall_scores_dict, output_dir, reference_data,
                                      receptor_type, allowed_mismatches):
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

    # Compute mean and std for each (Dataset, Model) pair
    grouped = df.groupby(['Model']).agg(
        Precision_mean=('Precision', 'mean'),
        Precision_std=('Precision', 'std'),
        Recall_mean=('Recall', 'mean'),
        Recall_std=('Recall', 'std')
    ).reset_index()

    plotting_data_file = os.path.join(fig_dir, "precision_recall_data.tsv")
    if not os.path.exists(plotting_data_file):
        grouped.to_csv(plotting_data_file, sep="\t", index=False)

    # Calculate average precision for each model across all datasets
    model_avg_precision = df.groupby('Model')['Precision'].mean().sort_values(ascending=False)
    all_models = list(model_avg_precision.index)

    # Plot precision
    _plot_grouped_bar(
        df=grouped,
        all_models=all_models,
        title=f"Mean Precision and Recall for {receptor_type} Sets (Hamming Distance: {allowed_mismatches})",
        output_path=os.path.join(fig_dir, "precision_recall.png")
    )


def plot_degree_distribution_by_dataset(analysis_config: AnalysisConfig, connectivity_distributions_all: dict):
    """Plot histograms of the node degree distributions in one plot (with error bars for generated).
    Args:
        analysis_config: AnalysisConfig object containing analysis settings
        connectivity_distributions_all:
    Returns:
        None
    """
    output_dir = analysis_config.analysis_output_dir
    ref_folder = "_".join(analysis_config.reference_data) if isinstance(analysis_config.reference_data,
                                                                        (list, tuple)) else str(analysis_config.reference_data)
    fig_dir = os.path.join(output_dir, ref_folder)
    os.makedirs(fig_dir, exist_ok=True)

    for dataset_name, model_dict in connectivity_distributions_all.items():
        for model_name, split_dict in model_dict.items():

            ref_dfs = []
            for data_split, dist_list in split_dict.items():
                if data_split == model_name:
                    freq_dfs = []
                    for i, dist in enumerate(dist_list):
                        norm = dist / dist.sum()
                        freq_dfs.append(norm.rename(f"freq_{i}").to_frame())
                    gen_merged = pd.concat(freq_dfs, axis=1).fillna(0)
                    gen_merged[f"freq_{data_split}"] = gen_merged.mean(axis=1)
                    gen_merged[f"std_{data_split}"] = gen_merged.std(axis=1)
                else:
                    assert len(dist_list) == 1, "Reference distributions should have only one entry."
                    dist = dist_list[0]
                    ref_freq = dist / dist.sum()
                    ref_freq = ref_freq.rename(f"{data_split}").to_frame()
                    ref_dfs.append(ref_freq)

            ref1_df, ref2_df = ref_dfs
            merged_df = (
                ref1_df
                .join(ref2_df, how="outer")
                .join(gen_merged[[f"freq_{model_name}", f"std_{model_name}"]], how="outer")
                .fillna(0)
                .sort_index()
            )

            png_path = f"{fig_dir}/histogram_{dataset_name}_{model_name}.png"
            tsv_path = f"{fig_dir}/histogram_{dataset_name}_{model_name}.tsv"
            if not os.path.exists(tsv_path):
                merged_df.to_csv(tsv_path, sep='\t')

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=merged_df.index,
                y=merged_df[ref1_df.columns[0]],
                name=ref1_df.columns[0],
            ))

            fig.add_trace(go.Bar(
                x=merged_df.index,
                y=merged_df[ref2_df.columns[0]],
                name=ref2_df.columns[0],
            ))

            fig.add_trace(go.Bar(
                x=merged_df.index,
                y=merged_df[f"freq_{model_name}"],
                name=model_name,
                error_y=dict(
                    type="data",
                    array=merged_df[f"std_{model_name}"],
                    visible=True
                ),
            ))

            fig.update_yaxes(
                tickvals=[10 ** i for i in range(-6, 3)],
                ticktext=[str(10 ** i) for i in range(-6, 3)],
            )

            dataset_name_clean = dataset_name.rsplit("_", 1)[0]
            fig.update_layout(
                width=1800,
                height=900,
                title={"text": f"Connectivity Distribution: Generated vs. Train and Test "
                      f"{analysis_config.receptor_type} Sets (Dataset {dataset_name_clean})",
                      "font": {"size": 20}},
                xaxis_title=f"Neighbor Count (Hamming Distance: {analysis_config.allowed_mismatches})",
                yaxis_title="Frequency (log scale)",
                yaxis_type="log",
                barmode="group",
                template="plotly_white",
                colorway=px.colors.qualitative.Safe[:2] + px.colors.qualitative.Safe[7:],  # skip colors to avoid confusion
                bargroupgap=0.15
            )

            fig.write_image(png_path)
            print(f"Plot saved as PNG at: {png_path}")
