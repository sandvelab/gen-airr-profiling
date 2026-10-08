import os
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc

from gen_airr_bm.core.analysis_config import AnalysisConfig


def get_collection_specification_for_title(receptor_type, collection=None):
    """ Returns the collection label used in plot titles, e.g. "Collection B (TCR)".
    An explicit collection (e.g. "D" for the Emerson repertoires) takes precedence, since several collections
    share a receptor type. Without one, the collection is inferred from the receptor type.
    """
    if collection is not None:
        return f"Collection {collection} ({receptor_type.replace(' UMI', '')})"
    elif receptor_type == "TCR":
        return "Collection B (TCR)"
    elif receptor_type == "BCR":
        return "Collection C (BCR)"
    elif receptor_type == "BCR UMI":
        return "Collection A (BCR)"
    else:
        raise ValueError(f"Unknown receptor type: {receptor_type}")


def title_case(text: str) -> str:
    """ Title-cases the words of a text, leaving words that already contain capitals as they are, so that
    abbreviations survive: "vj pairing" becomes "Vj Pairing", but "VJ pairing" becomes "VJ Pairing".
    """
    return " ".join(word if any(char.isupper() for char in word) else word.title() for word in text.split(" "))


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

    fig.write_image(png_path, scale=3)
    print(f"Plot saved as png at: {png_path}.png")


#TODO: Refactor this function (Dirty coding)
def plot_avg_innovation_scores(analysis_config, mean_scores_dict, std_scores_dict, output_dir, reference_data, file_name,
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
    color_palette = px.colors.qualitative.Safe

    fig = go.Figure(
        go.Bar(
            x=plotting_df["Model"],
            y=plotting_df["Mean_Score"],
            error_y=dict(type="data", array=plotting_df["Std_Dev"], visible=True),
            marker=dict(color="skyblue"),
        )
    )

    collection_specification = get_collection_specification_for_title(analysis_config.receptor_type,
                                                                     analysis_config.collection)
    fig.update_layout(
        title=wrap_title(f"Mean Unique Innovation Score for Generated {collection_specification} Repertoires"),
        xaxis_title="Models",
        yaxis_title=f"Mean Unique Innovation Score",
        xaxis_tickangle=-45,
        template="plotly_white",
        colorway=color_palette
    )

    fig.write_image(png_path, scale=3)
    print(f"Plot saved as png at: {png_path}.png")


DONOR_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "x", "cross", "star", "triangle-down", "pentagon",
                 "hexagon"]


def get_donor_symbols(donors) -> dict | None:
    """ Assigns a marker symbol to each donor, or returns None if no donor occurs more than once, since symbols only
    help to see which repertoires come from the same donor. """
    donors = list(donors)
    if len(set(donors)) == len(donors):
        return None
    return {donor: DONOR_SYMBOLS[i % len(DONOR_SYMBOLS)] for i, donor in enumerate(sorted(set(donors)))}


def plot_grouped_avg_scores(analysis_config: AnalysisConfig, mean_scores_by_ref, std_scores_by_ref,  file_name,
                            distribution_type, scoring_method="JSD", reference_score=None, error_bars_by_ref=None,
                            error_bar_caption=None, repertoire_scores_by_ref=None) -> None:
    """
    Plots grouped bar chart for mean scores across models and reference types.

    Args:
        analysis_config: AnalysisConfig object containing analysis settings
        mean_scores_by_ref: dict of {ref_label: {model: mean_score}}
        std_scores_by_ref: dict of {ref_label: {model: std_score}}, used for the error bars if error_bars_by_ref is
            not given
        file_name: output file name without extension
        distribution_type: used for titles. e.g. "connectivity"
        scoring_method: used for titles. e.g. "JSD"
        reference_score: optional float, to plot a reference line
        error_bars_by_ref: optional dict of {ref_label: {model: (lower, upper)}}, plotted as error bars instead of the
            standard deviation, e.g. a confidence interval
        error_bar_caption: optional caption that explains the error bars
        repertoire_scores_by_ref: optional dict of {ref_label: {model: list of (donor, score)}}, plotted as one dot
            per repertoire, with one marker symbol per donor if donors have several repertoires
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
    if error_bars_by_ref is not None:
        plotting_df["Error_Lower"] = [error_bars_by_ref.get(ref, {}).get(model, (np.nan, np.nan))[0]
                                      for ref, model in zip(plotting_df["Reference"], plotting_df["Model"])]
        plotting_df["Error_Upper"] = [error_bars_by_ref.get(ref, {}).get(model, (np.nan, np.nan))[1]
                                      for ref, model in zip(plotting_df["Reference"], plotting_df["Model"])]

    plotting_df.to_csv(plotting_data_file, sep="\t", index=False)

    data = []
    for ref in all_refs:
        ref_df = plotting_df[plotting_df["Reference"] == ref]
        if error_bars_by_ref is not None:
            error_y = dict(type="data", array=ref_df["Error_Upper"] - ref_df["Mean_Score"],
                           arrayminus=ref_df["Mean_Score"] - ref_df["Error_Lower"], visible=True)
        else:
            error_y = dict(type="data", array=ref_df["Std_Dev"], visible=True)
        data.append(go.Bar(name=ref.capitalize(), x=ref_df["Model"], y=ref_df["Mean_Score"], error_y=error_y,
                           offsetgroup=ref))
    if repertoire_scores_by_ref is not None:
        data.extend(get_repertoire_dot_traces(repertoire_scores_by_ref, all_refs, all_models))

    fig = go.Figure(data=data)
    color_palette = px.colors.qualitative.Safe
    title_text = (f"{title_case(distribution_type)} Distribution Comparison:<br>Generated vs. Train and "
                  f"Test {get_collection_specification_for_title(receptor_type, analysis_config.collection)} Repertoires")
    fig.update_layout(
        barmode='group',
        title={'text': title_text,
               'font': {'size': 20}},
        xaxis=dict(
            title=dict(text="Model", font=dict(size=20)),
            tickangle=-45,
            tickfont=dict(size=18)
        ),
        yaxis=dict(
            title=dict(text=f"Mean {scoring_method}", font=dict(size=20)),
            tickfont=dict(size=18)
        ),
        template="plotly_white",
        colorway=color_palette,
        showlegend=True,
        scattermode="group",
    )
    if error_bar_caption is not None:
        fig.add_annotation(text=error_bar_caption, xref="paper", yref="paper", x=1, y=1.02, showarrow=False,
                           xanchor="right", yanchor="bottom", font=dict(size=14))

    if reference_score is not None:
        fig.add_hline(
            y=reference_score,
            line=dict(color="black", dash="dash"),
            annotation_text=f"Train vs. Test = {reference_score:.3f}",
            annotation_position="top right",
            annotation_font=dict(size=18, color="black"),
            annotation=dict(yshift=14)
        )

    fig.write_image(png_path, scale=3)
    print(f"Plot saved as png at: {png_path}")


def get_repertoire_dot_traces(repertoire_scores_by_ref: dict, refs: list, models: list) -> list:
    """ Returns scatter traces with one dot per repertoire, placed on the bars of their reference. If donors have
    several repertoires, each donor gets its own marker symbol and legend entry. """
    points = [(ref, model, donor, score) for ref in refs for model in models
              for donor, score in repertoire_scores_by_ref.get(ref, {}).get(model, [])]
    first_ref_model = [(donor, model) for ref, model, donor, _ in points if ref == refs[0]]
    donor_symbols = get_donor_symbols(donor for donor, model in first_ref_model if model == first_ref_model[0][1]) \
        if first_ref_model else None

    traces = []
    for ref in refs:
        donors = sorted(donor_symbols) if donor_symbols else [None]
        for donor in donors:
            donor_points = [(model, score) for point_ref, model, point_donor, score in points
                            if point_ref == ref and (donor is None or point_donor == donor)]
            if not donor_points:
                continue
            traces.append(go.Scatter(
                x=[model for model, _ in donor_points], y=[score for _, score in donor_points], mode="markers",
                offsetgroup=ref, name=f"Donor {donor}", legendgroup=f"donor_{donor}",
                showlegend=donor is not None and ref == refs[0], hoverinfo="y",
                marker=dict(color="black", size=6, opacity=0.6,
                            symbol=donor_symbols[donor] if donor is not None else "circle")))
    return traces


def plot_paired_comparisons(comparisons_df: pd.DataFrame, output_path: str, title: str, x_label: str,
                            differences_df: pd.DataFrame = None) -> None:
    """ Plots paired comparisons as a forest plot, one row per comparison, labelled with the number of repertoires
    (and donors) in which the first source is lower. The labels assume lower scores are better, as for divergences.
    With statistical tests (comparisons_df has p-values), each row shows the mean difference with its confidence
    interval and the adjusted p-value. Without, each row shows the per-repertoire differences and their mean.
    Args:
        comparisons_df (pd.DataFrame): Comparisons as returned by statistics_utils.compare_paired.
        output_path (str): Output path without extension.
        title (str): Plot title.
        x_label (str): Label of the x-axis, e.g. "Difference in JSD".
        differences_df (pd.DataFrame): Per-repertoire differences as returned by statistics_utils.compare_paired,
            plotted as dots without statistical tests.
    Returns:
        None
    """
    if comparisons_df.empty:
        return
    statistical_tests = "p_adjusted" in comparisons_df.columns
    plotting_df = comparisons_df.iloc[::-1]
    labels = [f"{first} − {second}" for first, second in zip(plotting_df["first"], plotting_df["second"])]

    traces = []
    if statistical_tests:
        annotations = [f"lower in {row.n_first_lower}/{row.n_repertoires}, p<sub>adj</sub> = {row.p_adjusted:.2g}"
                       for row in plotting_df.itertuples()]
        traces.append(go.Scatter(
            x=plotting_df["mean_difference"], y=labels, mode="markers", showlegend=False,
            marker=dict(color="black", size=9),
            error_x=dict(type="data", array=plotting_df["ci_high"] - plotting_df["mean_difference"],
                         arrayminus=plotting_df["mean_difference"] - plotting_df["ci_low"], visible=True)))
    else:
        annotations = [f"lower in {row.n_first_lower}/{row.n_repertoires} repertoires"
                       + (f" ({row.n_donors_first_lower}/{row.n_donors} donors)" if row.n_donors < row.n_repertoires
                          else "")
                       for row in plotting_df.itertuples()]
        if differences_df is not None and not differences_df.empty:
            labelled = differences_df.assign(label=[f"{first} − {second}" for first, second
                                                    in zip(differences_df["first"], differences_df["second"])])
            first_comparison = labelled[labelled["label"] == labelled["label"].iloc[0]]
            donor_symbols = get_donor_symbols(first_comparison["donor"])
            for donor, donor_df in (labelled.groupby("donor") if donor_symbols else [(None, labelled)]):
                traces.append(go.Scatter(
                    x=donor_df["difference"], y=donor_df["label"], mode="markers", name=f"Donor {donor}",
                    showlegend=donor is not None,
                    marker=dict(color="grey", size=8, opacity=0.7,
                                symbol=donor_symbols[donor] if donor is not None else "circle")))
        traces.append(go.Scatter(x=plotting_df["mean_difference"], y=labels, mode="markers", name="Mean",
                                 marker=dict(color="black", size=22, symbol="line-ns", line=dict(width=3))))

    fig = go.Figure(traces)
    fig.add_vline(x=0, line=dict(color="grey", dash="dash"))
    # The statistics are written in a column to the right of the plot, so they never overlap the intervals
    for label, annotation in zip(labels, annotations):
        fig.add_annotation(text=annotation, x=1.02, xref="paper", y=label, yref="y", xanchor="left",
                           showarrow=False, font=dict(size=13))
    fig.update_layout(title={'text': wrap_title(title, width=80), 'font': {'size': 18}},
                      xaxis=dict(title=dict(text=x_label, font=dict(size=16)), tickfont=dict(size=14)),
                      yaxis=dict(tickfont=dict(size=14), categoryorder="array", categoryarray=labels),
                      template="plotly_white", showlegend=not statistical_tests, margin=dict(r=320, b=130),
                      legend=dict(orientation="h", x=0, xanchor="left", y=-0.25, yanchor="top"),
                      height=max(400, 60 * len(plotting_df) + 260), width=1300)
    fig.write_image(output_path + ".png", scale=2)
    print(f"Plot saved as png at: {output_path}.png")


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

    fig.write_image(output_path, scale=3)


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

    fig.write_image(output_path, scale=3)
    print(f"Grouped Precision/Recall bar chart saved as png at: {output_path}")


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

            distance_type = "Levenshtein" if analysis_config.indels else "Hamming"
            dataset_name_clean = dataset_name.rsplit("_", 1)[0]
            collection_specification = get_collection_specification_for_title(analysis_config.receptor_type,
                                                                                 analysis_config.collection)
            fig.update_layout(
                width=1600,
                height=800,
                title={"text": wrap_title(f"Connectivity Distribution: Generated vs. Train and Test "
                          f"{collection_specification} Repertoire (Dataset: {dataset_name_clean})", width=60),
                          "font": {"size": 40},
                           'y': 0.95,
                           'yanchor': 'top'
                           },
                margin=dict(t=100),
                xaxis_title={"text": f"Neighbor Count ({distance_type} Distance: {analysis_config.allowed_mismatches})",
                             "font": {"size": 30}},
                yaxis_title={"text": "Frequency (log scale)",
                             "font": {"size": 30}},
                yaxis_type="log",
                barmode="group",
                template="plotly_white",
                colorway=px.colors.qualitative.Safe[:2] + px.colors.qualitative.Safe[7:],  # skip colors to avoid confusion
                bargroupgap=0.15,
                xaxis=dict(tickfont=dict(size=25)),
                yaxis=dict(tickfont=dict(size=25)),
                legend=dict(font=dict(size=25))
            )

            fig.write_image(png_path, scale=3)
            print(f"Plot saved as png at: {png_path}")
