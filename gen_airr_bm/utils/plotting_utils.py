import os

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
        yaxis_title="Mean JSD for Length Distributions",
        xaxis_tickangle=-45,
        template="plotly_white"
    )

    png_path = os.path.join(fig_dir, file_name)
    fig.write_image(png_path)

    print(f"Plot saved as PNG at: {png_path}")