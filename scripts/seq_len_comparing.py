import argparse
import pandas as pd
import plotly.express as px
from scipy.stats import entropy
import numpy as np
from scipy.stats import gaussian_kde


def plot_seq_len_distributions(simulated_file, model_file, image_file, model_name):
    data1 = pd.read_csv(simulated_file)
    data2 = pd.read_csv(model_file)

    # Convert counts to frequencies
    for df in [data1, data2]:
        df['counts'] = df['counts'] / df['counts'].sum()
        df.rename(columns={'counts': 'frequencies'}, inplace=True)

    df_combine = {"Simulated (train)": data1, f"Generated ({model_name})": data2}
    df_combine = pd.concat(df_combine, names=["dataset"]).reset_index(level=0)

    # Create distribution plot with px
    figure = px.bar(df_combine, x="sequence_lengths", y="frequencies", color='dataset')

    figure.update_layout(barmode='group', xaxis=dict(tickmode='array', tickvals=df_combine["sequence_lengths"]),
                         yaxis=dict(tickmode='array'),
                         template="plotly_white", title=f"Sequence Length Distributions of simulated train data and generated ({model_name}) data",
                         font=dict(size=22))

    figure.write_html(image_file)
