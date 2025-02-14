import pandas as pd
import numpy as np


def subsample_data(input_file, output_file, sample_size: int = 1000, seed: int = 42):

    np.random.seed(seed)

    df = pd.read_csv(input_file)

    if len(df) < sample_size:
        print(f"Input file has only {len(df)} rows, sampling all rows.")
        sampled_df = df
    else:
        sampled_df = df.sample(n=sample_size, random_state=42)

    sampled_df.to_csv(output_file, index=False)


