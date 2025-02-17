import pandas as pd


def subsample_data(input_file, output_file, sample_size: int = 1000, seed: int = 42):

    df = pd.read_csv(input_file)

    if len(df) < sample_size:
        print(f"Input file has only {len(df)} rows, sampling all rows.")
        sampled_df = df
    else:
        sampled_df = df.sample(n=sample_size, random_state=seed)

    sampled_df.to_csv(output_file, index=False)


def main():
    input_file = "../results/dataset1/simulations/train/simulation_0/dataset/simulated_dataset.tsv"
    output_file = "dataset1_subsampled.tsv"
    subsample_data(input_file, output_file, 50)


if __name__ == "__main__":
    main()
