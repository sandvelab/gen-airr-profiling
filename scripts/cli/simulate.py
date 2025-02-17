import argparse

from scripts.simulation_methods import generate_pure_olga_sequences, generate_rare_and_frequent_olga_sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_conifg", type=str, help="path to the file where the olga generated sequences will be stored")
    parser.add_argument("generated", type=str, help="path to the file with experimental data. The file should contain columns with the following names: 'sequence_aa', 'v_call', 'j_call' and should be in tsv format.")
    args = parser.parse_args()
    generate_rare_and_frequent_olga_sequences(100, "humanTRB", 1, args.data_config, "pgens_file_path.tsv",
                                              args.generated + "/A.tsv", args.generated + "/B.tsv")
