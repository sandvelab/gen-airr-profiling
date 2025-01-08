import pandas as pd


def add_junction_from_cdr3(tsv_path_in, tsv_path_out):
    """
    TEMPORARY FUNCTION:
    Add junction_aa sequences with constant junction positions to tsv based cdr3_aa.
    SONIA and soNNia require junction positions for input and generates junction_aa.
    Currently implemented experimental datasets (Mason and Emerson) only includes cdr3.
    """
    df = pd.read_csv(tsv_path_in, sep='\t')

    if 'emerson' in tsv_path_in:
        df['junction_aa'] = 'C' + df['cdr3_aa'] + 'W'
    elif 'mason' in tsv_path_in:
        df['junction_aa'] = 'CSR' + df['cdr3_aa'] + 'YW'

    df.to_csv(tsv_path_out, sep='\t', index=False)

