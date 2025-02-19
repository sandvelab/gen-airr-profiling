import pandas as pd


def filter_by_cdr3_length(data_file, output_file, sequence_length):
    """
    Generate new filtered data file for given sequence length.
    """
    data = pd.read_csv(data_file, sep='\t')

    # TO DO: decide if junction should be default and if len-2 for cdr3
    if any(pd.isnull(data['junction_aa'])):
        data[data['cdr3_aa'].apply(len) == int(sequence_length)-2].to_csv(output_file, sep='\t', index=False)
    else:
        data[data['junction_aa'].apply(len) == int(sequence_length)].to_csv(output_file, sep='\t', index=False)
