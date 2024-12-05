import pandas as pd


def filter_by_cdr3_length(data_file, output_file, sequence_length, region_type='IMGT_CDR3'):
    """
    Generate new filtered data file for given sequence length.
    """
    data = pd.read_csv(data_file, sep='\t')

    # TO DO fix hack: standardize sequence extraction by column name
    sequence_aa = 'cdr3_aa' if region_type == 'IMGT_CDR3' else 'junction_aa'

    # Filter on sequence length
    data[data[sequence_aa].apply(len) == int(sequence_length)].to_csv(output_file, sep='\t', index=False)
