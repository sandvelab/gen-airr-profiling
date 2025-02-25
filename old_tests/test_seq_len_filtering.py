import pandas as pd
import numpy as np
import tempfile
import os
from unittest import TestCase

from old_scripts.seq_len_filtering import filter_by_cdr3_length


class TestSeqLenFiltering(TestCase):
    def check_filtering(self, test_data, expected_data):
        # Create temporary input and output files
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv') as input_file, \
                tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv') as output_file:
            # Write test data to the input file
            pd.DataFrame(test_data).to_csv(input_file.name, sep='\t', index=False)

            # Run the function with the specified sequence length
            filter_by_cdr3_length(input_file.name, output_file.name, sequence_length=6)

            # Read the output file and check it matches the expected result
            output_df = pd.read_csv(output_file.name, sep='\t')
            expected_df = pd.DataFrame(expected_data)

            # Assert that the output matches expected data
            pd.testing.assert_frame_equal(output_df, expected_df)

        # Clean up temporary files
        os.remove(input_file.name)
        os.remove(output_file.name)

    def test_filter_by_cdr3_length_imgt_junction(self):
        # Sample data
        test_data = {
            'junction_aa': ['CAAGGF', 'CGGTTTF', 'CCCCCF', 'CGGAAF'],
            'cdr3_aa': ['AAGG', 'GGTTT', 'CCCC', 'GGAA'],
            'other_column': [1, 2, 3, 4]  # Additional column to test preservation
        }

        # Expected output when filtering for sequences of length 4
        expected_data = {
            'junction_aa': ['CAAGGF', 'CCCCCF', 'CGGAAF'],
            'cdr3_aa': ['AAGG', 'CCCC', 'GGAA'],
            'other_column': [1, 3, 4]
        }

        self.check_filtering(test_data, expected_data)

    def test_filter_by_cdr3_length_imgt_cdr3(self):
        # Sample data
        test_data = {
            'junction_aa': [np.nan, np.nan, np.nan, np.nan],
            'cdr3_aa': ['AAGG', 'GGTTT', 'CCCC', 'GGAA'],
            'other_column': [1, 2, 3, 4]  # Additional column to test preservation
        }

        # Expected output when filtering for sequences of length 4
        expected_data = {
            'junction_aa': [np.nan, np.nan, np.nan],
            'cdr3_aa': ['AAGG', 'CCCC', 'GGAA'],
            'other_column': [1, 3, 4]
        }

        self.check_filtering(test_data, expected_data)