from unittest import TestCase
import pandas as pd

from scripts.aa_freq_plotting import get_aa_counts_frequencies_df


class TestAAFreqPlotting(TestCase):
    def test_get_aa_counts_frequencies_df(self):
        data = {
            'junction_aa': ['ACDEFG', 'CDEFGH', 'EFGHIK', 'FGHIKL']
        }
        df = pd.DataFrame(data)
        max_position = 6
        result_df, num_sequences = get_aa_counts_frequencies_df(df, max_position)

        self.assertEqual(num_sequences, len(df))
        assert not result_df.empty
        assert result_df.loc[(result_df['amino acid'] == 'A') & (result_df['position'] == 0), 'count'].iloc[0] == 1
        assert result_df.loc[(result_df['amino acid'] == 'C') & (result_df['position'] == 0), 'count'].iloc[0] == 1
        assert result_df.loc[(result_df['amino acid'] == 'A') & (result_df['position'] == 0), 'relative frequency'].iloc[0] == 1 / 4

    def test_get_aa_counts_frequencies_df_empty(self):
        df = pd.DataFrame({'junction_aa': []})
        result_df, num_sequences = get_aa_counts_frequencies_df(df, 6)
        self.assertIsNone(result_df)
        self.assertEqual(num_sequences, 0)






