import pytest
import pandas as pd
from unittest import TestCase
from scripts.utils import get_shared_region_type


class TestUtils(TestCase):
    def test_get_shared_region_type_junction(self):
        df1 = pd.DataFrame({'junction_aa': ['ACDEFG', 'CDEFGH', 'EFGHIK', 'FGHIKL']})
        df2 = pd.DataFrame({'junction_aa': ['ACDEFG', 'CDEFGH', 'EFGHIK', 'FGHIKL']})
        self.assertEqual(get_shared_region_type(df1, df2), 'junction_aa')

    def test_get_shared_region_type_cdr3(self):
        df1 = pd.DataFrame({'cdr3_aa': ['ACDE', 'CDEF', 'DEFG']})
        df2 = pd.DataFrame({'cdr3_aa': ['ACDE', 'CDEF', 'DEFG']})
        self.assertEqual(get_shared_region_type(df1, df2), 'cdr3_aa')

    def test_get_shared_region_type_both(self):
        df1 = pd.DataFrame({'junction_aa': ['ACDEFG', 'CDEFGH', 'EFGHIK', 'FGHIKL'],
                            'cdr3_aa': ['CDEF', 'DEFG', 'FGHI', 'GHIK']})
        df2 = pd.DataFrame({'junction_aa': ['ACDEFG', 'CDEFGH', 'EFGHIK', 'FGHIKL'],
                            'cdr3_aa': ['CDEF', 'DEFG', 'FGHI', 'GHIK']})
        self.assertEqual(get_shared_region_type(df1, df2), 'junction_aa')

    def test_get_shared_region_type_empty_junction(self):
        df1 = pd.DataFrame({'junction_aa': ['', '', '', ''],
                            'cdr3_aa': ['CDEF', 'DEFG', 'FGHI', 'GHIK']})
        df2 = pd.DataFrame({'junction_aa': ['', '', '', ''],
                            'cdr3_aa': ['CDEF', 'DEFG', 'FGHI', 'GHIK']})
        self.assertEqual(get_shared_region_type(df1, df2), 'cdr3_aa')

    def test_get_shared_region_type_no_shared(self):
        df1 = pd.DataFrame({'junction_aa': ['ACDEFG', 'CDEFGH', 'EFGHIK', 'FGHIKL']})
        df2 = pd.DataFrame({'cdr3_aa': ['ACDE', 'CDEF', 'DEFG']})
        with pytest.raises(ValueError):
            get_shared_region_type(df1, df2)
