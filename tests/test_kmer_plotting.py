from unittest import TestCase
import shutil

from gen_airr_bm.scripts.kmer_freq_plotting import get_kmer_counts, find_significantly_different_kmers, run_kmer_analysis
from immuneML.util.PathBuilder import PathBuilder
from immuneML.simulation.dataset_generation.RandomDatasetGenerator import RandomDatasetGenerator
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.IO.dataset_export.AIRRExporter import AIRRExporter


class TestKmerPlotting(TestCase):
    def test_get_kmer_counts(self):
        data_seqs = ['AAAT', 'GAAT']
        kmer_counts = get_kmer_counts(data_seqs, 3)
        expected_kmer_counts = {
            'AAA': 1, 'AAT': 2, 'GAA': 1
        }
        self.assertDictEqual(kmer_counts, expected_kmer_counts)

    def test_find_significantly_different_kmers(self):
        data_seqs1 = ['AAAT', 'GGAA', 'CGVT']
        data_seqs2 = ['CGVA', 'GVAC', 'AATG']
        kmer_counts1 = get_kmer_counts(data_seqs1, 3)
        kmer_counts2 = get_kmer_counts(data_seqs2, 3)
        kmer_comparison_df, significantly_different_kmers = find_significantly_different_kmers(kmer_counts1, 'data1',
                                                                           kmer_counts2, 'data2',
                                                                           3, 1)
        self.assertCountEqual(kmer_comparison_df['kmer'], ['CGV', 'AAT'])
        self.assertEqual(significantly_different_kmers.shape[0], 0)

    def test_find_significantly_different_kmers_empty(self):
        data_seqs1 = ['AAAT', 'GGAA', 'CGVT']
        data_seqs2 = ['CGVA', 'GVAC', 'AATG']
        kmer_counts1 = get_kmer_counts(data_seqs1, 3)
        kmer_counts2 = get_kmer_counts(data_seqs2, 3)
        kmer_comparison_df, significantly_different_kmers = find_significantly_different_kmers(kmer_counts1, 'data1',
                                                                           kmer_counts2, 'data2',
                                                                           4, 1)
        self.assertEqual(kmer_comparison_df.shape[0], 0)
        self.assertIsNone(significantly_different_kmers)

    def test_run_kmer_analysis(self):
        path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / 'kmer_analysis')
        dataset1 = RandomDatasetGenerator.generate_sequence_dataset(50, {4: 0.33, 5: 0.33, 7: 0.33}, {},
                                                                    path / 'dataset1')
        dataset2 = RandomDatasetGenerator.generate_sequence_dataset(50, {4: 0.33, 5: 0.33, 7: 0.33}, {},
                                                                    path / 'dataset2')
        path_exported_1 = path / "dataset1"
        path_exported_2 = path / "dataset2"
        AIRRExporter.export(dataset1, path_exported_1)
        AIRRExporter.export(dataset2, path_exported_2)

        dataset1_path = path_exported_1 / "sequence_dataset.tsv"
        dataset2_path = path_exported_2 / "sequence_dataset.tsv"

        run_kmer_analysis(dataset1_path, 'name1', dataset2_path, 'name2', path, k=3, kmer_count_threshold=1)

        self.assertTrue((path / 'kmer_comparison.html').exists())
        self.assertTrue((path / 'repeat_counts.tsv').exists())
        self.assertTrue((path / 'repeat_counts_fisher_exact_test_pval.txt').exists())

        shutil.rmtree(path)



