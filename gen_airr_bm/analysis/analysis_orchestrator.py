from gen_airr_bm.analysis.analyse_aa_distribution import run_aa_distribution_analysis
from gen_airr_bm.analysis.analyse_kmer_distribution import run_kmer_distribution_analysis
from gen_airr_bm.analysis.analyse_length_distribution import run_length_distribution_analysis
from gen_airr_bm.analysis.analyse_phenotype import run_phenotype_analysis
from gen_airr_bm.analysis.analyse_pgen import run_pgen_analysis
from gen_airr_bm.analysis.analyse_novelty import run_novelty_analysis
from gen_airr_bm.core.analysis_config import AnalysisConfig


class AnalysisOrchestrator:
    """Orchestrates which analysis method to run based on the config."""
    ANALYSES_METHODS = {
        "phenotype": run_phenotype_analysis,
        "pgen": run_pgen_analysis,
        "length_distribution": run_length_distribution_analysis,
        "kmer_distribution": run_kmer_distribution_analysis,
        "aminoacid_distribution": run_aa_distribution_analysis,
        "novelty": run_novelty_analysis,
    }

    def run_analysis(self, analysis_config: AnalysisConfig):
        """Runs the appropriate analysis based on config."""
        analysis = analysis_config.analysis
        if analysis not in self.ANALYSES_METHODS:
            raise ValueError(f"Unknown analysis type: {analysis}")

        print(f"Running analysis: {analysis}")
        return self.ANALYSES_METHODS[analysis](analysis_config)
