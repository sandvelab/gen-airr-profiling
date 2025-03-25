from gen_airr_bm.analysis.analyse_connectivity import run_connectivity_analysis
from gen_airr_bm.analysis.analyse_phenotype import run_phenotype_analysis
from gen_airr_bm.analysis.analyse_pgen import run_pgen_analysis
from gen_airr_bm.analysis.analyse_novelty import run_novelty_analysis
from gen_airr_bm.analysis.analyse_reduced_dimensionality import run_reduced_dimensionality_analyses
from gen_airr_bm.core.analysis_config import AnalysisConfig


class AnalysisOrchestrator:
    """Orchestrates which analysis method to run based on the config."""
    ANALYSES_METHODS = {
        "phenotype": run_phenotype_analysis,
        "pgen": run_pgen_analysis,
        "reduced_dimensionality": run_reduced_dimensionality_analyses,
        "novelty": run_novelty_analysis,
        "connectivity": run_connectivity_analysis
    }

    def run_analysis(self, analysis_config: AnalysisConfig):
        """Runs the appropriate analysis based on config."""
        analysis = analysis_config.analysis
        if analysis not in self.ANALYSES_METHODS:
            raise ValueError(f"Unknown analysis type: {analysis}")

        print(f"Running analysis: {analysis}")
        return self.ANALYSES_METHODS[analysis](analysis_config)
