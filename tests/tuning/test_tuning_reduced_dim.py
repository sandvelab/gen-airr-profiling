import pandas as pd
import pytest

# We'll import the module under test
from gen_airr_bm.tuning import tuning_reduced_dim
from gen_airr_bm.core.tuning_config import TuningConfig


def _write_tsv(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


@pytest.fixture
def sample_tuning_config(tmp_path):
    """Create a minimal TuningConfig for tests."""
    root_output_dir = tmp_path / "root_out"
    subfolder_name = "sub folder"
    reference_data = ["refA"]
    cfg = TuningConfig(
        tuning_method="reduced_dimensionality",
        model_names=["model1", "model2"],
        reference_data=reference_data,
        tuning_output_dir=str(tmp_path / "tuning_out"),
        root_output_dir=str(root_output_dir),
        k_values=[],
        subfolder_name=subfolder_name,
        hyperparameter_table_path=""
    )
    return cfg


def test_collect_analyses_results(sample_tuning_config, tmp_path):
    # Prepare fake analyses directory structure using the same normalization as the implementation
    root_output_dir = Path = tmp_path / "root_out"
    subfolder = "_".join(sample_tuning_config.subfolder_name.split())
    ref_join = "_".join(sample_tuning_config.reference_data)
    analyses_dir = root_output_dir / "analyses" / "reduced_dimensionality" / subfolder / ref_join
    analyses_dir.mkdir(parents=True)

    # Create two aminoacid files (one for model1, one for model2)
    aa1 = pd.DataFrame({
        "Reference": ["refA", "refA"],
        "Model": ["model1", "model1"],
        "Mean_Score": [0.1, 0.3]
    })
    aa2 = pd.DataFrame({
        "Reference": ["refA"],
        "Model": ["model2"],
        "Mean_Score": [0.2]
    })
    _write_tsv(analyses_dir / "aminoacid_model1.tsv", aa1)
    _write_tsv(analyses_dir / "aminoacid_model2.tsv", aa2)

    # kmer_grouped.tsv
    kmer = pd.DataFrame({
        "Reference": ["refA", "refA"],
        "Model": ["model1", "model2"],
        "Mean_Score": [0.05, 0.15]
    })
    _write_tsv(analyses_dir / "kmer_grouped.tsv", kmer)

    # length_grouped.tsv
    length_df = pd.DataFrame({
        "Reference": ["refA", "refA"],
        "Model": ["model1", "model2"],
        "Mean_Score": [0.2, 0.4]
    })
    _write_tsv(analyses_dir / "length_grouped.tsv", length_df)

    aa_summary, kmer_summary, length_summary = tuning_reduced_dim.collect_analyses_results(sample_tuning_config)

    # Verify that the aminoacid summary averaged the two rows for model1
    m1_score = aa_summary.loc[aa_summary.Model == "model1", "Score"].values[0]
    assert pytest.approx(m1_score, rel=1e-6) == 0.2  # mean of [0.1, 0.3]

    m2_score = aa_summary.loc[aa_summary.Model == "model2", "Score"].values[0]
    assert pytest.approx(m2_score, rel=1e-6) == 0.2

    # kmer and length checks
    assert float(kmer_summary.loc[kmer_summary.Model == "model1", "Score"]) == pytest.approx(0.05)
    assert float(kmer_summary.loc[kmer_summary.Model == "model2", "Score"]) == pytest.approx(0.15)
    assert float(length_summary.loc[length_summary.Model == "model1", "Score"]) == pytest.approx(0.2)
    assert float(length_summary.loc[length_summary.Model == "model2", "Score"]) == pytest.approx(0.4)


def test_run_reduced_dim_tuning(sample_tuning_config, tmp_path, mocker):
    # create matching analyses directory
    root_output_dir = Path = tmp_path / "root_out"
    subfolder = "_".join(sample_tuning_config.subfolder_name.split())
    ref_join = "_".join(sample_tuning_config.reference_data)
    analyses_dir = root_output_dir / "analyses" / "reduced_dimensionality" / subfolder / ref_join
    analyses_dir.mkdir(parents=True)

    # Create minimal files needed by collect_analyses_results
    aa = pd.DataFrame({"Reference": ["refA"], "Model": ["m"], "Mean_Score": [0.5]})
    _write_tsv(analyses_dir / "aminoacid_m.tsv", aa)
    _write_tsv(analyses_dir / "kmer_grouped.tsv", pd.DataFrame({"Reference":["refA"], "Model":["m"], "Mean_Score":[0.6]}))
    _write_tsv(analyses_dir / "length_grouped.tsv", pd.DataFrame({"Reference":["refA"], "Model":["m"], "Mean_Score":[0.7]}))

    # patch validate_analyses_data to avoid external checks and return the analyses path
    mocker.patch('gen_airr_bm.tuning.tuning_reduced_dim.validate_analyses_data', return_value=[str(analyses_dir)])

    calls = []

    def fake_save_and_plot(cfg_arg, analysis_name, df, outdir, plot_title=None):
        calls.append((analysis_name, df.copy(), outdir, plot_title))

    mocker.patch('gen_airr_bm.tuning.tuning_reduced_dim.save_and_plot_tuning_results', side_effect=fake_save_and_plot)

    # Run
    tuning_reduced_dim.run_reduced_dim_tuning(sample_tuning_config)

    # We expect three calls (aminoacid, kmer, length)
    assert len(calls) == 3
    names = [c[0] for c in calls]
    assert set(names) == {"aminoacid", "kmer", "length"}

    # Check that the dataframes contain the Score column
    for analysis_name, df, outdir, title in calls:
        assert "Score" in df.columns
        assert outdir == sample_tuning_config.tuning_output_dir
        assert title == f"JSD scores between reference and generated {analysis_name} distributions"
