import pandas as pd
import pytest

from gen_airr_bm.tuning import tuning_overlap
from gen_airr_bm.core.tuning_config import TuningConfig


def _write_tsv(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


@pytest.fixture
def sample_tuning_config(tmp_path):
    root_output_dir = tmp_path / "root_out"
    subfolder_name = "sub folder"
    cfg = TuningConfig(
        tuning_method="overlap",
        model_names=["model_1", "model_2"],
        reference_data=["test"],
        tuning_output_dir=str(tmp_path / "tuning_out"),
        root_output_dir=str(root_output_dir),
        k_values=[0, 1],
        subfolder_name=subfolder_name,
        hyperparameter_table_path=""
    )
    return cfg


def test_get_overlap_results(sample_tuning_config, tmp_path):
    # Prepare memorization file path: analyses/memorization/{subfolder}/memorization.tsv and _mean_ref.tsv
    root = tmp_path / "root_out"
    subfolder = "_".join(sample_tuning_config.subfolder_name.split())
    memorization_base = root / "analyses" / "memorization" / subfolder / "memorization"
    memorization_base.parent.mkdir(parents=True, exist_ok=True)

    mem_df = pd.DataFrame({"model": ["model_1", "model_2"], "mean_overlap_score": [0.1, 0.2]})
    _write_tsv(memorization_base.with_suffix('.tsv'), mem_df)

    # mean ref score file
    with open(str(memorization_base) + "_mean_ref.tsv", "w") as f:
        f.write("0.123\n")

    # Precision recall: analyses/precision_recall/{subfolder}/test/precision_recall_data.tsv
    prec_dir = root / "analyses" / "precision_recall" / subfolder / "test"
    prec_dir.mkdir(parents=True, exist_ok=True)
    prec_df = pd.DataFrame({"Model": ["model_1", "upper_reference"], "Precision_mean": [0.5, 0.55]})
    _write_tsv(prec_dir / "precision_recall_data.tsv", prec_df)

    # adjust config root_output_dir to our tmp root
    sample_tuning_config.root_output_dir = str(root)

    mem_df_out, mem_mean_ref_score, prec_df_out, prec_mean_ref_score = tuning_overlap.get_overlap_results(sample_tuning_config)

    pd.testing.assert_frame_equal(mem_df_out.reset_index(drop=True), mem_df.reset_index(drop=True))
    assert mem_mean_ref_score == pytest.approx(0.123)
    pd.testing.assert_frame_equal(prec_df_out.reset_index(drop=True), prec_df.reset_index(drop=True))
    assert prec_mean_ref_score == pytest.approx(0.55)


def test_compute_overlap_score(sample_tuning_config):
    # Prepare precision_recall_df including an upper_reference row
    prec_df = pd.DataFrame({
        "Model": ["model_1", "model_2", "upper_reference"],
        "Precision_mean": [0.5, 0.6, 0.9]
    })

    mem_df = pd.DataFrame({"model": ["model_1", "model_2"], "mean_overlap_score": [0.1, 0.2]})

    # k_values are [0,1] in fixture
    result = tuning_overlap.compute_overlap_score(sample_tuning_config, mem_df, prec_df)

    # Expect 2 models * 2 k values = 4 rows
    assert len(result) == 4

    # For k=0, Overlap_score_k_scaled should equal Precision_mean
    k0 = result[result.k_value == 0]
    for _, row in k0.iterrows():
        model = row.Model
        prec = prec_df.loc[prec_df.Model == model, "Precision_mean"].values[0]
        mem = mem_df.loc[mem_df.model == model, "mean_overlap_score"].values[0]
        assert row.Score == pytest.approx(prec)
        assert row.Overlap_score_k_scaled == pytest.approx(prec + 0 * mem)

    # For k=1, Overlap_score_k_scaled == Precision_mean + Memorization
    k1 = result[result.k_value == 1]
    for _, row in k1.iterrows():
        model = row.Model
        prec = prec_df.loc[prec_df.Model == model, "Precision_mean"].values[0]
        mem = mem_df.loc[mem_df.model == model, "mean_overlap_score"].values[0]
        assert row.Overlap_score_k_scaled == pytest.approx(prec + 1 * mem)


def test_run_overlap_tuning(sample_tuning_config, tmp_path, mocker):
    # create needed files for get_overlap_results
    root = tmp_path / "root_out"
    subfolder = "_".join(sample_tuning_config.subfolder_name.split())
    memorization_base = root / "analyses" / "memorization" / subfolder / "memorization"
    memorization_base.parent.mkdir(parents=True, exist_ok=True)

    mem_df = pd.DataFrame({"model": ["model_1"], "mean_overlap_score": [0.1]})
    _write_tsv(memorization_base.with_suffix('.tsv'), mem_df)
    with open(str(memorization_base) + "_mean_ref.tsv", "w") as f:
        f.write("0.2\n")

    prec_dir = root / "analyses" / "precision_recall" / subfolder / "test"
    prec_dir.mkdir(parents=True, exist_ok=True)
    prec_df = pd.DataFrame({"Model": ["model_1", "upper_reference"], "Precision_mean": [0.4, 0.45]})
    _write_tsv(prec_dir / "precision_recall_data.tsv", prec_df)

    sample_tuning_config.root_output_dir = str(root)
    sample_tuning_config.tuning_output_dir = str(tmp_path / "tuning_out")

    # Patch validate_analyses_data to return the analyses path to satisfy run_overlap_tuning
    mocker.patch('gen_airr_bm.tuning.tuning_overlap.validate_analyses_data', return_value=[str(root)])

    # Patch plotting and saving functions to avoid writing images and files
    plot_calls = {}
    mocker.patch('gen_airr_bm.tuning.tuning_overlap.plot_precision_memorization_scatter', side_effect=lambda *args, **kwargs: plot_calls.setdefault('scatter', True))
    mocker.patch('gen_airr_bm.tuning.tuning_overlap.plot_tuning_score_by_k', side_effect=lambda *args, **kwargs: plot_calls.setdefault('by_k', True))

    saved = {}
    def fake_save(cfg_arg, name, df, outdir, plot_title=None):
        saved['called'] = (name, outdir)
    mocker.patch('gen_airr_bm.tuning.tuning_overlap.save_and_plot_tuning_results', side_effect=fake_save)

    # Run
    tuning_overlap.run_overlap_tuning(sample_tuning_config)

    # Assert that plots were called and save was invoked for 'overlap'
    assert plot_calls.get('scatter', False) is True
    assert plot_calls.get('by_k', False) is True
    assert saved.get('called', (None, None))[0] == 'overlap'
    assert saved.get('called', (None, None))[1] == sample_tuning_config.tuning_output_dir

