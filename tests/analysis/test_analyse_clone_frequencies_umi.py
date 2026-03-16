import numpy as np
import pandas as pd
import plotly.graph_objects as go

from gen_airr_bm.analysis.analyse_clone_frequencies_umi import (
    get_frequencies_df,
    pseudo_log_transform,
    create_scatter_plot,
    plot_frequencies_by_dataset,
    plot_frequencies_combined,
)


def test_get_frequencies_df():
    sample_1 = ["AAA", "BBB", "AAA", "CCC"]  # counts: AAA=2, BBB=1, CCC=1
    sample_2 = ["AAA", "AAA", "DDD"]          # counts: AAA=2, DDD=1

    df = get_frequencies_df(sample_1, sample_2, "train", "test")

    # Index contains union of sequences
    expected_index = set(["AAA", "BBB", "CCC", "DDD"])
    assert set(df.index) == expected_index

    # Counts should be integers and correct
    assert df.loc["AAA", "count_train"] == 2
    assert df.loc["BBB", "count_train"] == 1
    assert df.loc["CCC", "count_train"] == 1
    assert df.loc["DDD", "count_train"] == 0

    assert df.loc["AAA", "count_test"] == 2
    assert df.loc["DDD", "count_test"] == 1
    assert df.loc["BBB", "count_test"] == 0

    # Frequencies should be counts / sample size
    assert np.isclose(df.loc["AAA", "freq_train"], 2/4)
    assert np.isclose(df.loc["AAA", "freq_test"], 2/3)
    assert np.isclose(df.loc["DDD", "freq_test"], 1/3)


def test_pseudo_log_transform():
    # Includes zeros and small/large positives to ensure it doesn't error and is monotonic non-decreasing
    x = np.array([0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0])
    y = pseudo_log_transform(x)

    # Should preserve shape and be finite
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))

    # Monotonic non-decreasing
    assert np.all(np.diff(y) >= -1e-12)

    # Zero maps to zero in typical symlog
    assert np.isclose(y[0], 0.0)


def test_create_scatter_plot():
    # Minimal combined df with required columns
    df = pd.DataFrame({
        "pseudo_freq_model": [0.0, 0.5, 1.0],
        "pseudo_freq_reference": [0.0, 0.4, 1.2],
        "sequence": ["AAA", "BBB", "CCC"],
    })

    fig = create_scatter_plot(df, name1="reference", name2="model", title_text="Test Title")

    assert isinstance(fig, go.Figure)
    # Scatter trace present
    assert any(trace.type == "scatter" for trace in fig.data)


def test_plot_frequencies_by_dataset(tmp_path, monkeypatch):
    # Prepare tiny frequency frames
    df = pd.DataFrame({
        "count_ref": [1, 1],
        "count_mod": [1, 0],
        "freq_ref": [0.5, 0.5],
        "freq_mod": [0.5, 0.0],
    }, index=["AAA", "BBB"])  # index acts as sequence ids

    freqs = {"set1": df}

    # Speed up: patch write_image to avoid heavy image engine; instead, create a small placeholder file
    created_files = []

    def fake_write_image(self, path):
        # just create an empty file to simulate output
        open(path, "wb").close()
        created_files.append(path)

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image, raising=False)

    outdir = tmp_path / "plots"
    outdir.mkdir()

    jsds = plot_frequencies_by_dataset(freqs, str(outdir), name1="ref", name2="mod")

    # Should have returned one JSD (since dataset name has no _all)
    assert len(jsds) == 1
    # File created with expected naming
    expected_png = outdir / "set1_mod_ref_symlog.png"
    assert expected_png.exists()


def test_plot_frequencies_combined(tmp_path, monkeypatch):
    # Build two datasets, one with _all and one without
    df_all = pd.DataFrame({
        "count_ref": [1, 1],
        "count_mod": [1, 0],
        "freq_ref": [0.5, 0.5],
        "freq_mod": [0.5, 0.0],
    }, index=["AAA", "BBB"])  

    df_part = pd.DataFrame({
        "count_ref": [1],
        "count_mod": [1],
        "freq_ref": [1.0],
        "freq_mod": [1.0],
    }, index=["CCC"])  

    freqs = {
        "dataset_all": df_all,
        "dataset_part": df_part,
    }

    created_files = []

    def fake_write_image(self, path):
        open(path, "wb").close()
        created_files.append(path)

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image, raising=False)

    outdir = tmp_path / "plots2"
    outdir.mkdir()

    # With filter_combined_rep=True -> only uses dataset_all
    plot_frequencies_combined(freqs, str(outdir), name1="ref", name2="mod", filter_combined_rep=True)
    expected_png = outdir / "combined_repertoires_mod_ref_symlog.png"
    assert expected_png.exists()

    # Ensure figure creation worked even when including all datasets
    # Overwrite file again should still succeed
    plot_frequencies_combined(freqs, str(outdir), name1="ref", name2="mod", filter_combined_rep=False)
    assert expected_png.exists()  # still present
