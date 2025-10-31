import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from gen_airr_bm.utils import tuning_utils
from gen_airr_bm.core.tuning_config import TuningConfig


def test_format_value():
    assert tuning_utils.format_value(3) == "3"
    assert tuning_utils.format_value(np.int64(7)) == "7"
    assert tuning_utils.format_value(3.14159) == "3.142"
    assert tuning_utils.format_value(np.float64(2.5)) == "2.5"
    assert tuning_utils.format_value("abc") == "abc"


def test_validate_analyses_data(tmp_path):
    root = tmp_path / "root"
    # create an analyses dir for 'foo_analysis'
    (root / "analyses" / "foo_analysis" / "sub_folder").mkdir(parents=True)

    cfg = TuningConfig(
        tuning_method="test",
        model_names=["m"],
        reference_data=["ref"],
        tuning_output_dir=str(tmp_path / "out"),
        root_output_dir=str(root),
        k_values=[],
        subfolder_name="sub folder",
        hyperparameter_table_path=""
    )

    # should succeed
    validated = tuning_utils.validate_analyses_data(cfg, required_analyses=["foo_analysis"])
    assert isinstance(validated, list)
    assert str(root / "analyses" / "foo_analysis" / "sub_folder") in [str(p) for p in validated]

    # missing analysis should raise
    cfg2 = TuningConfig(
        tuning_method="test",
        model_names=["m"],
        reference_data=["ref"],
        tuning_output_dir=str(tmp_path / "out"),
        root_output_dir=str(root),
        k_values=[],
        subfolder_name="does_not_exist",
        hyperparameter_table_path=""
    )

    with pytest.raises(FileNotFoundError):
        tuning_utils.validate_analyses_data(cfg2, required_analyses=["foo_analysis"])


def test_save_and_plot_tuning_results(tmp_path, mocker):
    outdir = tmp_path / "out"
    outdir.mkdir()

    # create a tuning config and hyperparameter table file
    hyperparams = pd.DataFrame({
        "Hyperparameters": ["lr", "batch"],
        "model_1": ["0.1", "32"],
        "model_2": ["0.2", "64"]
    })
    hyperparam_file = tmp_path / "hyperparams.tsv"
    hyperparams.to_csv(hyperparam_file, sep="\t", index=False)

    cfg = TuningConfig(
        tuning_method="test",
        model_names=["model_1", "model_2"],
        reference_data=["test"],
        tuning_output_dir=str(outdir),
        root_output_dir=str(tmp_path),
        k_values=[],
        subfolder_name="sub",
        hyperparameter_table_path=str(hyperparam_file)
    )

    # create a summary dataframe with an extra Reference and k_value to exercise filtering
    summary_df = pd.DataFrame({
        "Model": ["model_1", "model_2", "model_1"],
        "Score": [0.1, 0.2, 0.15],
        "Reference": ["test", "test", "other"],
        "k_value": [0, 0, 1]
    })

    # Fake figure to avoid requiring plotly image backends
    class FakeFig:
        def __init__(self):
            self.traces = []
            self.written = None

        def add_trace(self, *args, **kwargs):
            self.traces.append((args, kwargs))

        def update_traces(self, *args, **kwargs):
            pass

        def update_layout(self, *args, **kwargs):
            pass

        def update_yaxes(self, *args, **kwargs):
            pass

        def write_image(self, path):
            # create an empty file to emulate image writing
            Path(path).write_text("")
            self.written = str(path)

    # Monkeypatch make_subplots to return our FakeFig instance
    mocker.patch('gen_airr_bm.utils.tuning_utils.make_subplots', return_value=FakeFig())

    # Call the function
    tuning_utils.save_and_plot_tuning_results(cfg, "my_analysis", summary_df.copy(), str(outdir), plot_title="title")

    # Check that summary TSV was written and contains original rows
    summary_path = outdir / "my_analysis_summary.tsv"
    assert summary_path.exists()
    read_back = pd.read_csv(summary_path, sep="\t")
    # Ensure saved file contains the same number of rows as original
    assert len(read_back) == len(summary_df)

    # Check that image file was created
    img_path = outdir / "my_analysis_summary.png"
    assert img_path.exists()
    # file should be empty per FakeFig.write_image implementation
    assert img_path.read_text() == ""
