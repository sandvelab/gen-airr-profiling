import numpy as np
import pandas as pd
import pytest

from gen_airr_bm.data_preprocessing.data_generation_methods import adaptive_to_imgt_gene_name, \
    read_experimental_columns


@pytest.mark.parametrize(
    "adaptive_name,imgt_name",
    [
        ("TCRBV05-05*01", "TRBV5-5*01"),       # leading zeros removed, allele kept
        ("TCRBV07-09", "TRBV7-9"),             # no allele
        ("TCRBV10-01*01", "TRBV10-1*01"),      # zero inside the family number is kept
        ("TCRBJ02-07*01", "TRBJ2-7*01"),
        ("TCRBV02-01*01", "TRBV2*01"),         # single-gene family has no gene number in IMGT
        ("TCRBV28-01", "TRBV28"),
        ("TCRBD01-01*01", "TRBD1*01"),
        ("TCRBV20-01*01", "TRBV20-1*01"),      # family with an orphon keeps the gene number
        ("TCRBV20-or09_02", "TRBV20/OR9-2"),   # orphon
        ("TCRBV26-or09_02", "TRBV26/OR9-2"),   # orphon of a single-gene family
        ("TCRBVA-or09_02*01", "TRBVA/OR9-2*01"),
        ("TCRBV20", "TRBV20"),                 # family-only call stays a family call
        ("TCRBJ02", "TRBJ2"),
        ("TRBV5-5*01", "TRBV5-5*01"),          # IMGT names are not changed
        ("IGHV1/OR15-1*04", "IGHV1/OR15-1*04"),
    ],
)
def test_adaptive_to_imgt_gene_name(adaptive_name, imgt_name):
    assert adaptive_to_imgt_gene_name(adaptive_name) == imgt_name


def test_adaptive_to_imgt_gene_name_keeps_missing_values():
    assert adaptive_to_imgt_gene_name(np.nan) is np.nan
    assert adaptive_to_imgt_gene_name(None) is None


def test_read_experimental_columns_converts_adaptive_gene_names(tmp_path):
    input_path = tmp_path / "adaptive.tsv"
    pd.DataFrame({
        "cdr3_amino_acid": ["CASSPPRGDQETQYF", "CASSLGRGRVETQYF", "CASSLAGYEQYF"],
        "v_resolved": ["TCRBV05-05*01", "TCRBV20", np.nan],
        "j_resolved": ["TCRBJ02-05*01", "TCRBJ02-07*01", "TCRBJ02-07*01"],
        "locus": ["TCRB", "TCRB", "TCRB"],
    }).to_csv(input_path, sep="\t", index=False)

    data = read_experimental_columns(input_path, ["cdr3_amino_acid", "v_resolved", "j_resolved", "locus"])

    assert data["v_call"].tolist()[:2] == ["TRBV5-5*01", "TRBV20"]
    assert pd.isna(data["v_call"].iloc[2])
    assert data["j_call"].tolist() == ["TRBJ2-5*01", "TRBJ2-7*01", "TRBJ2-7*01"]
    assert data["locus"].tolist() == ["TRB", "TRB", "TRB"]
