"""Tests for collision-energy input in the CSV converter.

The ``Collision_Energy`` column accepts free-text values carrying their own unit
("20 V", "NCE=35%"). The optional ``Collision_Energy_Unit`` column lets users give plain
numbers instead, with the unit stated once per row (``eV`` or ``NCE``). Both spellings of
the same energy must produce the same encoded env; an unrecognised unit drops the row
rather than guessing.

Run: python -m pytest tests/test_csv_energy_units.py -q
"""
import pandas as pd
import pytest
import yaml

from molnetpack import config_path, molecules_to_records


@pytest.fixture(scope="module")
def encoding():
    return yaml.safe_load(open(config_path("encoding_etkdgv3.yml")))["encoding"]


def _records(df, encoding):
    recs = molecules_to_records(df, encoding)
    return {r["title"]: r for r in recs}


def test_unit_column_matches_free_text(encoding):
    df = pd.DataFrame([
        {"ID": "freetext_ev", "SMILES": "CCO", "Precursor_Type": "[M+H]+",
         "Collision_Energy": "20 V"},
        {"ID": "numeric_ev", "SMILES": "CCO", "Precursor_Type": "[M+H]+",
         "Collision_Energy": "20", "Collision_Energy_Unit": "eV"},
        {"ID": "numeric_nce", "SMILES": "CCO", "Precursor_Type": "[M+H]+",
         "Collision_Energy": "35", "Collision_Energy_Unit": "NCE"},
        {"ID": "freetext_nce", "SMILES": "CCO", "Precursor_Type": "[M+H]+",
         "Collision_Energy": "NCE=35%"},
    ])
    recs = _records(df, encoding)
    assert set(recs) == {"freetext_ev", "numeric_ev", "numeric_nce", "freetext_nce"}
    # same energy, either spelling -> identical encoded NCE
    assert recs["numeric_ev"]["env"][0] == pytest.approx(recs["freetext_ev"]["env"][0])
    assert recs["numeric_nce"]["env"][0] == pytest.approx(recs["freetext_nce"]["env"][0])
    assert recs["numeric_nce"]["env"][0] == pytest.approx(35.0)


def test_unknown_unit_drops_the_row(encoding):
    df = pd.DataFrame([
        {"ID": "ok", "SMILES": "CCO", "Precursor_Type": "[M+H]+",
         "Collision_Energy": "20 V"},
        {"ID": "typo_unit", "SMILES": "CCO", "Precursor_Type": "[M+H]+",
         "Collision_Energy": "35", "Collision_Energy_Unit": "NEC"},
    ])
    recs = _records(df, encoding)
    assert "ok" in recs and "typo_unit" not in recs


def test_missing_unit_cells_fall_back_to_free_text(encoding):
    # a unit column where some rows are empty: NaN cells behave as if the column were absent
    df = pd.DataFrame([
        {"ID": "with_unit", "SMILES": "CCO", "Precursor_Type": "[M+H]+",
         "Collision_Energy": "35", "Collision_Energy_Unit": "NCE"},
        {"ID": "no_unit", "SMILES": "CCO", "Precursor_Type": "[M+H]+",
         "Collision_Energy": "20 V", "Collision_Energy_Unit": None},
    ])
    recs = _records(df, encoding)
    assert set(recs) == {"with_unit", "no_unit"}
    assert recs["with_unit"]["env"][0] == pytest.approx(35.0)
