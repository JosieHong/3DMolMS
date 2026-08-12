"""Tests for MGF inference input.

MGF blocks are read for their metadata only (TITLE, SMILES, PRECURSOR_TYPE,
COLLISION_ENERGY). Peak lists and any stated PRECURSOR_MZ are ignored, so a
metadata-only block and a full spectral-library record must load identically;
blocks missing a required field are skipped rather than failing the load.

Run: python -m pytest tests/test_mgf_input.py -q
"""
import numpy as np
import pytest
import torch

from molnetpack import MolNet

MINIMAL = """BEGIN IONS
TITLE=bare
SMILES=CCO
PRECURSOR_TYPE=[M+H]+
COLLISION_ENERGY=20 V
END IONS
"""

WITH_PEAKS = """BEGIN IONS
TITLE=library_record
SMILES=CCO
PRECURSOR_TYPE=[M+H]+
PRECURSOR_MZ=47.0491
COLLISION_ENERGY=20 V
29.0386 100.0
31.0178 45.0
END IONS
"""

MISSING_CE = """BEGIN IONS
TITLE=no_energy
SMILES=CCO
PRECURSOR_TYPE=[M+H]+
END IONS
"""


@pytest.fixture(scope="module")
def engine():
    return MolNet(torch.device("cpu"), seed=42)


def _load(engine, tmp_path, text, name="in.mgf"):
    path = tmp_path / name
    path.write_text(text)
    engine.load_data(str(path))
    return engine.get_data()


def test_peakless_block_loads(engine, tmp_path):
    records = _load(engine, tmp_path, MINIMAL)
    assert [r["title"] for r in records] == ["bare"]


def test_peaks_and_precursor_mz_are_ignored(engine, tmp_path):
    bare = _load(engine, tmp_path, MINIMAL, "bare.mgf")
    full = _load(engine, tmp_path, WITH_PEAKS, "full.mgf")
    assert len(full) == 1
    np.testing.assert_array_equal(bare[0]["env"], full[0]["env"])
    np.testing.assert_array_equal(bare[0]["mol"], full[0]["mol"])


def test_block_missing_required_field_is_skipped(engine, tmp_path):
    records = _load(engine, tmp_path, MINIMAL + "\n" + MISSING_CE)
    assert [r["title"] for r in records] == ["bare"]
