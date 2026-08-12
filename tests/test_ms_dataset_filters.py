"""Unit tests for the instrument / fragmentation / metadata classification in build_ms_datasets.py.

Every string below is one that ACTUALLY occurs in NIST20, NIST23, MoNA, GNPS or Agilent. The point
of the test is that these classifications have already gone wrong several times in ways that were
silent -- the data still built, the model still trained, the numbers just got quietly worse:

  * `IT-FT/ion trap with FTMS` is NIST's spelling of ION-TRAP CID. It contains neither "hcd" nor
    "cid", so a substring classifier returns UNKNOWN, and the Orbitrap group's "unlabelled ->
    impute HCD" rule then swept 39,226 resonant-excitation spectra into an HCD-only dataset.
    Measured damage: Lumos replicate self-similarity 0.5697 with it, 0.9935 without.
  * GNPS stores FRAGMENTATION in `instrument_type` while every other source stores the ANALYSER
    there, so classifiers must pool both fields rather than trust either one.
  * `quadrupole` (QQQ) belongs with QTOF, but `ion trap` does not belong with either -- resonant
    excitation has a ~1/3 low-mass cutoff, the one fragmentation difference that 0.2 Da binning
    cannot wash out.

Run: python -m pytest tests/test_ms_dataset_filters.py -q
"""
import os
import sys
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from molnetpack import config_path

CFG = config_path("encoding_etkdgv3.yml")
from build_msms_dataset import group_of, frag_of, ce_to_nce, prec_mz_ok


# ---------------------------------------------------------------- analyser grouping
@pytest.mark.parametrize("source_instrument,instrument_type,expected", [
    # NIST: SOURCE_INSTRUMENT is a model name, INSTRUMENT_TYPE mixes fragmentation and analyser
    ("Thermo Finnigan Elite Orbitrap", "HCD", "orbi"),
    ("Orbitrap Fusion Lumos", "HCD", "orbi"),
    ("Thermo Finnigan Velos Orbitrap", "HCD", "orbi"),
    ("Thermo Finnigan Elite Orbitrap", "IT-FT/ion trap with FTMS", "orbi"),
    ("Agilent QTOF 6530", "Q-TOF", "qtof"),
    # GNPS: SOURCE_INSTRUMENT is the analyser, INSTRUMENT_TYPE is the fragmentation
    ("ftms", "hcd", "orbi"),
    ("orbitrap", "cid", "orbi"),
    ("qtof", "cid", "qtof"),
    ("quadrupole", "hcd", "qtof"),          # QQQ pools with QTOF
    ("ion trap", "cid", "iontrap"),         # excluded: resonant excitation
    # MoNA / Agilent: analyser lives in one field or the other
    ("Agilent 6530 Q-TOF", "LC-ESI-QTOF", "qtof"),
    ("Waters Xevo G2 Q-Tof", "LC-ESI-QTOF", "qtof"),
    ("", "ESI-QFT", "orbi"),                # Q Exactive FT
    ("Unknown", "ESI-QTOF", "qtof"),
])
def test_group_of(source_instrument, instrument_type, expected):
    assert group_of(source_instrument, instrument_type) == expected


def test_it_ft_is_orbitrap_analyser_not_ion_trap():
    """'IT-FT/ion trap with FTMS' says 'ion trap' but is FT-detected -> Orbitrap-class analyser."""
    assert group_of("Orbitrap Fusion Lumos", "IT-FT/ion trap with FTMS") == "orbi"
    assert group_of("ion trap", "cid") == "iontrap"


# ---------------------------------------------------------------- fragmentation
@pytest.mark.parametrize("source_instrument,instrument_type,expected", [
    ("Thermo Finnigan Elite Orbitrap", "HCD", "HCD"),
    ("ftms", "hcd", "HCD"),
    ("qtof", "cid", "CID"),
    ("orbitrap", "cid", "CID"),
    # THE REGRESSION THIS FILE EXISTS FOR: NIST's spelling of ion-trap CID
    ("Thermo Finnigan Elite Orbitrap", "IT-FT/ion trap with FTMS", "CID"),
    ("Orbitrap Fusion Lumos", "IT-FT/ion trap with FTMS", "CID"),
    # analyser names are NOT fragmentation labels
    ("Agilent QTOF 6530", "Q-TOF", "UNK"),
    ("Agilent 6530 Q-TOF", "LC-ESI-QTOF", "UNK"),
    ("Unknown", "ESI-QTOF", "UNK"),
])
def test_frag_of(source_instrument, instrument_type, expected):
    assert frag_of(source_instrument, instrument_type) == expected


def test_ion_trap_cid_never_imputed_as_hcd():
    """The orbi group keeps HCD and unlabelled records. Ion-trap CID must NOT read as unlabelled,
    or it gets imputed into an HCD-only dataset (measured: Lumos self-similarity 0.57 vs 0.99)."""
    assert frag_of("Orbitrap Fusion Lumos", "IT-FT/ion trap with FTMS") != "UNK"
    assert frag_of("Orbitrap Fusion Lumos", "IT-FT/ion trap with FTMS") != "HCD"


# ---------------------------------------------------------------- collision energy
@pytest.mark.parametrize("ce_str,prec_mz,charge,expected", [
    ("NCE=35% 25eV", 283.13, 1, 0.35),      # both stated -> use the STATED NCE, do not re-derive
    ("NCE=50% 30eV", 500.0, 1, 0.50),
    ("HCD (NCE 40%)", 400.0, 1, 0.40),
    ("90(NCE)", 350.0, 1, 0.90),
    ("40 (nominal)", 600.0, 1, 0.40),
    ("20 eV", 500.0, 1, 0.20),              # eV -> 20*500/500 = 20% -> 0.20
    ("", 300.0, 1, 0.0),                    # unparseable
])
def test_ce_to_nce(ce_str, prec_mz, charge, expected):
    assert ce_to_nce(ce_str, prec_mz, charge) == pytest.approx(expected, abs=1e-3)


def test_nce_is_always_a_fraction_never_percent():
    """A 100x scale mix (fraction vs percent in the same channel) shipped once already."""
    for s, mz in [("NCE=35% 25eV", 283.13), ("HCD (NCE 40%)", 400.0), ("20 eV", 500.0),
                  ("90(NCE)", 350.0), ("35", 283.13)]:
        assert 0.0 <= ce_to_nce(s, mz, 1) <= 3.0, f"{s!r} left the fraction scale"


def test_ce_to_nce_survives_missing_precursor():
    assert ce_to_nce("NCE=35%", 0.0, 1) == 0.0
    assert ce_to_nce("garbage", 300.0, 1) == 0.0


# ---------------------------------------------------------------- precursor consistency
def test_prec_mz_matches_theoretical():
    # caffeine, [M+H]+ = 195.0877
    ok, theo = prec_mz_ok("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "[M+H]+", 195.0877)
    assert ok and theo == pytest.approx(195.0877, abs=1e-3)


def test_prec_mz_rejects_wrong_annotation():
    """NIST23 has records sharing a SMILES and adduct while reporting 244.97 / 381.17 / 453.21 --
    a given adduct has exactly one m/z, so the structure annotation is wrong on some of them."""
    ok, _ = prec_mz_ok("Nc1nc(-c2cc(Cl)ccc2Cl)cs1", "[M+H]+", 381.1697)
    assert not ok
    ok, _ = prec_mz_ok("Nc1nc(-c2cc(Cl)ccc2Cl)cs1", "[M+H]+", 244.9702)
    assert ok


def test_prec_mz_rejects_unparseable_or_missing():
    assert prec_mz_ok("not a smiles", "[M+H]+", 195.0)[0] is False
    assert prec_mz_ok("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "[M+H]+", 0.0)[0] is False


# ---------------------------------------------------------------- CE parser regex coverage
@pytest.mark.parametrize("ce_str,prec_mz,expected_nce_pct", [
    ("NCE=35%", 300.0, 35.0),
    ("NCE=1.5%", 400.0, 1.5),        # DECIMAL NCE: `^NCE=[\d]+\%$` required an integer, so this
    ("NCE=27.5%", 500.0, 27.5),      # fell through to unparseable and env[0] silently became 0
    ("nce=42.5%", 350.0, 42.5),      # lowercase
    ("NCE=35.0% 25eV", 283.13, 35.0),  # decimal in the both-stated form
])
def test_decimal_nce_parses(ce_str, prec_mz, expected_nce_pct):
    from molnetpack.data_utils.utils import parse_collision_energy
    _, nce = parse_collision_energy(ce_str, prec_mz, 1)
    assert nce is not None, f"{ce_str!r} did not parse — it will become env[0]=0"
    assert nce == pytest.approx(expected_nce_pct, abs=1e-6)


def test_decimal_nce_reaches_env_as_fraction():
    """The builder must turn decimal NCE into a fraction like any other, not drop it to 0."""
    assert ce_to_nce("NCE=1.5%", 400.0, 1) == pytest.approx(0.015, abs=1e-6)
    assert ce_to_nce("NCE=27.5%", 500.0, 1) == pytest.approx(0.275, abs=1e-6)


# ---------------------------------------------------------------- adduct layout from config
def test_adduct_order_comes_from_the_config_one_hot():
    """The layout must be DERIVED from encoding.precursor_type, not hardcoded and not a CLI flag.
    The one-hot vector in the config IS the index, so trainer and data cannot drift apart."""
    import yaml
    from molnetpack.spectrum_heads import adduct_order
    pt = yaml.safe_load(open(CFG))["encoding"]["precursor_type"]
    order = adduct_order(CFG)
    assert len(order) == len(pt)
    for name, onehot in pt.items():
        assert order[list(onehot).index(1)] == name, f"{name} is at the wrong index"


def test_adduct_order_has_no_gaps():
    from molnetpack.spectrum_heads import adduct_order
    order = adduct_order(CFG)
    assert all(a is not None for a in order)
    assert len(set(order)) == len(order)


def test_precursor_bin_uses_the_config_layout():
    """A [M+H]+ one-hot must resolve to the protonated mass, i.e. index 0 of the config order."""
    import numpy as np
    from molnetpack.spectrum_heads import precursor_bin, adduct_order
    env = np.zeros(1 + len(adduct_order(CFG)))
    env[1 + adduct_order(CFG).index("[M+H]+")] = 1
    # caffeine [M+H]+ = 195.088 -> bin 975 at 0.2 Da
    assert precursor_bin("Cn1cnc2c1c(=O)n(C)c(=O)n2C", env, 0.2, 7500, CFG) == 975


# ---------------------------------------------------------------- adduct spelling normalisation
@pytest.mark.parametrize("raw,canonical", [
    # AllCCS writes the water loss BEFORE the proton; encoding_etkdgv3.yml writes it after.
    # The same file contains both orderings, so a hardcoded list drops one of them silently.
    ("[M-H2O+H]+", "[M+H-H2O]+"),
    ("[M+H-H2O]+", "[M+H-H2O]+"),
    ("[M-2H2O+H]+", "[M+H-2H2O]+"),
    ("[M+H-2H2O]+", "[M+H-2H2O]+"),
    ("[M-NH3+H]+", "[M+H-NH3]+"),
    ("[M-H2O-H]-", "[M-H-H2O]-"),
    (" [M+H]+ ", "[M+H]+"),          # whitespace
    ("[M+H] +", "[M+H]+"),           # internal space
])
def test_normalize_adduct(raw, canonical):
    from molnetpack.data_utils.utils import normalize_adduct
    assert normalize_adduct(raw) == canonical


def test_normalize_adduct_leaves_unknown_forms_alone():
    """An unrecognised adduct must pass through unchanged so a membership test fails LOUDLY
    rather than being silently rewritten into something wrong."""
    from molnetpack.data_utils.utils import normalize_adduct
    assert normalize_adduct("[M-SO3-H2O+H]+") == "[M-SO3-H2O+H]+"
    assert normalize_adduct("") == ""
    assert normalize_adduct(None) == ""


def test_both_water_loss_spellings_map_together():
    from molnetpack.data_utils.utils import normalize_adduct as n
    assert n("[M-H2O+H]+") == n("[M+H-H2O]+")
