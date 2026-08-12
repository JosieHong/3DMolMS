"""Tests for the startup cross-check between the shared data-encoding config and the
task model configs.

Several sizes are duplicated across the config files (molecules are featurised/padded/binned
with the ENCODING values, models are sized with the MODEL values) and nothing else ties them
together. A drift is a silent-mismatch bug — spectra binned on one grid, heads sized for
another — so MolNet must refuse to construct rather than predict garbage.

Run: python -m pytest tests/test_config_consistency.py -q
"""
import warnings

import pytest
import torch
import yaml

from molnetpack import MolNet, config_path

warnings.filterwarnings("ignore")


def _write_tampered(tmp_path, cfg_name, section, key, value):
    cfg = yaml.safe_load(open(config_path(cfg_name)))
    cfg[section][key] = value
    out = tmp_path / cfg_name
    yaml.safe_dump(cfg, open(out, "w"))
    return str(out)


def _molnet(**overrides):
    return MolNet(device=torch.device("cpu"), seed=0, **overrides)


def test_default_configs_are_consistent():
    _molnet()


@pytest.mark.parametrize("key,value,match", [
    ("resolution", 0.1, "resolution"),
    ("max_mz", 1000, "max_mz"),
    ("max_atom_num", 200, "max_atom_num"),
    ("in_dim", 14, "in_dim"),
    ("add_num", 5, "add_num"),
])
def test_msms_model_drift_is_refused(tmp_path, key, value, match):
    path = _write_tampered(tmp_path, "molnet.yml", "model", key, value)
    with pytest.raises(ValueError, match=match):
        _molnet(msms_config_path=path)


def test_ccs_add_num_must_match_its_own_adduct_layout(tmp_path):
    path = _write_tampered(tmp_path, "molnet_ccs_tl.yml", "model", "add_num", 7)
    with pytest.raises(ValueError, match="ccs.*add_num"):
        _molnet(ccs_config_path=path)


def test_rt_add_num_must_stay_placeholder_without_adduct_encoding(tmp_path):
    path = _write_tampered(tmp_path, "molnet_rt_tl.yml", "model", "add_num", 6)
    with pytest.raises(ValueError, match="rt.*placeholder"):
        _molnet(rt_config_path=path)


def test_encoding_drift_is_refused_too(tmp_path):
    """The check is symmetric: tampering the shared encoding config trips it as well."""
    path = _write_tampered(tmp_path, "encoding_etkdgv3.yml", "encoding", "resolution", 1.0)
    with pytest.raises(ValueError, match="resolution"):
        _molnet(data_config_path=path)
