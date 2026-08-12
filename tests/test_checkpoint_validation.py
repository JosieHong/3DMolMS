"""Unit tests for checkpoint/config consistency validation.

The validator is the only thing standing between a user and the silent-mismatch class of bug:
none of the critical keys change a tensor shape, so a wrong pairing loads cleanly and predicts
from the wrong model. Three regimes are pinned here:

  * NO embedded config (pre-v1.4.0)  -> warn only; a known legacy class, explicitly unverifiable.
  * config present, keys consistent  -> load.
  * config present, ANY critical key differing OR recorded on only one side -> ValueError.
    The one-sided case is what let a stale 'ce_scale'-less snapshot pair with a percent-scale
    config and mis-scale every collision energy 100x without a sound.

Run: python -m pytest tests/test_checkpoint_validation.py -q
"""
import logging

import pytest

from molnetpack.checkpoints import CRITICAL_CONFIG_KEYS, validate_checkpoint_config

BASE = {
    "in_dim": 21, "add_num": 6, "max_atom_num": 300, "emb_dim": 2048, "k": 6,
    "resolution": 0.2, "max_mz": 1500, "ce_scale": "fraction",
    "dropout": 0.2,  # not critical
}


def _ckpt(config):
    return {"model_state_dict": {}, "config": config}


def test_no_embedded_config_warns_but_loads(caplog):
    # the package logger runs with propagate=False (it owns a default handler);
    # re-enable propagation so caplog's root handler can see the record
    pkg_logger = logging.getLogger("molnetpack")
    pkg_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="molnetpack.checkpoints"):
            validate_checkpoint_config({"model_state_dict": {}}, dict(BASE), "legacy.pt")
    finally:
        pkg_logger.propagate = False
    assert "carries no config" in caplog.text


def test_identical_configs_pass():
    validate_checkpoint_config(_ckpt(dict(BASE)), dict(BASE), "ok.pt")


def test_non_critical_difference_passes():
    saved = dict(BASE, dropout=0.5)  # training-only knob; weights mean the same thing
    validate_checkpoint_config(_ckpt(saved), dict(BASE), "ok.pt")


def test_critical_value_mismatch_raises():
    saved = dict(BASE, ce_scale="percent")
    with pytest.raises(ValueError, match="ce_scale"):
        validate_checkpoint_config(_ckpt(saved), dict(BASE), "bad.pt")


def test_key_missing_from_checkpoint_raises():
    saved = dict(BASE)
    del saved["ce_scale"]  # a stale snapshot from before the key existed
    with pytest.raises(ValueError, match="not by the checkpoint.*ce_scale"):
        validate_checkpoint_config(_ckpt(saved), dict(BASE), "stale.pt")


def test_key_missing_from_config_raises():
    current = dict(BASE)
    del current["ce_scale"]  # a config from before the key existed
    with pytest.raises(ValueError, match="not by your config.*ce_scale"):
        validate_checkpoint_config(_ckpt(dict(BASE)), current, "newer-than-config.pt")


def test_key_absent_from_both_sides_is_not_an_inconsistency():
    # chirality is critical but optional; when NEITHER side records it
    # both were built under the same default, so nothing is inconsistent.
    assert "chirality" in CRITICAL_CONFIG_KEYS
    saved = dict(BASE)
    current = dict(BASE)
    validate_checkpoint_config(_ckpt(saved), current, "ok.pt")
