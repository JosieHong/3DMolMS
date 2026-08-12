"""Bidirectional spectrum head: reverse indexing and precursor masking.

Following NEIMS and MassFormer (Young et al. 2024, eqs 6-10), the spectrum head is not a plain
projection to m/z bins. It is two projections combined by a learned gate, plus a hard mask:

    y_F(s)_i               = (W_F s + b_F)_i                       forward
    y_R(s)_{m_p + tau - i} = (W_R s + b_R)_i                       reverse
    y_G(s)_i               = sigmoid(W_G s + b_G)_i                gate
    y_FR(s)_i              = y_G_i * y_F_i + (1 - y_G_i) * y_R_i
    y(s)_i                 = 1[i <= m_p + tau] * y_FR(s)_i

The forward head indexes bins from 0 upward, which is the natural frame for FRAGMENT masses. The
reverse head indexes them downward from the precursor, which is the natural frame for NEUTRAL
LOSSES -- a loss of 18 Da sits at the same offset for every molecule regardless of its mass. The
mask zeroes everything above the precursor, since a fragment cannot substantially exceed its parent.

MEASURED on this codebase (QTOF, 0.2 Da bins): plain MLP decoder 0.4131 -> bidirectional 0.4810,
+0.068 cosine and the single largest architectural gain found. The mechanism is visible in the
TRAIN cosine, which *falls* 0.9052 -> 0.6255: the mask deletes the entire above-precursor region
the plain decoder was free to memorise, so the model can no longer fit noise there and generalises
instead. This is NOT a capacity effect -- shrinking the plain decoder to [1024, 1024] HURT by 0.024.
"""
import numpy as np
import torch
import yaml
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from .data_utils.encoding import precursor_calculator


def reverse_prediction(rev, prec_idx, offset):
    """eq 7: y_R at bin (m_p + tau - i) takes value rev[i]  =>  out[j] = rev[m_p + tau - j].

    Args:
        rev:      [B, n_bins] raw reverse-head output.
        prec_idx: [B] long, precursor m/z expressed as a bin index.
        offset:   tolerance tau, in bins.
    """
    n_bins = rev.shape[1]
    j = torch.arange(n_bins, device=rev.device).unsqueeze(0)
    src = prec_idx.unsqueeze(1) + offset - j
    valid = (src >= 0) & (src < n_bins)
    return torch.gather(rev, 1, src.clamp(0, n_bins - 1)) * valid


def mask_prediction_by_mass(y, prec_idx, offset):
    """eq 10: zero every bin above the precursor (+ tolerance)."""
    j = torch.arange(y.shape[1], device=y.device).unsqueeze(0)
    return y * (j <= (prec_idx.unsqueeze(1) + offset))


# ---------------------------------------------------------------------------
# The adduct layout comes from the CONFIG, never from a hardcoded list.
#
# `encoding.precursor_type` maps each adduct to its one-hot vector, so it defines both the SET and
# the INDEX of every adduct in one place -- the same block the preprocessing uses to write `env`.
# Deriving the order from it means the model and the data cannot disagree.
# ---------------------------------------------------------------------------

_ADDUCT_ORDER_CACHE: dict = {}
_PRECURSOR_BIN_CACHE: dict = {}


def adduct_order(cfg_path):
    """-> list of adduct strings indexed by their position in the env one-hot."""
    if cfg_path not in _ADDUCT_ORDER_CACHE:
        with open(cfg_path) as f:
            pt = yaml.safe_load(f)["encoding"]["precursor_type"]
        order = [None] * len(next(iter(pt.values())))
        for name, onehot in pt.items():
            order[list(onehot).index(1)] = name
        if any(a is None for a in order):
            raise ValueError(f"encoding.precursor_type in {cfg_path} is not a clean one-hot set")
        _ADDUCT_ORDER_CACHE[cfg_path] = order
    return _ADDUCT_ORDER_CACHE[cfg_path]


def precursor_bin(smiles, env, resolution, n_bins, cfg_path, env_offset=1):
    """Exact mass + adduct shift -> bin index.

    Uses molnetpack's own `precursor_calculator` so the adduct chemistry (charge-2 division,
    negative mode, neutral losses) lives in ONE place.

    Args:
        env_offset: index in `env` where the adduct one-hot starts. For MS/MS this is 1, because
                    env[0] is the normalised collision energy.
    """
    names = adduct_order(cfg_path)
    ai = int(np.argmax(np.asarray(env)[env_offset:env_offset + len(names)]))
    key = (smiles, ai, resolution, n_bins)
    if key not in _PRECURSOR_BIN_CACHE:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            _PRECURSOR_BIN_CACHE[key] = 0
        else:
            try:
                mz = precursor_calculator(names[ai], ExactMolWt(mol))
            except ValueError:
                mz = 0.0
            _PRECURSOR_BIN_CACHE[key] = int(min(max(mz, 0.0) / resolution, n_bins - 1))
    return _PRECURSOR_BIN_CACHE[key]
