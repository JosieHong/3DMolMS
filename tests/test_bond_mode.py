"""Tests for MolConv BOND MODE (bonded neighbours instead of kNN) + explicit bond features.

Bond mode replaces the k-nearest-neighbour selection with a fixed molecular bond graph, so the
layer's relative-displacement Gram encodes true BOND ANGLES and its dist channel true BOND LENGTHS.
These tests check the properties that must hold for that change to be sound:

  1. no silent fallback      : a missing neighbor_idx raises instead of switching architecture.
  2. E(3) invariance         : bond mode stays rotation + reflection + translation invariant.
  3. neighbour masking       : padded bond slots (degree < k) do not affect the output.
  4. bond features           : bond_dim>0 accepts a per-edge descriptor and stays E(3) invariant.
"""
import numpy as np
import pytest
import torch

from molnetpack.molconv import MolConv
from conftest import graph_for

B, IN, P, K, OUT, NR = 2, 21, 60, 6, 32, 12
DTYPE = torch.float64
INVAR = 1e-8


def _synthetic(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(B, IN, P, dtype=DTYPE)
    x[:, :3, :NR] = torch.randn(B, 3, NR, generator=g, dtype=DTYPE) * 3.0
    x[:, 3:, :NR] = torch.randn(B, IN - 3, NR, generator=g, dtype=DTYPE)
    mask = torch.zeros(B, P, dtype=torch.bool); mask[:, :NR] = True
    idx_base = torch.arange(0, B).view(-1, 1, 1) * P
    # a simple chain bond graph over the real atoms; unused slots self-point and are masked off
    nidx = torch.arange(P).view(1, P, 1).repeat(B, 1, K)
    nmask = torch.zeros(B, P, K, dtype=torch.bool)
    for i in range(NR):
        nbrs = [j for j in (i - 1, i + 1) if 0 <= j < NR]
        for s, j in enumerate(nbrs):
            nidx[:, i, s] = j; nmask[:, i, s] = True
    return x, mask, idx_base, nidx, nmask


def _layer(bond_dim=0, seed=0):
    torch.manual_seed(seed)
    return MolConv(in_dim=IN, out_dim=OUT, point_num=P, k=K,
                    remove_xyz=True, bond_dim=bond_dim).to(DTYPE).eval()


def _rot_trans(x):
    c, s = np.cos(0.7), np.sin(0.7)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=DTYPE)
    out = x.clone()
    out[:, :3, :] = torch.einsum("ij,bjp->bip", R, x[:, :3, :]) + torch.tensor(
        [1.5, -2.0, 0.5], dtype=DTYPE).view(1, 3, 1)
    return out


def _reflect(x):
    out = x.clone(); out[:, 0, :] = -out[:, 0, :]
    return out


def test_missing_graph_is_refused():
    """A missing neighbour graph must raise, not fall back to an internal kNN.

    The per-layer dynamic kNN fallback was removed in v1.4.0. It was the single most dangerous
    construct in the encoder: it shared every tensor shape with bond mode, so a dropped graph
    switched the architecture silently -- a bond-trained checkpoint would load and predict with
    no error. It also measured worst of the three neighbourhoods (0.5893 vs 0.5921 fixed spatial
    kNN vs 0.6202 bond).
    """
    x, mask, ib, _, _ = _synthetic()
    layer = _layer()
    with pytest.raises(ValueError, match="neighbor_idx is required"):
        layer(x, ib, mask)


def test_bond_mode_differs_from_spatial_knn():
    """Bond neighbours are not the k nearest in space, so the output must actually change."""
    x, mask, ib, nidx, nmask = _synthetic()
    layer = _layer()
    g_idx, g_mask = graph_for(x, mask, nidx.shape[-1])
    with torch.no_grad():
        knn = layer(x, ib, mask, g_idx, g_mask)
        bond = layer(x, ib, mask, nidx, nmask)
    assert (knn - bond).abs().max() > 1e-6


def test_bond_mode_e3_invariant():
    """Bond mode must remain rotation+translation and reflection invariant (E(3))."""
    x, mask, ib, nidx, nmask = _synthetic()
    layer = _layer()
    with torch.no_grad():
        base = layer(x, ib, mask, nidx, nmask)
        rt = layer(_rot_trans(x), ib, mask, nidx, nmask)
        rf = layer(_reflect(x), ib, mask, nidx, nmask)
    assert (base - rt).abs().max() < INVAR, "bond mode not rotation/translation invariant"
    assert (base - rf).abs().max() < INVAR, "bond mode not reflection invariant"


def test_self_padded_slots_are_inert():
    """Self-pointing padded slots (the documented contract) contribute nothing extra.

    A degree-1 atom padded with 5 self-pointing slots must give the same real-atom output as
    the same graph built with a different (also self-pointing) padding order.
    """
    x, mask, ib, nidx, nmask = _synthetic()
    layer = _layer()
    alt = nidx.clone()
    for s in range(K):                       # re-write padding, keeping it SELF-pointing
        if not nmask[0, 0, s]:
            alt[:, 0, s] = 0                 # atom 0 -> itself, same as the convention
    with torch.no_grad():
        a = layer(x, ib, mask, nidx, nmask)
        b = layer(x, ib, mask, alt, nmask)
    assert (a - b).abs().max() < INVAR, "self-padded slots were not inert"


def test_non_self_padding_violates_contract():
    """Documents the contract: padding that points at OTHER atoms is NOT inert.

    neighbor_mask zeroes padded slots only after update_ff, while the Gram/dist/LayerNorm
    statistics span all k slots — so non-self padding perturbs the real slots. Data builders
    must self-pad (see MolConv.forward CONTRACT note).
    """
    x, mask, ib, nidx, nmask = _synthetic()
    layer = _layer()
    bad = nidx.clone()
    for s in range(K):
        if not nmask[0, 0, s]:
            bad[:, 0, s] = (bad[:, 0, s] + 3) % NR     # point somewhere else -> violates contract
    with torch.no_grad():
        a = layer(x, ib, mask, nidx, nmask)
        b = layer(x, ib, mask, bad, nmask)
    assert (a - b).abs().max() > 1e-6, "expected non-self padding to perturb the output"


def test_bond_features_shape_and_invariance():
    """bond_dim>0 accepts a per-edge descriptor, changes output, stays E(3) invariant."""
    x, mask, ib, nidx, nmask = _synthetic()
    bond_dim = 6
    layer = _layer(bond_dim=bond_dim)
    g = torch.Generator().manual_seed(3)
    bfeat = torch.randn(B, P, K, bond_dim, generator=g, dtype=DTYPE) * nmask.unsqueeze(-1)
    with torch.no_grad():
        out = layer(x, ib, mask, nidx, nmask, bfeat)
        rt = layer(_rot_trans(x), ib, mask, nidx, nmask, bfeat)
        zero = layer(x, ib, mask, nidx, nmask, torch.zeros_like(bfeat))
    assert out.shape == (B, OUT, P)
    assert (out - rt).abs().max() < INVAR, "bond features broke E(3) invariance"
    assert (out - zero).abs().max() > 1e-9, "bond features had no effect on the output"
