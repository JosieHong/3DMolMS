"""Acceptance test for the release: can the SHIPPED library actually serve the released models?

This is the check that was missing. The release models were developed in `experiments/`, which
defines its own `MolNetBidir` (encoder + trunk + forward/reverse/gate heads) and `MolNetScalar`
(encoder + `self.head`). Neither class exists in `molnetpack`, and nothing ever verified that a
checkpoint produced by the experiment trainers can be loaded by the library a user pip-installs.
It cannot -- every one of the four fails.

Two distinct failure modes are covered here, and the SECOND is the dangerous one:

  * ARCHITECTURE MISMATCH (loud). `MolNet_MS` has `decoder.*`; the MS/MS checkpoints have
    `trunk.*` + `forw/rev/gate.*`. `MolNetScalar` has `decoder.*`; the RT/CCS checkpoints have
    `head.*`. load_state_dict raises, so a user sees an error rather than a wrong answer.

  * ENCODER MODE MISMATCH (SILENT). The release models were trained in BOND mode (the encoder
    aggregates over covalent neighbours). No Dataset in `molnetpack/dataset.py` returns
    `neighbor_idx`/`neighbor_mask`, and `scripts/predict.py` calls `model(x, mask, env, idx_base)`,
    so the shipped inference path runs kNN mode. Bond and kNN mode have IDENTICAL parameter
    shapes -- the checkpoint loads without complaint and predicts from a different architecture.
    Measured output cosine between the two modes on one batch: 0.96.

Run: python -m pytest tests/test_release_checkpoints.py -q
"""
import os
import sys

import pytest
import torch
import yaml

M3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, M3)

from molnetpack.model import MolNet_MS, MolNetScalar
from conftest import graph_for

# (checkpoint, shipped class, config supplying the model block)
# molnet_orbitrap_v1.4.0.pt is absent until the Orbitrap arms finish; the fixture skips on it.
RELEASE = [
    ("molnet_qtof_v1.4.0.pt", MolNet_MS, "molnet.yml"),
    ("molnet_orbitrap_v1.4.0.pt", MolNet_MS, "molnet.yml"),
    ("molnet_rt_v1.4.0.pt", MolNetScalar, "molnet_rt_tl.yml"),
    ("molnet_ccs_v1.4.0.pt", MolNetScalar, "molnet_ccs_tl.yml"),
]


def _cfg(name):
    from molnetpack import config_path
    return yaml.safe_load(open(config_path(name)))["model"]


def _ckpt(name):
    path = f"{M3}/check_point/{name}"
    if not os.path.exists(path):
        pytest.skip(f"{name} not built yet")
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.parametrize("ckpt_name,cls,cfg_name", RELEASE)
def test_release_checkpoint_loads_into_shipped_class(ckpt_name, cls, cfg_name):
    """A released checkpoint must load into the class a pip-installed user gets."""
    ckpt = _ckpt(ckpt_name)
    model = cls(_cfg(cfg_name))
    model.load_state_dict(ckpt["model_state_dict"])


@pytest.mark.parametrize("ckpt_name,cls,cfg_name", RELEASE)
def test_release_checkpoint_records_its_architecture(ckpt_name, cls, cfg_name):
    """A checkpoint must carry enough metadata to be loaded correctly without guesswork.

    Today they carry only {model_state_dict, val_cos, seed} or {model_state_dict, val_mae, mu, sd,
    task}. Nothing records the encoder settings, the bond-graph contract, the adduct one-hot layout,
    resolution or max_mz -- so nothing can DETECT the silent mode mismatch below.
    """
    ckpt = _ckpt(ckpt_name)
    assert "config" in ckpt, (
        f"{ckpt_name} carries no config; keys are {sorted(ckpt)}. Without the resolved config the "
        f"loader cannot verify encoder mode, adduct layout or binning."
    )


def test_bond_and_knn_modes_are_not_distinguishable_by_shape():
    """Documents WHY the mode mismatch is silent, so nobody 'fixes' it by trusting load_state_dict.

    If this ever starts failing because the shapes diverged, the silent-failure risk is gone and
    the guard in the loader can be relaxed.
    """
    cfg = _cfg("molnet.yml")
    torch.manual_seed(0)
    model = MolNet_MS(cfg).eval()

    b, n, d, k = 2, int(cfg["max_atom_num"]), int(cfg["in_dim"]), int(cfg["k"])
    x = torch.zeros(b, d, n)
    x[:, :, :12] = torch.randn(b, d, 12)
    mask = torch.zeros(b, n, dtype=torch.bool)
    mask[:, :12] = True
    env = torch.randn(b, int(cfg["add_num"]))
    nidx = torch.randint(0, 12, (b, n, k))
    nmask = torch.zeros(b, n, k, dtype=torch.bool)
    nmask[:, :12, :3] = True

    prec_idx = torch.full((b,), 900, dtype=torch.long)
    g_idx, g_mask = graph_for(x, mask, k)
    with torch.no_grad():
        knn = model(x, mask, env, neighbor_idx=g_idx, neighbor_mask=g_mask, prec_idx=prec_idx)
        bond = model(x, mask, env, neighbor_idx=nidx, neighbor_mask=nmask, prec_idx=prec_idx)

    assert knn.shape == bond.shape, "shapes diverged -- silent mismatch no longer possible"
    cos = torch.nn.functional.cosine_similarity(knn, bond, dim=1).mean().item()
    assert cos < 0.999, (
        "kNN and bond mode produced the same output; the mode would then be harmless, which "
        "contradicts the measured 0.513 vs 0.489 MS/MS gap"
    )


@pytest.mark.parametrize("ckpt_name,cls,cfg_name", RELEASE)
def test_reproduction_scripts_build_the_released_architecture(ckpt_name, cls, cfg_name):
    """`scripts/` must be able to rebuild the released models without `experiments/`.

    experiments/ is private, so the public reproduction path is scripts/. Those trainers construct
    their models from `molnetpack` rather than from local copies -- if a copy ever drifted from the
    shipped class it would still train happily and produce a checkpoint the library cannot load.
    """
    sys.path.insert(0, os.path.join(M3, "scripts"))
    ckpt = _ckpt(ckpt_name)
    cfg = _cfg(cfg_name)
    if cls is MolNet_MS:
        model = MolNet_MS(cfg, out_relu=False)
    else:
        from train_rt_ccs import build_scalar_model
        model = build_scalar_model(cfg, cfg["add_num"])
    model.load_state_dict(ckpt["model_state_dict"])


def test_shipped_dataset_emits_the_bond_graph():
    """The library must be able to FEED bond mode, not merely accept it in forward()."""
    import inspect

    from molnetpack.dataset import MolMS_Dataset

    src = inspect.getsource(MolMS_Dataset.__getitem__)
    assert "neighbor_idx" in src, (
        "MolMS_Dataset.__getitem__ does not return neighbor_idx/neighbor_mask, so no shipped "
        "training or inference loop can run the encoder in the mode the release was trained in."
    )


def test_mirroring_is_a_no_op_so_the_augmentation_stays_removed():
    """Pins the reason `data_augmentation` was removed in v1.4.0.

    The old MolMS_Dataset doubled the dataset by mirroring the x coordinate. The encoder is
    E(3)-invariant in every shipped setting and reflection is an element of E(3), so the mirrored
    copy is bit-identical: the "augmented" half was exact duplicates, and epochs took twice as
    long for zero information. If this ever stops being a no-op (e.g. chirality=True becomes the
    default) mirroring becomes real augmentation and could be reinstated.
    """
    cfg = _cfg("molnet.yml")
    torch.manual_seed(0)
    model = MolNet_MS(cfg).eval()

    b, n, d = 2, int(cfg["max_atom_num"]), int(cfg["in_dim"])
    x = torch.zeros(b, d, n)
    x[:, :, :12] = torch.randn(b, d, 12)
    mask = torch.zeros(b, n, dtype=torch.bool)
    mask[:, :12] = True
    env = torch.randn(b, int(cfg["add_num"]))
    prec_idx = torch.full((b,), 900, dtype=torch.long)

    flipped = x.clone()
    flipped[:, 0, :] *= -1  # exactly what the old MolMS_Dataset did: mol[:, 0] *= -1

    kk = int(cfg["k"])
    with torch.no_grad():
        gi, gm = graph_for(x, mask, kk)
        fi, fm = graph_for(flipped, mask, kk)
        delta = (model(x, mask, env, neighbor_idx=gi, neighbor_mask=gm, prec_idx=prec_idx)
                 - model(flipped, mask, env, neighbor_idx=fi, neighbor_mask=fm,
                         prec_idx=prec_idx)).abs().max().item()

    assert delta == 0.0, f"mirroring now changes the output by {delta:.3e}"

    import inspect

    from molnetpack.dataset import MolMS_Dataset

    assert "data_augmentation" not in inspect.signature(MolMS_Dataset.__init__).parameters, (
        "the no-op mirroring augmentation is back"
    )
