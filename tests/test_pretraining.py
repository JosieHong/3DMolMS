"""The pretrained encoder behind every released model must stay loadable by the current code.

All four v1.4.0 checkpoints warm-start from `check_point/molnet_pre_geobond.pt`, produced by
`scripts/pretrain_geo.py` (geometric SSL: coordinate denoising into the per-atom bond Gram).

Unlike the fine-tuning trainers, that script builds its model INLINE (`GeoBondSSL`) rather than
from `molnetpack`, so nothing structural stops its encoder from drifting away from the shipped
`Encoder`. If it drifts, pretraining still runs happily and still writes a checkpoint -- but the
weights no longer fit the models they are supposed to initialise, and the failure only surfaces
much later as a bad `--pretrain` load. These tests are the guard.

Run: python -m pytest tests/test_pretraining.py -q
"""
import os
import sys

import pytest
import torch
import yaml

M3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, M3)
sys.path.insert(0, os.path.join(M3, "scripts"))

from molnetpack import config_path
from molnetpack.model import MolNet_MS, MolNetScalar
from conftest import graph_for

PRETRAINED = os.path.join(M3, "check_point", "molnet_pre_geobond.pt")


def _weights():
    if not os.path.exists(PRETRAINED):
        pytest.skip("molnet_pre_geobond.pt not present")
    return torch.load(PRETRAINED, map_location="cpu", weights_only=False)


def test_pretraining_script_builds_the_released_encoder():
    """`scripts/pretrain_geo.py` must still construct the encoder whose weights were released."""
    from pretrain_geo import CFG, GeoBondSSL

    ckpt = _weights()
    model = GeoBondSSL(CFG)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)


@pytest.mark.parametrize("cls,cfg_name", [
    (MolNet_MS, "molnet.yml"),
    (MolNetScalar, "molnet_rt_tl.yml"),
    (MolNetScalar, "molnet_ccs_tl.yml"),
])
def test_pretrained_encoder_fits_every_finetuning_target(cls, cfg_name):
    """The whole point of pretraining is transfer, so the weights must fit with nothing missing.

    `strict=False` is what the trainers use (the head is deliberately left fresh), which means a
    silently-empty transfer would NOT raise there -- hence the explicit count assertions here.
    """
    ckpt = _weights()
    model = cls(yaml.safe_load(open(config_path(cfg_name)))["model"])
    missing, unexpected = model.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
    assert not missing, f"{cfg_name}: encoder tensors absent from the pretrained checkpoint: {missing[:5]}"
    assert not unexpected, f"{cfg_name}: pretrained tensors the encoder has no slot for: {unexpected[:5]}"


def test_pretrained_checkpoint_identifies_its_pretext():
    """`sigma` is written only by the geometric pretext; the topological one writes val_type_acc.

    This is the evidence that ties the released encoder to the script we ship, so it is worth
    asserting rather than leaving to a filename.
    """
    ckpt = _weights()
    assert "sigma" in ckpt, f"no 'sigma' -- this is not a geo-pretrained checkpoint; keys {sorted(ckpt)}"
    assert ckpt["k"] == 6, "k must match the fine-tuning encoder's neighbour count for transfer"


def test_all_three_tasks_share_one_pretrained_encoder():
    """RT, CCS and MS/MS must warm-start from the SAME encoder.

    The released models are meant to be one network specialised three ways. Per-task pretraining
    would break that silently -- every run would still print 'warm-started encoder' and every
    metric would still look reasonable.
    """
    from train_msms_release import DEFAULT_PRETRAIN as MSMS
    from train_rt_ccs import DEFAULT_PRETRAIN as SCALAR

    assert MSMS == SCALAR, (
        f"MS/MS defaults to {MSMS!r} but RT/CCS defaults to {SCALAR!r}; the two trainers have "
        f"drifted apart on which encoder they start from."
    )
    assert os.path.basename(MSMS) == os.path.basename(PRETRAINED), (
        f"trainers default to {MSMS!r}, which is not the released pretrained encoder"
    )


def test_missing_pretrain_file_stops_the_run():
    """A missing --pretrain must raise, not fall through to training from scratch.

    Tested through the real function rather than by grepping the source: the previous version of
    this test searched for the old `and os.path.exists(...)` idiom and matched the COMMENT that
    documents its removal, so it failed on correct code.
    """
    from train_rt_ccs import warm_start_encoder

    model = MolNet_MS(yaml.safe_load(open(config_path("molnet.yml")))["model"])
    with pytest.raises(SystemExit, match="does not exist"):
        warm_start_encoder(model, "check_point/definitely_not_here.pt")


def test_empty_pretrain_is_an_explicit_opt_out():
    from train_rt_ccs import warm_start_encoder

    model = MolNet_MS(yaml.safe_load(open(config_path("molnet.yml")))["model"])
    assert warm_start_encoder(model, "") is None


def test_partial_encoder_match_stops_the_run(tmp_path):
    """A checkpoint that only half-fits must fail rather than warm-start a mostly-random encoder.

    The trainers load with strict=False so the head can stay fresh; that same flag would happily
    accept an encoder sharing almost no keys.
    """
    from train_rt_ccs import warm_start_encoder

    path = tmp_path / "bad.pt"
    torch.save({"encoder_state_dict": {"hidden_layers.0.nonsense": torch.zeros(1)}}, path)
    model = MolNet_MS(yaml.safe_load(open(config_path("molnet.yml")))["model"])
    with pytest.raises(SystemExit, match="does not fit"):
        warm_start_encoder(model, str(path))


def test_scalar_model_output_shape_matches_targets():
    """MolNetScalar must return [B], not [B, 1].

    With a [B, 1] prediction against a [B] target, `MSELoss` broadcasts to [B, B]: it compares
    every prediction against every other row's target. Training proceeds, the number goes down,
    and the only signal is a UserWarning. This bit the RT reproduction script the moment its
    inline model (which squeezed) was replaced by the library class (which did not).
    """
    cfg = yaml.safe_load(open(config_path("molnet_rt_tl.yml")))["model"]
    model = MolNetScalar(cfg).eval()

    b, n, d = 4, int(cfg["max_atom_num"]), int(cfg["in_dim"])
    x = torch.zeros(b, d, n)
    x[:, :, :10] = torch.randn(b, d, 10)
    mask = torch.zeros(b, n, dtype=torch.bool)
    mask[:, :10] = True
    env = torch.zeros(b, int(cfg["add_num"]))

    nidx, nmask = graph_for(x, mask, int(cfg["k"]))
    with torch.no_grad():
        out = model(x, mask, env, neighbor_idx=nidx, neighbor_mask=nmask)
    assert out.shape == (b,), f"expected [B], got {tuple(out.shape)}"

    target = torch.randn(b)
    loss = torch.nn.functional.mse_loss(out, target)
    assert loss.shape == (), "loss is not a scalar -- the prediction broadcast against the target"


def test_library_ships_no_neighbour_selection():
    """The encoder has exactly one source of neighbours: the graph the caller supplies.

    No kNN, no fallback, no builder. Anything that constructs a graph is test scaffolding
    (tests/conftest.py) or preprocessing (bond_graph_array), never the model.
    """
    import molnetpack.molconv as molconv
    import molnetpack.steps as steps

    assert not hasattr(molconv, "spatial_knn_graph")
    assert not hasattr(steps, "neighbor_kwargs")


def test_no_neighbourhood_mode_is_configurable():
    """One neighbourhood, so no config key selects it -- a key with a single legal value is a
    trap, not an option."""
    for name in ("molnet.yml", "molnet_rt_tl.yml", "molnet_ccs_tl.yml"):
        cfg = yaml.safe_load(open(config_path(name)))["model"]
        assert "neighbor_mode" not in cfg, f"{name} still ships a neighbor_mode key"
