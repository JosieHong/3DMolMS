"""Tests for the transfer-learning (fine-tuning) load path.

The contract: ``transfer=True`` moves ONLY the pretrained ``encoder.*`` weights into the
target model. Head weights are never transferred — the head's first layer consumes
``[emb_dim + add_num]`` inputs, so its columns encode the SOURCE task's env layout
(collision energy + adduct one-hot), which is meaningless at a different task. Auxiliary
heads on the pretraining source are ignored the same way. Consistency is validated on the
ENCODER-defining keys only: a pretraining source and a fine-tune target are expected to
differ on task keys (add_num, resolution, max_mz, ce_scale), and that must not block a
transfer.

Also pins a fixed bug: the old filter was `not startswith("decoder")`, a leftover from when
every head was named `decoder`. For MolNet_MS (head = trunk/forw/rev/gate) it transferred
the head too and froze EVERY parameter, so MS/MS fine-tuning trained nothing.

Run: python -m pytest tests/test_transfer_loading.py -q
"""
import pytest
import torch

from molnetpack.checkpoints import load_weights
from molnetpack.model import MolNet_MS, MolNetScalar, _build_encoder

ENCODER_CFG = {
    "in_dim": 21, "emb_dim": 64, "max_atom_num": 40, "k": 4,
    "encode_layers": [32, 32],
}
PRETRAIN_CFG = dict(ENCODER_CFG)
MSMS_CFG = dict(ENCODER_CFG, add_num=6, resolution=0.2, max_mz=100.0,
                decode_layers=[64, 64], dropout=0.1, ce_scale="fraction")
RT_CFG = dict(ENCODER_CFG, add_num=1, decode_layers=[64, 64], dropout=0.1)


class _PretrainSource(torch.nn.Module):
    """Stand-in for any pretraining model: the shared encoder plus an auxiliary head.

    Built here rather than imported so the transfer contract is tested independently of which
    pretext produced the checkpoint. The auxiliary head is the point -- transfer must ignore it,
    exactly as it ignores a task head. (This used to be MolNet_SSL, removed in v1.4.0; the
    released encoder comes from the geometric pretext in scripts/pretrain_geo.py, whose head is
    likewise an auxiliary the fine-tuned models never load.)
    """

    def __init__(self, cfg):
        super().__init__()
        self.encoder = _build_encoder(cfg)
        self.aux_head = torch.nn.Linear(int(cfg["emb_dim"]), 8)


@pytest.fixture()
def ssl_checkpoint(tmp_path):
    """A pretraining checkpoint: full config with a 'model' section, encoder.* + aux head."""
    torch.manual_seed(0)
    model = _PretrainSource(PRETRAIN_CFG)
    path = tmp_path / "pretrained.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"model": PRETRAIN_CFG, "train": {}},
    }, path)
    return str(path), model


def _transfer(target, path, cfg, **kw):
    load_weights(target, path, torch.device("cpu"), transfer=True,
                 current_model_config=cfg, **kw)


def test_pretrain_to_msms_transfers_encoder_only(ssl_checkpoint):
    path, source = ssl_checkpoint
    torch.manual_seed(1)
    target = MolNet_MS(MSMS_CFG)
    fresh_head = {k: v.clone() for k, v in target.state_dict().items()
                  if not k.startswith("encoder.")}

    _transfer(target, path, MSMS_CFG)

    # encoder weights match the pretrained source ...
    for k, v in source.state_dict().items():
        if k.startswith("encoder."):
            assert torch.equal(target.state_dict()[k], v), k
    # ... head weights are untouched fresh init (env-consuming trunk included)
    for k, v in fresh_head.items():
        assert torch.equal(target.state_dict()[k], v), k


def test_encoder_frozen_head_trainable(ssl_checkpoint):
    path, _ = ssl_checkpoint
    target = MolNet_MS(MSMS_CFG)
    _transfer(target, path, MSMS_CFG)
    for name, p in target.named_parameters():
        if name.startswith("encoder."):
            assert not p.requires_grad, name
        else:
            # the fixed bug: trunk/forw/rev/gate used to end up frozen too
            assert p.requires_grad, name


def test_freeze_encoder_false_trains_everything(ssl_checkpoint):
    path, _ = ssl_checkpoint
    target = MolNet_MS(MSMS_CFG)
    _transfer(target, path, MSMS_CFG, freeze_encoder=False)
    assert all(p.requires_grad for p in target.parameters())


def test_env_width_change_does_not_block_transfer(ssl_checkpoint):
    """pretrain (no env) -> RT (add_num=1): task keys differ, encoder keys agree -> must work."""
    path, source = ssl_checkpoint
    target = MolNetScalar(RT_CFG)
    _transfer(target, path, RT_CFG)
    for k, v in source.state_dict().items():
        if k.startswith("encoder."):
            assert torch.equal(target.state_dict()[k], v), k


def test_encoder_key_mismatch_refused(ssl_checkpoint):
    path, _ = ssl_checkpoint
    bad = dict(MSMS_CFG, k=6)  # pretrained with k=4
    target = MolNet_MS(bad)
    with pytest.raises(ValueError, match="different encoder"):
        _transfer(target, path, bad)


def test_source_without_encoder_weights_refused(tmp_path):
    path = tmp_path / "headless.pt"
    torch.save({"model_state_dict": {"decoder.fc.weight": torch.zeros(1, 1)},
                "config": {"model": PRETRAIN_CFG}}, path)
    target = MolNet_MS(MSMS_CFG)
    with pytest.raises(ValueError, match="no 'encoder"):
        _transfer(target, str(path), MSMS_CFG)


def test_freeze_encoder_without_transfer_is_refused(ssl_checkpoint):
    """freeze_encoder is meaningless without transfer=True; silently ignoring it would let a
    user believe they froze (or unfroze) an encoder when nothing happened."""
    path, _ = ssl_checkpoint
    target = MolNet_MS(MSMS_CFG)
    for value in (True, False):
        with pytest.raises(ValueError, match="freeze_encoder only applies"):
            load_weights(target, path, torch.device("cpu"),
                         transfer=False, freeze_encoder=value)


def test_unspecified_freeze_encoder_defaults_to_frozen(ssl_checkpoint):
    path, _ = ssl_checkpoint
    target = MolNet_MS(MSMS_CFG)
    load_weights(target, path, torch.device("cpu"), transfer=True,
                 current_model_config=MSMS_CFG)  # freeze_encoder unspecified
    assert all(not p.requires_grad for n, p in target.named_parameters()
               if n.startswith("encoder."))
