"""Train the RT and CCS models with the SAME encoder as the MS/MS release (MolConv-bond).

Why not scripts/train.py: `MolNetScalar.forward(x, mask, env, idx_base)` does not accept
`neighbor_idx` / `neighbor_mask`. Its Encoder supports bond mode, but the wrapper never passes the
bond graph through, so the shipped scalar-target path silently runs kNN neighbourhoods — a
different architecture from the released MS/MS model (measured worse on MS/MS: 0.489 vs 0.513).
That wrapper should be fixed for the release; this script exists so RT/CCS can train with parity
in the meantime.

Differences from the MS/MS trainer, all inherent to the task:
  * scalar target, so out_dim=1 and the bidirectional head does not apply (it indexes m/z bins)
  * MSE loss on standardised targets; MAE and median relative error reported in the original units
  * targets are standardised using TRAIN statistics only -- fitting the scaler on all data would
    leak test information through the mean and variance

Usage: python scripts/train_rt_ccs.py --task ccs --gpu 0
"""
import os

# Data and checkpoints are resolved relative to the CURRENT WORKING DIRECTORY, so
# these scripts work from any clone. Config files come from the installed
# molnetpack via config_path(), never from a path next to this file.
import argparse
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

from molnetpack import config_path

from molnetpack.model import MolNetScalar
from molnetpack.utils import make_idx_base

K = 6

# The encoder every released model starts from. RT, CCS and MS/MS must share it: the
# fine-tuned encoders are meant to be the same network specialised three ways, and a
# per-task pretraining would quietly break that.
DEFAULT_PRETRAIN = "check_point/molnet_pre_geobond.pt"


class OthDS(Dataset):
    def __init__(self, path, key):
        self.data = pickle.load(open(path, "rb")); self.key = key
        for d in self.data:
            d["mol"] = np.asarray(d["mol"], dtype=np.float32)
            d["maskpts"] = ~np.all(d["mol"] == 0, axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        return (torch.from_numpy(d["mol"]), torch.from_numpy(d["maskpts"]),
                torch.from_numpy(d["neighbor_idx"]), torch.from_numpy(d["neighbor_mask"]),
                torch.from_numpy(np.asarray(d["env"], np.float32)),
                torch.tensor(float(np.asarray(d[self.key]).ravel()[0]), dtype=torch.float32))


# The scalar model that used to be defined here is now the shipped `molnetpack.MolNetScalar`:
# the same MolConv-bond encoder as the released MS/MS model with a 1-dimensional head. Building
# it from the library guarantees this script trains the architecture that the release actually
# ships, and that its checkpoint loads without any key renaming.


def build_scalar_model(cfg, env_dim):
    """MolNetScalar sized for this task. `add_num` must equal the width of the env vector the data
    carries -- 1 placeholder column for RT, a 6-way adduct one-hot for CCS."""
    cfg = dict(cfg)
    cfg["add_num"] = int(env_dim)
    return MolNetScalar(cfg)


def warm_start_encoder(model, path, device=None):
    """Load a pretraining checkpoint into `model.encoder`. Every failure mode is loud.

    Earlier this was inlined as `if a.pretrain and os.path.exists(a.pretrain)`, which trained
    from scratch when the file was missing while the log line -- which did not name the file --
    still read `warm-started encoder`. A released model could therefore have had no pretraining
    at all with nothing in the record to show it. RT and CCS are required to start from the same
    encoder as MS/MS, so a skipped or partial load must stop the run.
    """
    if not path:
        print("NO pretraining: training the encoder from scratch", flush=True)
        return None
    if not os.path.exists(path):
        raise SystemExit(f"--pretrain {path} does not exist. Pass --pretrain '' to train from "
                         f"scratch deliberately.")
    state = torch.load(path, map_location=device or "cpu", weights_only=False)["encoder_state_dict"]
    missing, unexpected = model.encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{path} does not fit this encoder: {len(missing)} missing, "
                         f"{len(unexpected)} unexpected. A partial load would leave most of the "
                         f"encoder randomly initialised while still reporting a warm start.")
    print(f"warm-started encoder from {path} | missing 0 unexpected 0", flush=True)
    return path


def run(model, loader, dev, npoint, mu, sd, opt=None, clip=0.0, frozen=False):
    train = opt is not None; model.train(train)
    if frozen:
        model.encoder.eval()      # a frozen encoder must not update its normalisation statistics
    se = ae = n = 0.0; rel = []
    for mol, mask, nidx, nmask, env, y in loader:
        x = mol.to(dev).permute(0, 2, 1); mask = mask.to(dev)
        nidx, nmask, env, y = nidx.to(dev), nmask.to(dev), env.to(dev), y.to(dev)
        ib = make_idx_base(x.size(0), npoint, dev)
        with torch.set_grad_enabled(train):
            pred_z = model(x, mask, env, ib, nidx, nmask)
            loss = nn.functional.mse_loss(pred_z, (y - mu) / sd)
            if train:
                opt.zero_grad(); loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
        with torch.no_grad():
            p = pred_z * sd + mu                      # back to original units
            se += float(((p - y) ** 2).sum()); ae += float((p - y).abs().sum())
            rel.append(((p - y).abs() / y.clamp(min=1e-6)).cpu())
            n += y.numel()
    rel = torch.cat(rel)
    return (se / n) ** 0.5, ae / n, float(rel.median()) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rt", "ccs"], required=True)
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrain", default=DEFAULT_PRETRAIN,
                    help="encoder checkpoint to warm-start from; '' trains from scratch")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--set", action="append", default=[])
    # CCS has 3,563 training rows against a 17.8M-parameter encoder (~5,000 params per row even
    # with a trivial head). Freezing the encoder makes it a linear/shallow probe on geo-SSL
    # features, which is the standard move at that data scale. Measured on MS/MS with ~100x more
    # data the ordering was the other way (full fine-tune 0.4971 vs linear probe 0.4568), so this
    # is worth testing rather than assuming.
    ap.add_argument("--freeze_encoder", action="store_true")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}")

    _yml = yaml.safe_load(open(config_path("molnet.yml")))
    for kv in a.set:
        k, v = kv.split("=", 1); sec, key = k.split(".", 1)
        if sec not in _yml or key not in _yml[sec]:
            raise SystemExit(f"--set {kv}: no such config key {sec}.{key}")
        old = _yml[sec][key]
        # coerce to the EXISTING type. `type(old)(v)` is wrong for lists -- list("[512, 256]")
        # splits the string into characters -- and for bools, where bool("false") is True.
        if isinstance(old, bool):
            new = v.strip().lower() in ("1", "true", "yes")
        elif isinstance(old, (list, tuple, dict)):
            new = yaml.safe_load(v)
            if not isinstance(new, type(old)):
                raise SystemExit(f"--set {kv}: expected {type(old).__name__}, got {new!r}")
        else:
            new = type(old)(v)
        _yml[sec][key] = new
        print(f"override {sec}.{key}: {old} -> {_yml[sec][key]}", flush=True)
    cfg, tc = _yml["model"], _yml["train"]
    ckpt = a.ckpt or f"check_point/release_{a.task}.pt"

    tr = OthDS(f"data/{a.task}_bond_train.pkl", a.task)
    va = OthDS(f"data/{a.task}_bond_val.pkl", a.task)
    te = OthDS(f"data/{a.task}_bond_test.pkl", a.task)
    env_dim = int(np.asarray(tr.data[0]["env"]).shape[0])
    ys = np.array([float(np.asarray(d[a.task]).ravel()[0]) for d in tr.data])
    mu, sd = float(ys.mean()), float(ys.std())         # TRAIN statistics only
    print(f"[{a.task}] train {len(tr)} / val {len(va)} / test {len(te)} | env_dim {env_dim} | "
          f"target mean {mu:.2f} sd {sd:.2f}", flush=True)

    bs = int(tc["batch_size"])
    trl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
    val = DataLoader(va, batch_size=bs, num_workers=4, pin_memory=True)
    tel = DataLoader(te, batch_size=bs, num_workers=4, pin_memory=True)

    model = build_scalar_model(cfg, env_dim).to(dev)
    warm_start_encoder(model, a.pretrain, dev)
    if a.freeze_encoder:
        for prm in model.encoder.parameters():
            prm.requires_grad = False
        model.encoder.eval()
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"#params {tot:,} (trainable {trn:,}{' — ENCODER FROZEN' if a.freeze_encoder else ''})",
          flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=float(tc["lr"]),
                            weight_decay=float(tc["weight_decay"]))
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    clip = float(tc.get("grad_clip", 0.0)); pat = int(tc["early_stop_patience"])
    best, bad = 1e18, 0
    for ep in range(1, int(tc["epochs"]) + 1):
        trm = run(model, trl, dev, int(cfg["max_atom_num"]), mu, sd, opt, clip, a.freeze_encoder)
        vam = run(model, val, dev, int(cfg["max_atom_num"]), mu, sd)
        sch.step(vam[1]); flag = ""
        if vam[1] < best:
            best, bad, flag = vam[1], 0, " *best*"
            # `pretrain` is recorded so a released model's provenance is a fact in the file rather
            # than something to be reconstructed from logs later.
            torch.save({"model_state_dict": model.state_dict(), "val_mae": vam[1],
                        "mu": mu, "sd": sd, "task": a.task,
                        "pretrain": a.pretrain or None}, ckpt)
        else:
            bad += 1
        print(f"[{a.task}] epoch {ep:3d} | train RMSE {trm[0]:.3f} MAE {trm[1]:.3f} | "
              f"val RMSE {vam[0]:.3f} MAE {vam[1]:.3f} medRel {vam[2]:.2f}%{flag}", flush=True)
        if bad >= pat:
            print(f"early stop (best val MAE {best:.3f})", flush=True); break
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model_state_dict"])
    tem = run(model, tel, dev, int(cfg["max_atom_num"]), mu, sd)
    print(f"\nDONE [{a.task}] best val MAE {best:.3f} | TEST RMSE {tem[0]:.3f} MAE {tem[1]:.3f} "
          f"medRel {tem[2]:.2f}% (single evaluation) -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
