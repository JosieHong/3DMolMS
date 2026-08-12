"""Train MolNetMS in BOND MODE — MolConv with bonded neighbours instead of kNN.

Same MolConv architecture, same binned head, same env, same loss mean(1-cos) and eval metric as
the 0.489 kNN anchor. The ONLY change: neighbours are the molecular bond graph (neighbor_idx), so
MolConv's relative-displacement Gram encodes true BOND ANGLES and dist true BOND LENGTHS.
k is set to K=6 (max degree 4 + pad). Compare vs 0.489 (kNN MolConv) and 0.513 (standalone graph).

Usage: python scripts/train_msms_release.py --gpu 0 --epochs 200 --batch 32 --seed 0
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml

from molnetpack import config_path
from molnetpack.model import MolNet_MS
DATA_CFG = config_path("encoding_etkdgv3.yml")
from molnetpack.utils import make_idx_base
from molnetpack.spectrum_heads import (
                            precursor_bin, adduct_order)


# The bidirectional head that used to be defined here is now the shipped model: as of v1.4.0
# `molnetpack.MolNet_MS` IS encoder + trunk + forward/reverse/gate + precursor mask. Training
# against the library class is what makes this script reproduce a RELEASED checkpoint rather than
# a look-alike -- a local copy could drift from the shipped one without anything failing.


# The encoder every released model starts from. RT, CCS and MS/MS must share it: the
# fine-tuned encoders are meant to be the same network specialised three ways, and a
# per-task pretraining would quietly break that.
DEFAULT_PRETRAIN = "check_point/molnet_pre_geobond.pt"

K = 6   # neighbor slots (dataset max degree 4, self-padded to 6)


class BondDS(Dataset):
    def __init__(self, path, need_prec=False, resolution=0.2, nbin=7500):
        self.data = pickle.load(open(path, "rb"))
        for d in self.data:
            d["mol"] = np.asarray(d["mol"], dtype=np.float32)
            d["maskpts"] = ~np.all(d["mol"] == 0, axis=1)
        if need_prec:
            # precursor m/z -> bin index (precursor_bin caches internally across datasets)
            for d in self.data:
                d["prec_idx"] = precursor_bin(d["smiles"], d["env"], resolution, nbin, DATA_CFG)
        else:
            for d in self.data:
                d["prec_idx"] = 0

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        bf = d["neighbor_bondfeat"] if "neighbor_bondfeat" in d else np.zeros((300, K, 6), np.float32)
        return (torch.from_numpy(d["mol"]), torch.from_numpy(d["maskpts"]),
                torch.from_numpy(d["neighbor_idx"]), torch.from_numpy(d["neighbor_mask"]),
                torch.from_numpy(np.asarray(bf, dtype=np.float32)),
                torch.from_numpy(d["env"]).float(), torch.from_numpy(d["spec"]).float(),
                torch.tensor(d["prec_idx"], dtype=torch.long))


def msms_metric(pred, y):
    pred = pred.detach()
    pred_max = pred.max(dim=1, keepdim=True).values.clamp(min=1e-8)
    p = pred / pred_max
    p = torch.where(p > 0.01, p, torch.zeros_like(p))
    return F.cosine_similarity(torch.pow(p, 2), torch.pow(y, 2), dim=1).sum().item()


def run(model, loader, dev, npoint, opt=None, nogeo=False, bidir=False, clip=0.0):
    train = opt is not None; model.train(train)
    cos = nn.CosineSimilarity(dim=1); loss_sum = metric_sum = n = 0
    for mol, mask, nidx, nmask, bfeat, env, spec, pidx in loader:
        x = mol.to(dev).permute(0, 2, 1)           # [B, 21, N]
        if nogeo:
            x[:, :3, :] = 0.0                      # ABLATION: zero xyz, keep bonds + atom feats
        mask = mask.to(dev); nidx = nidx.to(dev); nmask = nmask.to(dev); bfeat = bfeat.to(dev)
        env = env.to(dev); y = spec.to(dev); bs = env.size(0); pidx = pidx.to(dev)
        ib = make_idx_base(bs, npoint, dev)
        with torch.set_grad_enabled(train):
            kw = {"neighbor_idx": nidx, "neighbor_mask": nmask, "bond_feat": bfeat}
            if bidir:
                kw["prec_idx"] = pidx
            pred = model(x, mask, env, ib, **kw)
            loss = torch.mean(1 - cos(pred, y))
            if train:
                opt.zero_grad(); loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
        with torch.no_grad():
            metric_sum += msms_metric(pred, y)
        loss_sum += loss.item() * bs; n += bs
    return loss_sum / n, metric_sum / n


def main():
    ap = argparse.ArgumentParser()
    # Only things that vary PER RUN live on the command line. The training recipe (epochs, batch,
    # lr, weight decay, grad clip, patience) and the architecture (decoder widths, bidirectional
    # head) come from molnet.yml, so a run cannot silently disagree with the released config.
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrain", default=DEFAULT_PRETRAIN,
                    help="encoder checkpoint to warm-start from; '' trains from scratch")
    ap.add_argument("--allow_biased_eval", action="store_true",
                    help="permit selecting on the test set when --val_data is empty")
    ap.add_argument("--train_data", default="data/orbi_hcd_train.pkl")
    ap.add_argument("--val_data", default="data/orbi_hcd_val.pkl",
                    help="early-stop / model-selection set; must NOT be the test set")
    ap.add_argument("--test_data", default="data/orbi_hcd_test.pkl")
    # Sweeps override config values WITHOUT reintroducing a flag per hyperparameter. The config
    # stays the single source of truth; overrides are explicit, logged, and dotted-path so a typo
    # fails loudly instead of being silently ignored.
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a molnet.yml value, e.g. --set train.lr=1e-3")
    ap.add_argument("--ckpt", default=""); a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev = torch.device(f"cuda:{a.gpu}")
    ckpt = a.ckpt or f"check_point/molnet_bond_s{a.seed}.pt"
    _yml = yaml.safe_load(open(config_path("molnet.yml")))
    for _kv in a.set:
        _k, _v = _kv.split("=", 1)
        _sec, _key = _k.split(".", 1)
        if _sec not in _yml or _key not in _yml[_sec]:
            raise SystemExit(f"--set {_kv}: no such config key {_sec}.{_key}")
        _old = _yml[_sec][_key]
        # coerce to the EXISTING type. type(old)(v) is wrong for lists: list("[512, 256]") splits
        # the string into single characters.
        if isinstance(_old, bool):
            _new = _v.strip().lower() in ("1", "true", "yes")
        elif isinstance(_old, (list, tuple, dict)):
            _new = yaml.safe_load(_v)
            if not isinstance(_new, type(_old)):
                raise SystemExit(f"--set {_kv}: expected {type(_old).__name__}, got {_new!r}")
        else:
            _new = type(_old)(_v)
        _yml[_sec][_key] = _new
        print(f"override {_sec}.{_key}: {_old} -> {_yml[_sec][_key]}", flush=True)
    cfg, tcfg = _yml["model"], _yml["train"]
    epochs   = int(tcfg["epochs"]);      batch    = int(tcfg["batch_size"])
    lr       = float(tcfg["lr"]);        wd       = float(tcfg["weight_decay"])
    clip     = float(tcfg.get("grad_clip", 0.0))
    patience = int(tcfg["early_stop_patience"])
    bidir    = bool(tcfg.get("bidirectional", True))
    print(f"recipe from molnet.yml: epochs {epochs} batch {batch} lr {lr} wd {wd} "
          f"clip {clip} patience {patience} bidirectional {bidir}", flush=True)
    cfg["k"] = K                                              # bond mode: k = neighbor slots
    cfg["bond_dim"] = 0
    cfg["chirality"] = False   # True -> SE(3), can distinguish enantiomers                  # explicit per-edge bond features
    npoint = int(cfg["max_atom_num"])
    nbin_ds = int(float(cfg["max_mz"]) / float(cfg["resolution"]))
    # adduct layout is read from the config, not passed in -- see spectrum_heads.adduct_order
    dskw = dict(need_prec=bidir, resolution=float(cfg["resolution"]), nbin=nbin_ds)
    if bidir:
        print(f"adduct layout (from config): {adduct_order(DATA_CFG)}", flush=True)
    tr = BondDS(a.train_data, **dskw); te = BondDS(a.test_data, **dskw)
    # Early stopping and checkpoint selection must NOT happen on the test set. With --val_data we
    # select on val and touch test exactly once, after training.
    # Early stopping and checkpoint selection must not touch the test set, or the reported number
    # is selection-biased. This has bitten us: the mf_nist runs were launched with
    # `--test_data mf_nist_val.pkl --val_data mf_nist_val.pkl`, so every figure from them was
    # selected on the set it was reported on. Guard the ACTUAL hazard -- val and test being the
    # same data -- not merely an empty --val_data.
    same = (not a.val_data) or os.path.realpath(a.val_data) == os.path.realpath(a.test_data)
    if same and not a.allow_biased_eval:
        raise SystemExit(
            f"--val_data and --test_data resolve to the same data ({a.val_data!r} vs "
            f"{a.test_data!r}). Model selection would happen on the test set, so the reported "
            f"number would be biased. Pass a separate validation set, or --allow_biased_eval "
            f"to do it deliberately.")
    if same:
        print("WARNING: selecting on TEST (biased protocol) — do not report this number", flush=True)
    va_ds = BondDS(a.val_data, **dskw) if a.val_data else te
    # auto-detect env width: the QTOF+Orbitrap merge appends an INSTRUMENT FLAG (6 -> 7 dims),
    # so add_num must follow the data rather than the static config.
    env_dim = int(np.asarray(tr.data[0]["env"]).shape[0])
    if env_dim != int(cfg["add_num"]):
        print(f"env dim {env_dim} != config add_num {cfg['add_num']} -> overriding", flush=True)
        cfg["add_num"] = env_dim
    print(f"dev {dev} | BOND MODE k={K} | train {len(tr)} test {len(te)}", flush=True)
    trl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=8, pin_memory=True,
                     persistent_workers=False, drop_last=True)
    tel = DataLoader(te, batch_size=batch, num_workers=4, pin_memory=True)
    val = DataLoader(va_ds, batch_size=batch, num_workers=4, pin_memory=True)
    model = MolNet_MS(cfg, out_relu=False).to(dev)
    if bidir:
        print("BIDIRECTIONAL head: forward/reverse/gate + precursor mask", flush=True)
    if a.pretrain:
        sd = torch.load(a.pretrain, map_location=dev, weights_only=False)["encoder_state_dict"]
        # chirality=True widens the FIRST layer (extra signed-volume channel), so those tensors
        # have different shapes than the (achiral) pretrained encoder -> drop just those.
        own = model.encoder.state_dict()
        skipped = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
        for k in skipped:
            sd.pop(k)
        miss, unexp = model.encoder.load_state_dict(sd, strict=False)
        if skipped:
            print(f"  shape-mismatched (skipped, chirality layer): {len(skipped)} tensors", flush=True)
        print(f"warm-started encoder from {a.pretrain} | missing {len(miss)} unexpected {len(unexp)}", flush=True)
    print(f"#params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    best = -1; bad = 0
    for ep in range(1, epochs + 1):
        trl_loss, tr_m = run(model, trl, dev, npoint, opt, nogeo=False, bidir=bidir, clip=clip,)
        _, va_m = run(model, val, dev, npoint, None, nogeo=False, bidir=bidir,)  # VAL, not test
        sched.step(va_m); flag = ""
        if va_m > best:
            best = va_m; bad = 0; flag = " *best*"
            torch.save({"model_state_dict": model.state_dict(), "val_cos": va_m, "seed": a.seed}, ckpt)
        else: bad += 1
        print(f"epoch {ep:3d} | train loss {trl_loss:.4f} cos {tr_m:.4f} | val cos {va_m:.4f}"
              f"  (kNN 0.489 / graph 0.513){flag}", flush=True)
        if bad >= patience: print(f"early stop (best {best:.4f})", flush=True); break
    # ---- final, single evaluation on TEST using the val-selected checkpoint ----
    sd = torch.load(ckpt, map_location=dev, weights_only=False)["model_state_dict"]
    model.load_state_dict(sd)
    _, test_m = run(model, tel, dev, npoint, None, nogeo=False, bidir=bidir,)
    print(f"\nDONE best VAL cos {best:.4f} | TEST cos {test_m:.4f} (single evaluation) -> {ckpt}",
          flush=True)


if __name__ == "__main__":
    main()
