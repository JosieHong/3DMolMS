"""Geometric SSL for MolConv-bond (coordinate denoising, E(3)-invariant-safe).

Add Gaussian noise to atom coordinates, then from the encoder's (rotation-INVARIANT) per-atom
features reconstruct the CLEAN local geometry of each atom's bonds: the per-atom neighbor Gram
G_i[a,b] = <d_ia, d_ib> (d = clean relative displacement), whose diagonal is squared bond length
and off-diagonal encodes bond angle. Both are O(3)-invariant scalars, so the target is compatible
with the invariant encoder (unlike predicting the noise vector, which needs an equivariant net).

This teaches the encoder to denoise / represent true bond lengths + angles — the geometry
MolConv-bond newly exposes. Pretrained encoder transfers to MS/MS via molconv_bond_train.py --pretrain.

Usage: python scripts/pretrain_geo.py --gpu 5 --epochs 50 --batch 128 --sigma 0.2
"""

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
from molnetpack.model import Encoder
from molnetpack.utils import make_idx_base

K = 6
CFG = yaml.safe_load(open(config_path("molnet.yml")))["model"]
NP = int(CFG["max_atom_num"]); EMB = int(CFG["emb_dim"])


class GeoBondDS(Dataset):
    def __init__(self, path):
        self.data = pickle.load(open(path, "rb"))
        for d in self.data:
            d["mol"] = np.asarray(d["mol"], dtype=np.float32)
            d["valid"] = ~np.all(d["mol"] == 0, axis=1)

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        return (torch.from_numpy(d["mol"]), torch.from_numpy(d["valid"]),
                torch.from_numpy(d["neighbor_idx"]), torch.from_numpy(d["neighbor_mask"]))


def clean_gram(pos, nidx, nmask):
    """Per-atom neighbor Gram of clean relative displacements. pos [B,N,3], nidx/nmask [B,N,K].
    Returns gram [B,N,K,K] and a pair-mask [B,N,K,K]."""
    B, N, _ = pos.shape
    nb = torch.gather(pos.unsqueeze(2).expand(B, N, K, 3), 1,
                      nidx.unsqueeze(-1).expand(B, N, K, 3))     # neighbor positions [B,N,K,3]
    d = nb - pos.unsqueeze(2)                                     # displacements [B,N,K,3]
    gram = torch.matmul(d, d.transpose(-1, -2))                  # [B,N,K,K]
    pair = nmask.unsqueeze(-1) & nmask.unsqueeze(-2)             # [B,N,K,K]
    return gram, pair


class GeoBondSSL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(in_dim=int(cfg["in_dim"]), layers=cfg["encode_layers"], emb_dim=EMB,
                               point_num=NP, k=K, chirality=False)
        self.head = nn.Sequential(nn.Linear(EMB, EMB // 2), nn.SiLU(), nn.Linear(EMB // 2, K * K))

    def forward(self, x, mask, idx_base, nidx, nmask):
        per_atom, _ = self.encoder(x, idx_base, mask, return_per_atom=True,
                                   neighbor_idx=nidx, neighbor_mask=nmask)   # [B,EMB,N]
        h = per_atom.permute(0, 2, 1)                                        # [B,N,EMB]
        return self.head(h).view(h.size(0), h.size(1), K, K)                 # [B,N,K,K]


def run(model, loader, dev, sigma, opt=None):
    train = opt is not None; model.train(train)
    tot = n = 0
    for mol, valid, nidx, nmask in loader:
        mol = mol.to(dev); valid = valid.to(dev); nidx = nidx.to(dev); nmask = nmask.to(dev)
        bs = mol.size(0)
        clean_pos = mol[:, :, :3].clone()
        target, pair = clean_gram(clean_pos, nidx, nmask)                    # clean geometry target
        noisy = mol.clone()
        noise = torch.randn_like(clean_pos) * sigma * valid.unsqueeze(-1)    # noise real atoms only
        noisy[:, :, :3] = clean_pos + noise
        x = noisy.permute(0, 2, 1)                                           # [B,21,N]
        ib = make_idx_base(bs, NP, dev)
        with torch.set_grad_enabled(train):
            pred = model(x, valid, ib, nidx, nmask)                          # [B,N,K,K]
            diff = (pred - target) ** 2
            loss = (diff * pair).sum() / pair.sum().clamp(min=1)             # masked MSE
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * bs; n += bs
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5); ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=0.2); ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--ckpt", default="check_point/molnet_pre_geobond.pt")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev = torch.device(f"cuda:{a.gpu}")
    tr = GeoBondDS("data/chembl_bond_train.pkl"); te = GeoBondDS("data/chembl_bond_valid.pkl")
    print(f"dev {dev} | GEO-SSL (bond) sigma {a.sigma} | train {len(tr)} valid {len(te)}", flush=True)
    trl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=16, pin_memory=True,
                     persistent_workers=True, drop_last=True)
    tel = DataLoader(te, batch_size=a.batch, num_workers=8, pin_memory=True)
    model = GeoBondSSL(CFG).to(dev)
    print(f"#params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    best = 1e9; bad = 0
    for ep in range(1, a.epochs + 1):
        trl_loss = run(model, trl, dev, a.sigma, opt)
        va = run(model, tel, dev, a.sigma, None)
        sched.step(va); flag = ""
        if va < best:
            best = va; bad = 0; flag = " *best*"
            torch.save({"encoder_state_dict": model.encoder.state_dict(), "val_loss": va,
                        "k": K, "sigma": a.sigma}, a.ckpt)
        else: bad += 1
        print(f"epoch {ep:3d} | train {trl_loss:.4f} | valid {va:.4f}{flag}", flush=True)
        if bad >= a.patience: print(f"early stop (best {best:.4f})", flush=True); break
    print(f"\nDONE GEO-SSL best valid {best:.4f} -> {a.ckpt}", flush=True)


if __name__ == "__main__":
    main()
