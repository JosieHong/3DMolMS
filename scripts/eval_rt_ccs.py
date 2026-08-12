"""R^2 / RMSE / MAE / median relative error for the RT and CCS checkpoints, on val and test.

R^2 = 1 - SS_res/SS_tot, computed in the ORIGINAL units against the mean of the evaluated split.
Note this makes R^2 sensitive to the spread of the split being scored: CCS val and test have only
~445 rows each, so a couple of outliers move it noticeably.
"""
import os

# Data and checkpoints are resolved relative to the CURRENT WORKING DIRECTORY, so
# these scripts work from any clone. Config files come from the installed
# molnetpack via config_path(), never from a path next to this file.
import argparse
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from train_rt_ccs import OthDS, build_scalar_model
from molnetpack import config_path
from molnetpack.utils import make_idx_base

CKPTS = [("CCS full fine-tune", "ccs", "check_point/release_ccs.pt", None),
         ("CCS frozen encoder", "ccs", "check_point/ccs_frozen.pt", None),
         ("RT  full head", "rt", "check_point/release_rt.pt", None),
         ("RT  narrow head", "rt", "check_point/rt_narrow.pt", [512, 256])]


def evaluate(task, ckpt, dev, layers=None):
    d = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = dict(yaml.safe_load(open(config_path("molnet.yml")))["model"])
    if layers:
        cfg["decode_layers"] = layers
    out = {}
    for split in ("val", "test"):
        ds = OthDS(f"data/{task}_bond_{split}.pkl", task)
        env_dim = int(np.asarray(ds.data[0]["env"]).shape[0])
        m = build_scalar_model(cfg, env_dim).to(dev)
        m.load_state_dict(d["model_state_dict"]); m.eval()
        mu, sd = d["mu"], d["sd"]
        P, Y = [], []
        with torch.no_grad():
            for mol, mask, nidx, nmask, env, y in DataLoader(ds, batch_size=64, num_workers=4):
                x = mol.to(dev).permute(0, 2, 1)
                ib = make_idx_base(x.size(0), int(cfg["max_atom_num"]), dev)
                p = m(x, mask.to(dev), env.to(dev), ib, nidx.to(dev), nmask.to(dev)) * sd + mu
                P.append(p.cpu().numpy()); Y.append(y.numpy())
        p = np.concatenate(P); y = np.concatenate(Y)
        r2 = 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[split] = (r2, float(np.sqrt(((y - p) ** 2).mean())), float(np.abs(y - p).mean()),
                      float(np.median(np.abs(y - p) / np.maximum(y, 1e-6)) * 100), len(y))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=6)
    a = ap.parse_args(); dev = torch.device(f"cuda:{a.gpu}")
    print(f"{'model':<22}{'split':<7}{'n':>6}{'R^2':>9}{'RMSE':>10}{'MAE':>9}{'medRel%':>10}")
    for name, task, ck, layers in CKPTS:
        if not os.path.exists(ck):
            print(f"{name:<22}(no checkpoint yet)"); continue
        try:
            res = evaluate(task, ck, dev, layers)
        except Exception as e:
            print(f"{name:<22}ERROR: {type(e).__name__}: {str(e)[:60]}"); continue
        for split, (r2, rmse, mae, rel, n) in res.items():
            print(f"{name:<22}{split:<7}{n:>6}{r2:>9.4f}{rmse:>10.3f}{mae:>9.3f}{rel:>10.2f}")
