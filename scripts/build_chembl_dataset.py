"""Build bonded-neighbor indices for ChEMBL SSL data -> pretraining MolConv-bond.

Same as build_molconv_bond_data.py but for the ChEMBL SSL set (no spec/env; carries mol+mask).
K=6 to MATCH the MolConv-bond fine-tuning encoder, so pretrained weights transfer. Bond-length
sanity is checked on a sample (alignment already verified on qtof; same conformation pipeline).

Usage: python scripts/build_chembl_dataset.py
"""
import argparse
import os
import pickle
import random
import numpy as np
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
random.seed(0)

MAXN = 300
K = 6


def bond_neighbors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    n = mol.GetNumAtoms()
    if n > MAXN:
        return None
    idx = np.tile(np.arange(n, dtype=np.int64)[:, None], (1, K))
    msk = np.zeros((n, K), dtype=bool)
    trunc = 0
    for a in mol.GetAtoms():
        i = a.GetIdx()
        nb = [x.GetIdx() for x in a.GetNeighbors()]
        if len(nb) > K:
            trunc += len(nb) - K; nb = nb[:K]
        for s, j in enumerate(nb):
            idx[i, s] = j; msk[i, s] = True
    return idx, msk, n, trunc


def convert(inp, outp, sanity_sample=400):
    data = pickle.load(open(inp, "rb"))
    out, skipped, tot_trunc, maxdeg = [], 0, 0, 0
    sample_ids = set(random.sample(range(len(data)), min(sanity_sample, len(data))))
    blens = []
    for ci, d in enumerate(data):
        r = bond_neighbors(d["smiles"])
        if r is None:
            skipped += 1; continue
        nidx, nmsk, n, trunc = r
        tot_trunc += trunc; maxdeg = max(maxdeg, int(nmsk.sum(1).max()))
        mol = np.asarray(d["mol"], dtype=np.float32)
        full_idx = np.tile(np.arange(MAXN, dtype=np.int64)[:, None], (1, K))
        full_msk = np.zeros((MAXN, K), dtype=bool)
        full_idx[:n] = nidx; full_msk[:n] = nmsk
        if ci in sample_ids:                                   # sampled bond-length sanity
            pos = mol[:n, 0:3]
            if not np.all(pos[-1] == 0):
                for i in range(n):
                    for s in range(K):
                        if nmsk[i, s]:
                            blens.append(float(np.linalg.norm(pos[i] - pos[nidx[i, s]])))
        out.append({"title": d.get("title"), "smiles": d["smiles"], "mol": mol,
                    "neighbor_idx": full_idx, "neighbor_mask": full_msk,
                    "mask": np.asarray(d["mask"]) if "mask" in d else full_msk.any(1)})
    pickle.dump(out, open(outp, "wb"))
    b = np.array(blens)
    print(f"{outp}: {len(out)} kept, {skipped} skipped | max degree {maxdeg} | truncated bonds {tot_trunc}",
          flush=True)
    print(f"   bond-len sanity (n={len(b)}): mean {b.mean():.3f} p1 {np.percentile(b,1):.3f} "
          f"p99 {np.percentile(b,99):.3f} frac[0.9,1.9] {((b>0.9)&(b<1.9)).mean():.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Add the covalent bond graph to the ChEMBL SSL pickles, for geo pretraining.")
    ap.add_argument("--train_in", default="data/chembl_ssl_train.pkl")
    ap.add_argument("--valid_in", default="data/chembl_ssl_valid.pkl")
    ap.add_argument("--train_out", default="data/chembl_bond_train.pkl")
    ap.add_argument("--valid_out", default="data/chembl_bond_valid.pkl")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing outputs (they are multi-GB, so refuse by default)")
    a = ap.parse_args()

    for out in (a.valid_out, a.train_out):
        if os.path.exists(out) and not a.force:
            raise SystemExit(
                f"{out} already exists. Pass --force to rebuild. (Guarded because this script had "
                f"no CLI and would silently rewrite a multi-GB file on any invocation.)")

    convert(a.valid_in, a.valid_out)
    convert(a.train_in, a.train_out)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
