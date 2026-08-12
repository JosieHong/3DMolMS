"""Rebuild the RT and CCS datasets in the SAME format as the MS/MS release data.

WHY A REBUILD. The shipped `allccs_etkdgv3_*` / `metlin_etkdgv3_*` pickles predate the bond-graph
work and carry only {mol, env, target}:
  * no `neighbor_idx` / `neighbor_mask`, so the release encoder (MolConv-BOND) cannot run on them --
    it would silently fall back to kNN neighbourhoods, which measured worse (0.489 vs 0.513). A
    release trained on them would be a DIFFERENT architecture from the MS/MS model.
  * the CCS pickle has no SMILES at all (only `AllCCS...` ids), so the graph cannot be regenerated
    from it.
  * the RT pickle stores SMILES under the key `smiels` -- a typo -- so any code reading
    `e["smiles"]` raises KeyError, hence the raw SMRT file is the source here.
    (Correction, 2026-08-10: an earlier version of this note claimed the raw file also avoids the
    old pickle's atom-map annotations. It does not -- 15,084 of 79,951 built rows carry `[c:0]`
    style maps. Harmless: maps change neither the graph, the ETKDG geometry, nor the InChIKey, so
    a mapped and an unmapped copy of one structure land in the SAME skeleton split, no leak.)

Both rebuilds go back to source:
    CCS  data/origin/allccs_download.csv        (AllCCS ID, Name, Structure, Formula, Type, Adduct, m/z, CCS, ...)
    RT   data/origin/SMRT_dataset.csv                  (id, smiles, rt)  --rt_csv

OUTPUT matches the MS/MS release format: {title, smiles, mol, neighbor_idx, neighbor_mask, env,
<target>} with a non-stereochemical InChIKey 80/10/10 split, so nothing leaks across partitions.

CCS keeps only **Experimental** measurements -- AllCCS also ships predicted values, and training a
predictor on another model's predictions would launder its errors into ours.

Usage: python scripts/build_rt_ccs_dataset.py --task ccs --workers 40
"""
import os

# Data and checkpoints are resolved relative to the CURRENT WORKING DIRECTORY, so
# these scripts work from any clone. Config files come from the installed
# molnetpack via config_path(), never from a path next to this file.
import csv
import pickle
import random
import argparse
import collections
import numpy as np
from multiprocessing import Pool
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

from build_msms_dataset import featurize, skel
from molnetpack.data_utils.utils import normalize_adduct

CCS_CSV = "data/origin/allccs_download.csv"
# METLIN SMRT (id, smiles, rt). Download it and pass --rt_csv, or drop it at this default path.
RT_CSV = os.path.join("data", "origin", "SMRT_dataset.csv")
# CCS env = adduct one-hot only (no collision energy). Spellings are taken from the FILE, not
# assumed: AllCCS writes "[M-H2O+H]+", not the "[M+H-H2O]+" used elsewhere in this repo, and a
# mismatched string silently drops every row of that adduct. These six cover 5,398 of the 6,275
# experimental measurements (86%).
CCS_ADDUCTS = ["[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+HCOO]-", "[M+H-H2O]+"]


def _w(s):
    return s, featurize(s)


def load_ccs(CCS_CSV=None):
    CCS_CSV = CCS_CSV or globals()["CCS_CSV"]
    rows, stats = [], collections.Counter()
    with open(CCS_CSV) as fh:
        for r in csv.DictReader(fh):
            stats["read"] += 1
            # the CSV is several downloads concatenated: 16,599 repeated header rows and one
            # "please sign in" auth-failure artifact
            if r.get("AllCCS ID", "").strip() in ("AllCCS ID", ""):
                stats["drop:repeated header / artifact"] += 1; continue
            if r.get("Type", "").strip() != "Experimental CCS":
                stats["drop:not experimental"] += 1; continue
            add = normalize_adduct(r.get("Adduct", ""))
            if add not in CCS_ADDUCTS:
                stats["drop:adduct"] += 1; continue
            try:
                ccs = float(r["CCS"])
            except (ValueError, KeyError, TypeError):
                stats["drop:bad CCS"] += 1; continue
            if not (50.0 < ccs < 400.0):          # physical range; guards parse errors
                stats["drop:CCS out of range"] += 1; continue
            smi = (r.get("Structure") or "").strip()
            if not smi:
                stats["drop:no structure"] += 1; continue
            rows.append({"title": r.get("AllCCS ID", ""), "smiles": smi,
                         "adduct": add, "y": ccs})
    return rows, stats, "ccs"


def load_rt(RT_CSV=None):
    RT_CSV = RT_CSV or globals()["RT_CSV"]
    rows, stats = [], collections.Counter()
    with open(RT_CSV) as fh:
        for r in csv.DictReader(fh):
            stats["read"] += 1
            smi = (r.get("smiles") or "").strip()
            try:
                rt = float(r["rt"])
            except (ValueError, KeyError, TypeError):
                stats["drop:bad RT"] += 1; continue
            if not smi:
                stats["drop:no smiles"] += 1; continue
            if rt <= 0:
                stats["drop:non-positive RT"] += 1; continue
            rows.append({"title": f"smrt_{r.get('id','')}", "smiles": smi, "adduct": None, "y": rt})
    return rows, stats, "rt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rt", "ccs"], required=True)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--ccs_csv", default=CCS_CSV, help="AllCCS download CSV")
    ap.add_argument("--rt_csv", default=RT_CSV, help="METLIN SMRT CSV")
    a = ap.parse_args()

    rows, stats, key = (load_ccs(a.ccs_csv) if a.task == "ccs" else load_rt(a.rt_csv))
    print(f"{a.task}: {len(rows):,} usable rows", flush=True)
    for k, v in stats.most_common(6):
        print(f"    {k:<28}{v:>10,}", flush=True)

    smis = sorted({r["smiles"] for r in rows})
    print(f"\nunique SMILES {len(smis):,} -> skeletons", flush=True)
    with Pool(a.workers) as p:
        sk = dict(zip(smis, p.map(skel, smis, chunksize=256)))
    rows = [r for r in rows if sk.get(r["smiles"])]
    for r in rows:
        r["skel"] = sk[r["smiles"]]

    print(f"featurising {len(smis):,} molecules (ETKDG + bond graph)", flush=True)
    feats = {}
    with Pool(a.workers) as p:
        for i, (s, f) in enumerate(p.imap_unordered(_w, smis, chunksize=32)):
            if f is not None:
                feats[s] = f
            if (i + 1) % 5000 == 0:
                print(f"    {i+1}/{len(smis)}", flush=True)
    rows = [r for r in rows if r["smiles"] in feats]
    print(f"featurised {len(feats):,}/{len(smis):,}; rows remaining {len(rows):,}", flush=True)

    keys = sorted({r["skel"] for r in rows})
    rng = random.Random(42); rng.shuffle(keys)
    n = len(keys); nv = int(0.10 * n); nt = int(0.10 * n)
    val_k, test_k = set(keys[:nv]), set(keys[nv:nv + nt])
    train_k = set(keys[nv + nt:])
    print(f"\nskeletons: train {len(train_k):,} / val {len(val_k):,} / test {len(test_k):,}", flush=True)

    for name, ks in (("train", train_k), ("val", val_k), ("test", test_k)):
        out = []
        for r in rows:
            if r["skel"] not in ks:
                continue
            arr, idx, msk, _ = feats[r["smiles"]]
            if a.task == "ccs":
                env = np.zeros(len(CCS_ADDUCTS), dtype=np.float32)
                env[CCS_ADDUCTS.index(r["adduct"])] = 1.0
            else:
                env = np.zeros(1, dtype=np.float32)      # RT has no experimental covariates here
            out.append({"title": r["title"], "smiles": r["smiles"], "mol": arr,
                        "neighbor_idx": idx, "neighbor_mask": msk, "env": env,
                        key: np.asarray([r["y"]], dtype=np.float32)})
        path = f"data/{a.task}_bond_{name}.pkl"
        with open(path + ".tmp", "wb") as fh:
            pickle.dump(out, fh)
        os.replace(path + ".tmp", path)
        y = np.array([float(e[key][0]) for e in out])
        print(f"  {path}: {len(out):,} rows | {len({e['smiles'] for e in out}):,} molecules | "
              f"{key} p5 {np.percentile(y,5):.1f} p50 {np.percentile(y,50):.1f} "
              f"p95 {np.percentile(y,95):.1f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
