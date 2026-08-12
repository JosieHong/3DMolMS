"""Build the Orbitrap-HCD and QTOF/QQQ datasets, each with a proper 3-way split.

TWO SEPARATE DATASETS, selected with --group:
    orbi : Orbitrap / FT, HCD only  (records explicitly labelled CID are dropped)
    qtof : QTOF + QQQ, BOTH CID and HCD (no fragmentation filter at all)

Fragmentation is deliberately NOT written into `env`. Because the two groups are built as separate
datasets, each is internally consistent in its collision-energy convention (Orbitrap reports NCE%,
QTOF reports eV), so there is no unit ambiguity to annotate around. If the two are ever MERGED for
joint training, the collision-energy channel WILL mix eV and NCE% and must be harmonised at that
point -- see molnetpack/data_utils/utils.py: ce2nce / nce2ce.

DESIGN DECISIONS (all from the 2026-08-08 data review; see DATASET_NOTES.md)

1. GROUP BY ANALYSER, pooling both metadata keys and matching substrings, because each source
   populates them differently:
       NIST    SOURCE_INSTRUMENT = model,  INSTRUMENT_TYPE = HCD / IT-FT / Q-TOF  (mixed meaning)
       GNPS    SOURCE_INSTRUMENT = analyser, INSTRUMENT_TYPE = hcd / cid          (fragmentation)
       MoNA    SOURCE_INSTRUMENT = model,  INSTRUMENT_TYPE = LC-ESI-QFT etc       (analyser)

2. FRAGMENTATION HANDLING DIFFERS BY GROUP, because the label evidence differs:
     orbi -> HCD only. Drop explicit CID; treat unlabelled as HCD. Supported: labelled Orbitrap/FT
             records are 1,470,461 HCD vs 8,025 CID = 99.5% HCD.
     qtof -> keep everything. Only ~9% of the group is labelled at all, and among genuine TOF
             records the labels are 98.9% CID, so "HCD only" would delete the group. Filtering
             here would select on label AVAILABILITY rather than on physics.

3. ION TRAP EXCLUDED. Resonant-excitation CID imposes a ~1/3 low-mass cutoff, so its spectra are
   genuinely different in which peaks EXIST -- the one fragmentation difference that 0.2 Da binning
   cannot wash out.

4. NOISE CONTROLS
     - peaks above precursor + tolerance are physically impossible -> removed
     - intensity floor: peaks below MIN_REL_INT of the base peak are noise -> removed
     - the existing filters are kept: allowed atom types, 10-300 atoms, m/z 50-1500, >= MIN_PEAKS

5. REDUNDANCY CAP. Orbitrap averages ~38 spectra per compound (long collision-energy ladders).
   Capping at MAX_CE_PER_KEY spectra per (skeleton, adduct) cuts volume hard while losing ZERO
   compounds -- and compounds, not spectra, are what the learning curve responds to
   (+0.064 val per doubling of structural diversity).

6. SPLIT: non-stereochemical InChIKey first block (the skeleton), 80/10/10, assigned ONCE per
   skeleton so no compound spans partitions. CASMI query skeletons are forced OUT of train so the
   identification benchmark is honest (currently 75.6% of CASMI 2016 leaks into training, which
   inflated Top-1 by ~8x and left only 36 of 124 queries usable).

Usage: python scripts/build_msms_dataset.py --group orbi --workers 64
"""
import os

# Data and checkpoints are resolved relative to the CURRENT WORKING DIRECTORY, so
# these scripts work from any clone. Config files come from the installed
# molnetpack via config_path(), never from a path next to this file.
import csv
import glob
import pickle
import argparse
import collections
import random
import numpy as np
from multiprocessing import Pool
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

from rdkit.Chem import AllChem
import yaml

from molnetpack import config_path
from molnetpack.data_utils.utils import parse_collision_energy, precursor_calculator

# --- molecule featurisation (was experiments/build_massformer_style.py, folded in here) ---------
ATOM_TYPE = yaml.safe_load(open(config_path("encoding_etkdgv3.yml")))["encoding"]["atom_type"]
_MCFG = yaml.safe_load(open(config_path("molnet.yml")))["model"]
MAXN, K = 300, 6
RES = float(_MCFG["resolution"]); MAXMZ = float(_MCFG["max_mz"]); NBIN = int(MAXMZ / RES)


def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    n = mol.GetNumAtoms()
    if n > MAXN or n < 2: return None
    ps = AllChem.ETKDGv3(); ps.randomSeed = 0xF00D; ps.maxIterations = 1000
    if AllChem.EmbedMolecule(mol, ps) == -1: return None
    try: conf = mol.GetConformer()
    except Exception: return None
    xyz = conf.GetPositions(); xyz = xyz - xyz.mean(0)
    rows = []
    for i, a in enumerate(mol.GetAtoms()):
        s = a.GetSymbol()
        if s not in ATOM_TYPE: return None
        rows.append(list(xyz[i]) + [a.GetDegree(), a.GetExplicitValence(), a.GetMass()/100,
                                    a.GetFormalCharge(), a.GetNumImplicitHs(),
                                    int(a.GetIsAromatic()), int(a.IsInRing())] + list(ATOM_TYPE[s]))
    arr = np.pad(np.asarray(rows, dtype=np.float32), ((0, MAXN - n), (0, 0)))
    idx = np.tile(np.arange(MAXN, dtype=np.int64)[:, None], (1, K))
    msk = np.zeros((MAXN, K), dtype=bool)
    for a in mol.GetAtoms():
        i = a.GetIdx()
        for s_, nb in enumerate([x.GetIdx() for x in a.GetNeighbors()][:K]):
            idx[i, s_] = nb; msk[i, s_] = True
    try: ik = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)).split("-")[0]
    except Exception: ik = None
    return arr, idx, msk, ik

from rdkit.Chem.Descriptors import ExactMolWt

# Directory holding the raw `original_*.mgf` source files (NIST, MoNA, GNPS, Agilent, Waters).
# These are licensed datasets we cannot redistribute, so point --raw_dir at your own copy.
D = os.path.join("data", "origin", "mgf")  # override with --raw_dir
# The *_qtof / *_orbitrap files OVERLAP -- GNPS and NIST20 are byte-identical pairs, NIST23 is not.
# So read every source and de-duplicate on TITLE, then classify by analyser. Trusting the filenames
# would double-count.
def list_sources(raw_dir):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(raw_dir, "original_*.mgf")))
# CASMI 2016 challenge solutions. Their skeletons are forced out of train and val so the
# identification benchmark stays honest. Optional: omit and no holdout is applied.
CASMI = []

ADDUCTS = ["[M+H]+", "[M-H]-", "[M+H-H2O]+", "[M+Na]+", "[M+2H]2+"]
AIDX = {a: i for i, a in enumerate(ADDUCTS)}
ADDUCT_CHARGE = {"[M+H]+": 1, "[M-H]-": 1, "[M+H-H2O]+": 1, "[M+Na]+": 1, "[M+2H]2+": 2}
MIN_PEAKS = 5
PPM_TOL = 10.0               # reported vs theoretical precursor m/z
MIN_MZ, MAX_MZ = 50.0, 1500.0
MIN_REL_INT = 0.001          # 0.1% of base peak
PREC_TOL_DA = 2.0            # peaks above precursor + this are impossible
MAX_CE_PER_KEY = 10          # default; override with --max_ce (0 = keep everything)


def prec_mz_ok(smiles, adduct, reported, tol_ppm=PPM_TOL):
    """Reject records whose reported precursor m/z disagrees with the mass implied by SMILES.

    NIST23 contains records sharing a SMILES and adduct but reporting precursor m/z of 244.97,
    381.17 and 453.21 -- a given [M+H]+ has exactly ONE m/z, so the structure annotation is wrong
    on some of them. Training on those teaches the model a structure->spectrum mapping that is
    simply false. The shipped filter.py already does this check via `ppm_tolerance`; the standalone
    builder was missing it.
    Returns (ok, theoretical_mz).
    """
    if not reported or reported <= 0:
        return False, 0.0
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False, 0.0
    try:
        theo = precursor_calculator(adduct, ExactMolWt(m))
    except Exception:
        return False, 0.0
    if theo <= 0:
        return False, 0.0
    return abs(reported - theo) / theo * 1e6 <= tol_ppm, theo


def group_of(si, it):
    s = f"{si} {it}".lower()
    if "ion trap" in s and "ft" not in s:
        return "iontrap"
    if any(k in s for k in ("orbitrap", "ftms", "qft", "it-ft", "fourier")):
        return "orbi"
    if "tof" in s or "quadrupole" in s:
        return "qtof"
    return "other"


def frag_of(si, it):
    """-> 'HCD' | 'CID' | 'UNK'.

    'IT-FT/ion trap with FTMS' is NIST's spelling of ION-TRAP CID and spells neither "hcd" nor
    "cid". A plain substring test returned UNK for it, and the Orbitrap group's
    unlabelled-means-HCD rule then imputed 39,226 resonant-excitation spectra into an HCD-only
    dataset. Measured damage: Lumos replicate self-similarity 0.5697 with them, 0.9935 without.
    Ion-trap CID is exactly the mode that must not be pooled -- resonant excitation imposes a
    ~1/3 low-mass cutoff, so whole fragments are ABSENT rather than merely shifted, which no
    amount of coarse binning reconciles.
    """
    s = f"{si} {it}".lower()
    # check ion-trap CID FIRST: 'IT-FT/ion trap with FTMS' would otherwise fall through to UNK
    if "it-ft" in s or "ion trap" in s or "resonan" in s:
        return "CID"
    if "hcd" in s:
        return "HCD"
    if "cid" in s:
        return "CID"
    return "UNK"


def parse_mgf(path):
    """Stream one MGF, yielding dicts. Peaks kept as (mz, intensity) lists."""
    cur, mz, it = None, [], []
    for line in open(path, errors="ignore"):
        line = line.rstrip("\n")
        if line == "BEGIN IONS":
            cur, mz, it = {}, [], []
        elif line == "END IONS":
            if cur is not None:
                cur["_mz"], cur["_it"] = mz, it
                yield cur
            cur = None
        elif cur is not None:
            if "=" in line:
                k, v = line.split("=", 1); cur[k.upper()] = v
            elif line and (line[0].isdigit() or line[0] == "."):
                p = line.split()
                if len(p) >= 2:
                    try:
                        mz.append(float(p[0])); it.append(float(p[1]))
                    except ValueError:
                        pass


def clean_peaks(mz, inten, prec_mz):
    """Remove impossible and noise peaks. Returns (mz, inten) or None if too few survive."""
    a = np.asarray(mz, float); b = np.asarray(inten, float)
    keep = (a >= MIN_MZ) & (a <= MAX_MZ)
    if prec_mz and prec_mz > 0:
        keep &= a <= (prec_mz + PREC_TOL_DA)     # nothing above the parent
    a, b = a[keep], b[keep]
    if b.size == 0 or b.max() <= 0:
        return None
    keep = b >= MIN_REL_INT * b.max()            # noise floor
    a, b = a[keep], b[keep]
    return (a, b) if a.size >= MIN_PEAKS else None


def bin_spec(mz, inten):
    v = np.zeros(NBIN, dtype=np.float32)
    for m, i in zip(mz, inten):
        k = int(round(m / RES))
        if 0 <= k < NBIN:
            v[k] += i
    if v.max() > 0:
        v /= v.max()
    return v


def ce_to_nce(ce_str, prec_mz, charge):
    """-> NCE as a FRACTION (0.35 == 35%), or 0.0 if unparseable. Never returns eV.

    molnetpack.parse_collision_energy returns NCE on the PERCENT scale for every input format, so
    the conversion here is an unconditional /100. It used to apply a magnitude heuristic
    ("divide by 100 only if > 2") to paper over a library bug where NCE-parsed strings came back
    as fractions while eV-derived ones came back as percentages. With that bug fixed the heuristic
    became harmful in its own right: a genuine 'NCE=1.5%' is <= 2, so it was passed through as
    1.5 instead of 0.015 -- a 100x error on exactly the low-collision-energy spectra the guard was
    supposed to protect.
    """
    if not prec_mz or prec_mz <= 0:
        return 0.0
    try:
        _, nce = parse_collision_energy(str(ce_str).strip(), float(prec_mz), int(charge))
    except Exception:
        return 0.0
    if nce is None:
        return 0.0
    nce = float(nce) / 100.0
    return nce if 0.0 <= nce <= 3.0 else 0.0        # discard absurd values


def skel(s):
    m = Chem.MolFromSmiles(s or "")
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m).split("-")[0]
    except Exception:
        return None


def _w(s):
    return s, featurize(s)


def report_stats(out_prefix, label):
    """Statistics + integrity checks for a built dataset. Also run standalone via --stats_only."""
    print(f"\n{'='*74}\n{label}   ({os.path.basename(out_prefix)})\n{'='*74}", flush=True)
    mols_by_split, tot = {}, 0
    for split in ("train", "val", "test"):
        path = f"{out_prefix}_{split}.pkl"
        if not os.path.exists(path):
            print(f"  {split}: MISSING"); continue
        d = pickle.load(open(path, "rb"))
        E = np.stack([e["env"] for e in d]); nce = E[:, 0]
        npk = np.array([int((e["spec"] > 0).sum()) for e in d])
        per_mol = collections.Counter(e["smiles"] for e in d)
        mols = set(per_mol); mols_by_split[split] = mols; tot += len(d)
        src = collections.Counter()
        for e in d:
            t = e["title"].lower()
            src["nist" if "nist" in t else "mona" if "mona" in t else "gnps" if "gnps" in t
                else "agilent" if "agilent" in t else "other"] += 1
        print(f"\n  --- {split} ---")
        print(f"  spectra {len(d):>8,}   molecules {len(mols):>7,}   "
              f"spectra/molecule mean {len(d)/max(len(mols),1):>5.1f} max {max(per_mol.values()):>3}")
        print(f"  NCE (fraction): p5 {np.percentile(nce,5):.3f}  p50 {np.percentile(nce,50):.3f}  "
              f"p95 {np.percentile(nce,95):.3f}  |  unparseable {100*(nce<=0).mean():.2f}%")
        print(f"  peaks/spectrum: p5 {np.percentile(npk,5):.0f}  p50 {np.percentile(npk,50):.0f}  "
              f"p95 {np.percentile(npk,95):.0f}")
        print("  adducts: " + "  ".join(f"{a}={int(E[:,1+i].sum()):,}"
                                         for i, a in enumerate(ADDUCTS) if E[:, 1+i].sum() > 0))
        print("  sources: " + "  ".join(f"{k}={v:,}" for k, v in src.most_common()))
    tr, va, te = (mols_by_split.get(s, set()) for s in ("train", "val", "test"))
    print(f"\n  TOTAL {tot:,} spectra")
    print(f"  molecule overlap  train&val {len(tr&va)}  train&test {len(tr&te)}  "
          f"val&test {len(va&te)}   (all must be 0)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--raw_dir", default=D,
                    help="directory of raw original_*.mgf files")
    ap.add_argument("--casmi", nargs="*", default=CASMI,
                    help="CASMI challenge-solution CSVs; their skeletons are held out of train/val")
    ap.add_argument("--group", choices=["orbi", "qtof"], required=True)
    # Redundancy cap per (skeleton, adduct). 0 disables it entirely. Capping cuts volume without
    # losing compounds, but keeping every collision energy gives denser CE coverage per molecule,
    # which matters because CE is a model INPUT -- the model must interpolate across it.
    ap.add_argument("--max_ce", type=int, default=MAX_CE_PER_KEY)
    ap.add_argument("--cap_scope", choices=["compound", "compound_adduct"], default="compound",
                    help="cap per compound (per instrument type), or per compound+adduct")
    ap.add_argument("--out", default="")
    ap.add_argument("--stats_only", action="store_true",
                    help="skip the build; just report statistics for an existing --out prefix")
    a = ap.parse_args()
    if not a.out:
        a.out = "data/" + ("orbi_hcd" if a.group == "orbi" else "qtof_all")

    if a.stats_only:
        report_stats(a.out, "Orbitrap (HCD)" if a.group == "orbi" else "QTOF/QQQ (CID+HCD)")
        return

    # ---------- 1. stream, filter, group ----------
    stats = collections.Counter()
    recs = []
    seen_titles = set()
    for f in list_sources(a.raw_dir):
        n0 = len(recs)
        for r in parse_mgf(os.path.join(a.raw_dir, f)):
            stats["read"] += 1
            t = r.get("TITLE", "")
            if t and t in seen_titles:
                stats["drop:duplicate title across source files"] += 1; continue
            if t:
                seen_titles.add(t)
            si = r.get("SOURCE_INSTRUMENT", ""); it = r.get("INSTRUMENT_TYPE", "")
            g = group_of(si, it)
            if g != a.group:
                stats[f"drop:group {g}"] += 1; continue
            # MS2 ONLY. NIST23's Orbitrap Fusion Lumos records include MS3 (3,253) and MS4
            # (1,662) -- fragmentation OF FRAGMENTS, which is a different experiment entirely.
            # Their presence is measurable: Lumos "replicates" of the same (compound, adduct, CE)
            # agreed at only 0.55 cosine, versus 0.96 for Elite (which is pure MS2), and WORSE
            # than the 0.83 cross-instrument Elite-vs-Lumos agreement.
            lvl = str(r.get("MS_LEVEL", "MS2")).strip().upper().replace("MS", "")
            if lvl not in ("2", ""):
                stats[f"drop:MS level {lvl}"] += 1; continue
            # HCD-only applies to the Orbitrap group ONLY (see design note 2)
            if a.group == "orbi" and frag_of(si, it) == "CID":
                stats["drop:labelled CID"] += 1; continue
            pt = r.get("PRECURSOR_TYPE", "")
            if pt not in AIDX:
                stats["drop:adduct"] += 1; continue
            smi = r.get("SMILES", "")
            if not smi:
                stats["drop:no smiles"] += 1; continue
            try:
                pmz = float(r.get("PRECURSOR_MZ", 0) or 0)
            except ValueError:
                pmz = 0.0
            ok, _theo = prec_mz_ok(smi, pt, pmz)
            if not ok:
                stats["drop:precursor m/z disagrees with SMILES"] += 1; continue
            cp = clean_peaks(r["_mz"], r["_it"], pmz)
            if cp is None:
                stats["drop:peaks after cleaning"] += 1; continue
            recs.append({"title": r.get("TITLE", ""), "smiles": smi, "adduct": pt,
                         "ce": r.get("COLLISION_ENERGY", ""), "prec_mz": pmz,
                         "mz": cp[0], "it": cp[1]})
        print(f"  {f}: +{len(recs)-n0} kept", flush=True)
    print(f"\nafter filtering: {len(recs)} spectra", flush=True)
    for k, v in stats.most_common(8):
        print(f"    {k:<34}{v:>10,}", flush=True)

    # ---------- 2. skeletons, dedup, CE-ladder cap ----------
    smis = sorted({r["smiles"] for r in recs})
    print(f"\nunique SMILES: {len(smis)} -> resolving skeletons", flush=True)
    with Pool(a.workers) as p:
        sk = dict(zip(smis, p.map(skel, smis, chunksize=256)))
    seen, capped = set(), collections.Counter()
    kept = []
    for r in recs:
        k = sk.get(r["smiles"])
        if k is None:
            continue
        r["skel"] = k
        # COMPOUND + ENV. env is (adduct, NCE), and NCE is the PARSED value -- deduping on the raw
        # CE string would keep "35" and "NCE=35%" as two records describing one experiment.
        nce = ce_to_nce(r["ce"], r["prec_mz"], ADDUCT_CHARGE[r["adduct"]])
        r["nce"] = nce
        dedup = (k, r["adduct"], round(nce, 2))
        if dedup in seen:
            continue
        seen.add(dedup)
        ckey = k if a.cap_scope == "compound" else (k, r["adduct"])
        if a.max_ce and capped[ckey] >= a.max_ce:
            continue
        capped[ckey] += 1
        kept.append(r)
    print(f"after dedup (compound+env) + cap <= {a.max_ce or 'inf'} per {a.cap_scope}: "
          f"{len(kept)} spectra, "
          f"{len({r['skel'] for r in kept})} skeletons", flush=True)

    # ---------- 3. featurise ----------
    smis = sorted({r["smiles"] for r in kept})
    print(f"\nfeaturising {len(smis)} molecules (ETKDG + bond graph)", flush=True)
    feats = {}
    with Pool(a.workers) as p:
        for i, (s, f) in enumerate(p.imap_unordered(_w, smis, chunksize=32)):
            if f is not None:
                feats[s] = f
            if (i + 1) % 5000 == 0:
                print(f"    {i+1}/{len(smis)}", flush=True)
    kept = [r for r in kept if r["smiles"] in feats]
    print(f"featurised {len(feats)}/{len(smis)}; spectra remaining {len(kept)}", flush=True)

    # ---------- 4. split by skeleton, CASMI forced out of train ----------
    casmi = set()
    for c in a.casmi:
        if not os.path.exists(c):
            continue
        with open(c) as fh:
            for row in csv.DictReader(fh):
                s = row.get("SMILES") or row.get("smiles") or ""
                k = skel(s)
                if k:
                    casmi.add(k)
    keys = sorted({r["skel"] for r in kept})
    rng = random.Random(42); rng.shuffle(keys)
    n = len(keys); n_val = int(0.10 * n); n_test = int(0.10 * n)
    val_k = set(keys[:n_val]); test_k = set(keys[n_val:n_val + n_test])
    train_k = set(keys[n_val + n_test:])
    # CASMI queries must be in TEST only. Moving them out of train is not enough: val drives early
    # stopping and checkpoint selection, so CASMI compounds sitting in val make model selection
    # partly a function of CASMI performance -- a milder bias than training on them, but still one,
    # and invisible in the reported number. (Measured before this fix: 14 of 188 CASMI skeletons
    # were in qtof_all_val.)
    moved_tr = train_k & casmi
    moved_va = val_k & casmi
    train_k -= moved_tr; val_k -= moved_va
    test_k |= (moved_tr | moved_va)
    print(f"\nskeletons: train {len(train_k)} / val {len(val_k)} / test {len(test_k)}", flush=True)
    print(f"  CASMI query skeletons moved to test: {len(moved_tr)} from train, "
          f"{len(moved_va)} from val", flush=True)
    assert not (train_k & casmi) and not (val_k & casmi), "CASMI leaked into train or val"

    # ---------- 5. write ----------
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for name, ks in (("train", train_k), ("val", val_k), ("test", test_k)):
        out = []
        for r in kept:
            if r["skel"] not in ks:
                continue
            arr, idx, msk, _ = feats[r["smiles"]]
            env = np.zeros(6, dtype=np.float32)
            # ALL spectra carry NCE, never eV. molnetpack's parse_collision_energy handles ~15
            # source-specific CE string formats and converts eV -> NCE with
            #     nce = ce * 500 * charge_factor / precursor_mz
            # It returns NCE as a FRACTION (0.35 for 35%). Using one convention across both
            # instrument groups is what makes them safely mergeable later.
            env[0] = r["nce"]
            env[1 + AIDX[r["adduct"]]] = 1.0
            out.append({"title": r["title"], "smiles": r["smiles"], "mol": arr,
                        "neighbor_idx": idx, "neighbor_mask": msk, "env": env,
                        "spec": bin_spec(r["mz"], r["it"])})
        path = f"{a.out}_{name}.pkl"
        with open(path + ".tmp", "wb") as fh:
            pickle.dump(out, fh)
        os.replace(path + ".tmp", path)
        print(f"  {path}: {len(out)} spectra, {len({r['smiles'] for r in out})} molecules", flush=True)
    report_stats(a.out, "Orbitrap (HCD)" if a.group == "orbi" else "QTOF/QQQ (CID+HCD)")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
