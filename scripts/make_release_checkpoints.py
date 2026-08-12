"""Convert trained checkpoints into release artifacts: shipped key names + embedded config.

Two things are wrong with a raw training checkpoint, and both are release blockers.

1. NAMING. The experiment trainers used their own attribute names -- `head.*` for the scalar
   (RT/CCS) head. The shipped `MolNetScalar` calls it `decoder.*`, so the checkpoint does not load.
   The MS/MS trainer's names (`trunk`, `forw`, `rev`, `gate`) match the v1.4.0 `MolNet_MS` and
   pass through unchanged.

2. NO METADATA. A raw checkpoint stores only weights and a metric. Nothing records the encoder
   mode, the adduct one-hot layout, or the binning. None of those change a tensor shape, so a
   mismatch loads cleanly and predicts from the wrong model. `MolNet.load_checkpoint` refuses a
   checkpoint whose embedded config contradicts the loaded one -- but only if the config is there.

Usage:
    python scripts/make_release_checkpoints.py --dry_run     # show what would be written
    python scripts/make_release_checkpoints.py
    python scripts/make_release_checkpoints.py --refresh_meta

--refresh_meta re-embeds the CURRENT config into already-converted release checkpoints
(weights untouched). Use it when a config key is added after conversion -- e.g. `ce_scale`,
added when the percent/fraction NCE mismatch was found -- and the source training
checkpoints are no longer on this machine.
"""
import argparse
import os
import sys

import torch
import yaml

M3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, M3)

from molnetpack import __version__

CKPT_DIR = os.path.join(M3, "check_point")


# SELECTION RULE, applied uniformly:
#   1. rank on VALIDATION only -- test is reported but never used to choose;
#   2. when the val spread is inside the resolution of the split, prefer the arm that uses the
#      SHARED recipe (lr 1e-4, weight_decay 0.05), so one documented recipe reproduces every
#      released model instead of four per-model footnotes.
#
# QTOF (val cosine, 7,275-spectrum split):
#     lr 3e-4 (the original release run)   val 0.5795  test 0.5527
#     lr 1e-4                              val 0.6133  test 0.5922
#     lr 3e-5                              val 0.6208  test 0.5934
#     lr 1e-4 + wd 0.05                    val 0.6202  test 0.5952   <- shipped (rule 2, -0.0006)
#
# CCS (val MAE, 446-row split -- differences below ~0.3 MAE are not resolvable here):
#     lr 1e-4 + wd 0.05                    val 4.178   test 3.962 / medRel 1.77%  <- shipped
#     lr 1e-4 + wd 0.2                     val 4.116   test 4.005 / medRel 1.87%
#   wd 0.2 leads on val by 0.06, well inside the noise, and trails on both test figures.
#
# source checkpoint -> (release name, config file, task)
# Orbitrap (val cosine, 43,715-spectrum split), both arms early-stopped:
#     lr 1e-4                              val 0.5206  test 0.5193
#     lr 1e-4 + wd 0.05                    val 0.5251  test 0.5233   <- shipped, wins outright
#
# RT was retrained on the fixed trainer so the checkpoint records its own pretraining
# (val MAE 50.208 -> 49.950, test medRel 3.70% -> 3.62%).
RELEASE = [
    ("tune_lr1e4_wd05.pt", "molnet_qtof_v1.4.0.pt", "molnet.yml", "msms"),
    ("orbi_lr1e4_wd05.pt", "molnet_orbitrap_v1.4.0.pt", "molnet.yml", "msms"),
    ("release_rt_v2.pt", "molnet_rt_v1.4.0.pt", "molnet_rt_tl.yml", "rt"),
    ("release_ccs.pt", "molnet_ccs_v1.4.0.pt", "molnet_ccs_tl.yml", "ccs"),
]

PENDING = []

# experiment attribute -> shipped attribute
RENAME = {"head.": "decoder."}

# The ChEMBL-pretrained encoder ships too (the transfer source for all four task models).
# Its raw form is pretrain_geo.py's {encoder_state_dict, k, sigma, val_loss}; the release
# form wraps it as a standard transfer source: encoder.*-prefixed model_state_dict plus the
# encoder-defining config subset, so `MolNet.train(resume_path=..., transfer=True)` loads
# and validates it like any other checkpoint.
PRETRAIN_SRC = "molnet_pre_geobond.pt"
ENCODER_CFG_KEYS = ("in_dim", "encode_layers", "emb_dim", "max_atom_num", "k")


def convert_pretrain(dry_run):
    from molnetpack import config_path
    src = os.path.join(CKPT_DIR, PRETRAIN_SRC)
    if not os.path.exists(src):
        print(f"SKIP {PRETRAIN_SRC}: not found")
        return
    raw = torch.load(src, map_location="cpu", weights_only=False)
    if "model_state_dict" in raw:
        print(f"{PRETRAIN_SRC}: already in release format")
        return
    mcfg = yaml.safe_load(open(config_path("molnet.yml")))["model"]
    payload = {
        "model_state_dict": {f"encoder.{k}": v for k, v in raw["encoder_state_dict"].items()},
        "config": {k: mcfg[k] for k in ENCODER_CFG_KEYS},
        "task": "pretrain",
        "version": __version__,
        "encoder_mode": "bond",
        "sigma": raw.get("sigma"),
        "val_loss": raw.get("val_loss"),
    }
    print(f"{PRETRAIN_SRC:28s} -> release transfer-source format")
    if not dry_run:
        torch.save(payload, src + ".tmp")
        os.replace(src + ".tmp", src)


def convert(state_dict):
    out, renamed = {}, 0
    for key, value in state_dict.items():
        new_key = key
        for old, new in RENAME.items():
            if key.startswith(old):
                new_key = new + key[len(old):]
                renamed += 1
                break
        out[new_key] = value
    return out, renamed


def refresh_meta(dry_run):
    """Re-embed the current config into existing release checkpoints (weights untouched)."""
    from molnetpack import config_path

    for _, dst_name, cfg_name, task in RELEASE:
        dst = os.path.join(CKPT_DIR, dst_name)
        if not os.path.exists(dst):
            print(f"SKIP {dst_name}: not found")
            continue
        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        cfg = yaml.safe_load(open(config_path(cfg_name)))["model"]
        added = sorted(set(cfg) - set(ckpt.get("config", {})))
        changed = sorted(k for k in set(cfg) & set(ckpt.get("config", {}))
                         if cfg[k] != ckpt["config"][k])
        ckpt["config"] = cfg
        print(f"{dst_name:28s} | config refreshed from {cfg_name} | "
              f"added {added or '[]'} | changed {changed or '[]'}")
        if not dry_run:
            torch.save(ckpt, dst + ".tmp")
            os.replace(dst + ".tmp", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--refresh_meta", action="store_true",
                    help="re-embed the current config into existing release checkpoints")
    args = ap.parse_args()

    if args.refresh_meta:
        refresh_meta(args.dry_run)
        return

    convert_pretrain(args.dry_run)

    for src_name, dst_name, cfg_name, task in RELEASE:
        src = os.path.join(CKPT_DIR, src_name)
        dst = os.path.join(CKPT_DIR, dst_name)
        if not os.path.exists(src):
            print(f"SKIP {src_name}: not found (still training?)")
            continue

        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        from molnetpack import config_path
        cfg = yaml.safe_load(open(config_path(cfg_name)))
        state, renamed = convert(ckpt["model_state_dict"])

        payload = {
            "model_state_dict": state,
            "config": cfg["model"],
            "task": task,
            "version": __version__,
            "source_checkpoint": src_name,
            # bond mode is not a config key -- it is a property of how the model was fed, so record
            # it explicitly rather than leaving it to be inferred
            "encoder_mode": "bond",
        }
        for carry in ("val_cos", "val_mae", "mu", "sd", "seed", "pretrain"):
            if carry in ckpt:
                payload[carry] = ckpt[carry]

        metric = ckpt.get("val_cos", ckpt.get("val_mae"))
        print(f"{src_name:28s} -> {dst_name:28s} | {renamed:3d} keys renamed | "
              f"val {metric:.4f} | config {cfg_name}")
        if not args.dry_run:
            torch.save(payload, dst + ".tmp")
            os.replace(dst + ".tmp", dst)

    for note in PENDING:
        print(f"PENDING  {note}")
    if args.dry_run:
        print("\n(dry run -- nothing written)")


if __name__ == "__main__":
    main()
