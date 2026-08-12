"""Unified prediction CLI for all 3DMolMS tasks.

Usage:
    python scripts/predict.py --task msms --test_data examples/demo_input.csv \
        --result_path out.mgf --instrument qtof
    python scripts/predict.py --task ccs  --test_data examples/demo_input.csv  --result_path out.csv
    python scripts/predict.py --task rt   --test_data examples/demo_input.csv   --result_path out.csv

Supported inputs: .csv, .mgf (msms only), .pkl

This is a thin wrapper over `molnetpack.MolNet`. It used to re-implement data conversion, model
construction, checkpoint loading and the inference loop, and those copies had already drifted from
the library: the duplicate MS/MS loop normalised by a GLOBAL `torch.max(pred)` (correct only at
batch size 1, which it then hardcoded), the property loop passed `env=None` so the CCS adduct was
never encoded, and neither loop passed the covalent bond graph or validated the checkpoint's
config. Delegating means one implementation to keep correct.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import init_random_seed, setup_device

from molnetpack import MolNet


def parse_args():
    parser = argparse.ArgumentParser(description="3DMolMS unified prediction script")
    parser.add_argument("--task", required=True, choices=["msms", "ccs", "rt"],
                        help="Prediction task")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Input file (.csv / .mgf / .pkl)")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Output file (.csv or .mgf for msms; .csv for ccs/rt)")
    parser.add_argument("--instrument", type=str, default="qtof", choices=["qtof", "orbitrap"],
                        help="MS/MS instrument model (ignored for ccs/rt)")
    parser.add_argument("--resume_path", type=str, default="",
                        help="Custom checkpoint. Default: the bundled release weights, downloaded "
                             "on first use into $MOLNETPACK_HOME (or the user cache directory).")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    init_random_seed(args.seed)
    device = setup_device(args.device, args.no_cuda)
    print(f"Task: {args.task}  |  Device: {device}")

    molnet = MolNet(device, seed=args.seed)
    molnet.load_data(args.test_data, batch_size=args.batch_size)

    checkpoint = args.resume_path or None
    if args.task == "msms":
        molnet.pred_msms(path_to_results=args.result_path,
                         path_to_checkpoint=checkpoint,
                         instrument=args.instrument)
    elif args.task == "ccs":
        molnet.pred_ccs(path_to_results=args.result_path, path_to_checkpoint=checkpoint)
    else:
        molnet.pred_rt(path_to_results=args.result_path, path_to_checkpoint=checkpoint)


if __name__ == "__main__":
    main()
