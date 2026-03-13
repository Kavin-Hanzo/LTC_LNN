"""
promote.py  —  Promote a trained model to artifacts/best/

Copies model.pth, scaler.pkl, meta.json, metrics.json
from artifacts/{arch}/  →  artifacts/best/

Usage:
    python promote.py --model lstm
    python promote.py --model lnn
    python promote.py --auto          # auto-picks lowest test_rmse
"""

import argparse
import json
import os
import shutil

from models import SUPPORTED_MODELS


def parse_args():
    parser = argparse.ArgumentParser(description="Promote a model to artifacts/best/.")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=SUPPORTED_MODELS,
                       help="Manually choose which model to promote")
    group.add_argument("--auto",  action="store_true",
                       help="Auto-promote model with lowest test_rmse")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def pick_best_auto(base_dir: str) -> str:
    best_arch  = None
    best_rmse  = float("inf")

    for arch in SUPPORTED_MODELS:
        path = os.path.join(base_dir, arch, "metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            m = json.load(f)
        rmse = m.get("test_rmse", float("inf"))
        if rmse < best_rmse:
            best_rmse = rmse
            best_arch = arch

    if best_arch is None:
        raise RuntimeError("No trained models found. Run train.py first.")
    return best_arch


def main():
    args     = parse_args()
    base_dir = "artifacts"
    best_dir = os.path.join(base_dir, "best")

    if args.auto:
        arch = pick_best_auto(base_dir)
        print(f"\n  Auto-selected: {arch.upper()}")
    else:
        arch = args.model

    src = os.path.join(base_dir, arch)
    required = ["model.pth", "scaler.pkl", "meta.json", "metrics.json"]

    # ── verify source artifacts exist
    for fname in required:
        fpath = os.path.join(src, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Missing artifact: {fpath}\n"
                f"Run:  python train.py --model {arch}"
            )

    # ── copy to best/
    os.makedirs(best_dir, exist_ok=True)
    for fname in required:
        shutil.copy2(os.path.join(src, fname), os.path.join(best_dir, fname))

    # ── load and print metrics
    with open(os.path.join(best_dir, "metrics.json")) as f:
        m = json.load(f)
    with open(os.path.join(best_dir, "meta.json")) as f:
        meta = json.load(f)

    print(f"\n{'='*50}")
    print(f"  Promoted  :  {arch.upper()}  →  artifacts/best/")
    print(f"  Ticker    :  {meta.get('ticker')}")
    print(f"  Horizon   :  {meta.get('forecast_horizon')} days")
    print(f"  Test RMSE :  {m.get('test_rmse')}")
    print(f"  Test MAE  :  {m.get('test_mae')}")
    print(f"  Test MAPE :  {m.get('test_mape')}%")
    print(f"{'='*50}")
    print(f"\n  FastAPI server will load from artifacts/best/\n")


if __name__ == "__main__":
    main()
