"""
promote.py  —  Promote a trained model to artifacts/best/

Copies model.pth, scaler.pkl, meta.json, metrics.json
from artifacts/{arch}/  →  artifacts/best/

Usage:
    python promote.py --model lstm
    python promote.py --model lnn
    python promote.py --auto              # auto-picks lowest test_rmse
    python promote.py --auto --sort mape  # auto-picks lowest test_mape
"""

import argparse
import json
import os
import shutil

from models import SUPPORTED_MODELS


SORT_KEYS = {
    "rmse": "test_rmse",
    "mae":  "test_mae",
    "mape": "test_mape",
    "r2":   "test_r2",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Promote a model to artifacts/best/.")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=SUPPORTED_MODELS,
                       help="Manually choose which model to promote")
    group.add_argument("--auto",  action="store_true",
                       help="Auto-promote model with best metric")
    parser.add_argument("--sort", default="rmse", choices=list(SORT_KEYS.keys()),
                        help="Metric to use for --auto selection (default: rmse)")
    return parser.parse_args()


def pick_best_auto(base_dir: str, sort_key: str) -> str:
    best_arch  = None
    best_val   = float("inf")

    for arch in SUPPORTED_MODELS:
        path = os.path.join(base_dir, arch, "metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            m = json.load(f)
        val = m.get(sort_key, float("inf"))
        # for R2: higher is better — invert comparison
        if sort_key == "test_r2":
            val = -val
        if val < best_val:
            best_val  = val
            best_arch = arch

    if best_arch is None:
        raise RuntimeError("No trained models found. Run train.py first.")
    return best_arch


def main():
    args     = parse_args()
    base_dir = "artifacts"
    best_dir = os.path.join(base_dir, "best")
    sort_key = SORT_KEYS[args.sort]

    if args.auto:
        arch = pick_best_auto(base_dir, sort_key)
        print(f"\n  Auto-selected by {args.sort.upper()}: {arch.upper()}")
    else:
        arch = args.model

    src      = os.path.join(base_dir, arch)
    required = ["model.pth", "scaler.pkl", "meta.json", "metrics.json"]

    for fname in required:
        fpath = os.path.join(src, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Missing artifact: {fpath}\n"
                f"Run:  python train.py --model {arch}"
            )

    os.makedirs(best_dir, exist_ok=True)
    for fname in required:
        shutil.copy2(os.path.join(src, fname), os.path.join(best_dir, fname))

    with open(os.path.join(best_dir, "metrics.json")) as f:
        m = json.load(f)
    with open(os.path.join(best_dir, "meta.json")) as f:
        meta = json.load(f)

    mode = meta.get("training_mode", "?").upper()
    print(f"\n{'='*52}")
    print(f"  Promoted  :  {arch.upper()}  →  artifacts/best/")
    print(f"  Ticker    :  {meta.get('ticker')}")
    print(f"  Mode      :  {mode}")
    print(f"  Horizon   :  {meta.get('forecast_horizon')} days")
    print(f"  Output dim:  {meta.get('output_dim', '?')} neuron(s)")
    print(f"  Test RMSE :  ${m.get('test_rmse')}")
    print(f"  Test MAE  :  ${m.get('test_mae')}")
    print(f"  Test MAPE :  {m.get('test_mape')}%")
    print(f"  Test R2   :  {m.get('test_r2')}")

    # MIMO: show first few per-step RMSE values
    if m.get("per_step_metrics"):
        steps = list(m["per_step_metrics"].items())
        print(f"  Per-step  :  ", end="")
        for step, metrics in steps[:5]:
            print(f"{step}=${metrics['rmse']}  ", end="")
        if len(steps) > 5:
            print(f"... (+{len(steps)-5} more)")
        else:
            print()

    print(f"{'='*52}")
    print(f"\n  FastAPI server will load from artifacts/best/\n")


if __name__ == "__main__":
    main()