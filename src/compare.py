"""
compare.py  —  Print a leaderboard of all trained models.

Reads artifacts/{arch}/metrics.json for each arch that has been trained.

Usage:
    python compare.py
    python compare.py --sort rmse
    python compare.py --sort mape
"""

import argparse
import json
import os

from models import SUPPORTED_MODELS


SORT_KEYS = {
    "rmse": "test_rmse",
    "mae":  "test_mae",
    "mape": "test_mape",
    "val":  "best_val_loss",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare trained models.")
    parser.add_argument("--sort",    default="rmse", choices=list(SORT_KEYS.keys()),
                        help="Sort leaderboard by this metric (default: rmse)")
    parser.add_argument("--config",  default="config.yaml")
    return parser.parse_args()


def load_metrics(base_dir: str) -> list:
    rows = []
    for arch in SUPPORTED_MODELS:
        path = os.path.join(base_dir, arch, "metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def main():
    args     = parse_args()
    base_dir = "artifacts"

    rows = load_metrics(base_dir)

    if not rows:
        print("\n  No trained models found in artifacts/.")
        print("  Run:  python train.py --model lstm\n")
        return

    sort_key = SORT_KEYS[args.sort]
    rows.sort(key=lambda r: r.get(sort_key, float("inf")))

    # ── header
    col = 10
    print(f"\n{'='*80}")
    print(f"  {'RANK':<6}{'ARCH':<8}{'TEST MAE':>{col}}{'TEST RMSE':>{col}}"
          f"{'TEST MAPE%':>{col}}{'TEST R2':>{col}}{'VAL LOSS':>{col}}{'EPOCHS':>{col}}")
    print(f"  {'-'*74}")

    best_arch = None
    for i, r in enumerate(rows):
        rank_str = f"#{i+1}"
        if i == 0:
            rank_str += " ★"
            best_arch = r["arch"]
        print(
            f"  {rank_str:<6}"
            f"{r['arch'].upper():<8}"
            f"{r.get('test_mae',  'N/A'):>{col}}"
            f"{r.get('test_rmse', 'N/A'):>{col}}"
            f"{r.get('test_mape', 'N/A'):>{col}}"
            f"{r.get('test_r2',   'N/A'):>{col}}"
            f"{r.get('best_val_loss', 'N/A'):>{col}}"
            f"{r.get('epochs_trained', 'N/A'):>{col}}"
        )

    print(f"{'='*80}")

    not_trained = [a for a in SUPPORTED_MODELS
                   if not os.path.exists(os.path.join(base_dir, a, "metrics.json"))]
    if not_trained:
        print(f"\n  Not yet trained: {', '.join(not_trained)}")
        print(f"  Run: python train.py --model <arch>")

    if best_arch:
        print(f"\n  Best model by {args.sort.upper()}: {best_arch.upper()}")
        print(f"  To promote:  python promote.py --model {best_arch}\n")


if __name__ == "__main__":
    main()
