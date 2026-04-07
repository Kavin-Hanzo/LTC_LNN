"""
compare.py  —  Print a leaderboard of all trained models.

Reads artifacts/{arch}/metrics.json for each trained arch.
For MIMO models, also shows per-step RMSE across the forecast horizon.

Usage:
    python compare.py
    python compare.py --sort rmse
    python compare.py --sort mape
    python compare.py --sort r2
    python compare.py --steps        # show per-step RMSE table (MIMO only)
"""

import argparse
import json
import os

from models import SUPPORTED_MODELS


SORT_KEYS = {
    "rmse": "test_rmse",
    "mae":  "test_mae",
    "mape": "test_mape",
    "r2":   "test_r2",
    "val":  "best_val_loss",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare trained models.")
    parser.add_argument("--sort",  default="rmse", choices=list(SORT_KEYS.keys()),
                        help="Sort leaderboard by this metric (default: rmse)")
    parser.add_argument("--steps", action="store_true",
                        help="Show per-step RMSE breakdown (MIMO models only)")
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


def print_main_table(rows: list, sort_key: str):
    col = 10
    print(f"\n{'='*84}")
    print(f"  {'RANK':<6}{'ARCH':<8}{'MODE':<9}"
          f"{'MAE':>{col}}{'RMSE':>{col}}{'MAPE%':>{col}}"
          f"{'R2':>{col}}{'VAL LOSS':>{col}}{'EPOCHS':>{col}}")
    print(f"  {'-'*78}")

    best_arch = None
    for i, r in enumerate(rows):
        rank_str = f"#{i+1}"
        if i == 0:
            rank_str += " ★"
            best_arch = r.get("arch", "?")

        mode = r.get("eval_mode", "?")[:8]
        print(
            f"  {rank_str:<6}"
            f"{r.get('arch','?').upper():<8}"
            f"{mode:<9}"
            f"{r.get('test_mae',       'N/A'):>{col}}"
            f"{r.get('test_rmse',      'N/A'):>{col}}"
            f"{r.get('test_mape',      'N/A'):>{col}}"
            f"{r.get('test_r2',        'N/A'):>{col}}"
            f"{r.get('best_val_loss',  'N/A'):>{col}}"
            f"{r.get('epochs_trained', 'N/A'):>{col}}"
        )

    print(f"{'='*84}")
    return best_arch


def print_per_step_table(rows: list):
    """Print RMSE per forecast step for MIMO models."""
    mimo_rows = [r for r in rows if r.get("per_step_metrics")]
    if not mimo_rows:
        print("\n  No MIMO per-step metrics found.")
        print("  Per-step metrics are only available for models trained with --mode mimo")
        return

    # find all step keys from first row
    steps = list(mimo_rows[0]["per_step_metrics"].keys())
    show  = steps[:15]   # cap at 15 steps for readability

    print(f"\n  PER-STEP RMSE ($)  —  step_1=next day, step_N=last forecasted day")
    print(f"  {'ARCH':<8}", end="")
    for s in show:
        label = s.replace("step_", "d")
        print(f"  {label:>7}", end="")
    print()
    print(f"  {'-'*max(8 + len(show)*9, 40)}")

    for r in mimo_rows:
        print(f"  {r.get('arch','?').upper():<8}", end="")
        for s in show:
            v = r["per_step_metrics"].get(s, {}).get("rmse", "?")
            print(f"  {v:>7}", end="")
        print()
    print()


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

    best_arch = print_main_table(rows, sort_key)

    if args.steps:
        print_per_step_table(rows)

    not_trained = [a for a in SUPPORTED_MODELS
                   if not os.path.exists(os.path.join(base_dir, a, "metrics.json"))]
    if not_trained:
        print(f"\n  Not yet trained: {', '.join(not_trained)}")
        print(f"  Run: python train.py --model <arch>")

    if best_arch:
        print(f"\n  Best by {args.sort.upper()}: {best_arch.upper()}")
        print(f"  To promote:  python promote.py --model {best_arch}\n")


if __name__ == "__main__":
    main()