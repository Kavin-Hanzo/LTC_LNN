"""
run_experiments.py  —  Run all 4 models sequentially and compare results.

This script:
  1. Trains each architecture one by one  (same data, same config)
  2. Evaluates each on the held-out test set
  3. Prints a leaderboard
  4. Optionally auto-promotes the best model to artifacts/best/

Usage:
    # Run all 4 models with defaults
    python run_experiments.py

    # Run specific models only
    python run_experiments.py --models lstm gru

    # Run all + auto-promote best at the end
    python run_experiments.py --promote

    # Override ticker / horizon for all runs
    python run_experiments.py --ticker TSLA --horizon 14 --promote

    # Skip already-trained models (resume interrupted run)
    python run_experiments.py --skip-trained
"""

import argparse
import json
import os
import random
import shutil
import time

import numpy as np
import torch
import joblib

from pipeline.data_pipeline import build_dataloaders, load_config
from pipeline.trainer       import Trainer
from pipeline.evaluate      import Evaluator
from models                 import build_model, SUPPORTED_MODELS


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all model experiments and compare results."
    )
    parser.add_argument(
        "--models", nargs="+", choices=SUPPORTED_MODELS, default=SUPPORTED_MODELS,
        help=f"Models to train (default: all). Choose from {SUPPORTED_MODELS}"
    )
    parser.add_argument("--ticker",       type=str,   default=None,
                        help="Stock ticker (overrides config)")
    parser.add_argument("--horizon",      type=int,   default=None,
                        help="Forecast horizon in days (overrides config)")
    parser.add_argument("--epochs",       type=int,   default=None,
                        help="Max epochs per model (overrides config)")
    parser.add_argument("--lr",           type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--config",       type=str,   default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--promote",      action="store_true",
                        help="Auto-promote best model to artifacts/best/ when done")
    parser.add_argument("--skip-trained", action="store_true",
                        help="Skip models that already have a metrics.json")
    return parser.parse_args()


# ── Per-model train + evaluate ────────────────────────────────────────────────

def run_one(arch: str, config: dict, loaders, device: torch.device) -> dict:
    """Train + evaluate a single architecture. Returns metrics dict."""

    input_dim  = len(config["data"]["features"])
    n_features = input_dim

    model = build_model(arch, config, input_dim=input_dim)

    trainer = Trainer(
        model        = model,
        train_loader = loaders.train,
        val_loader   = loaders.val,
        config       = config,
        arch         = arch,
        device       = device,
    )
    train_result = trainer.run()

    # reload best checkpoint before eval
    ckpt = torch.load(train_result["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    evaluator = Evaluator(
        model         = model,
        test_loader   = loaders.test,
        scaler        = loaders.scaler,
        close_col_idx = loaders.close_col_idx,
        n_features    = n_features,
        arch          = arch,
        config        = config,
        device        = device,
    )
    metrics = evaluator.run()

    # patch training info into metrics and re-save
    metrics["epochs_trained"] = train_result["epochs_trained"]
    metrics["best_val_loss"]  = round(train_result["best_val_loss"], 6)
    metrics_path = os.path.join(config["artifacts"]["base_dir"], arch, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # save scaler + meta
    out_dir     = os.path.join(config["artifacts"]["base_dir"], arch)
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    joblib.dump(loaders.scaler, scaler_path)

    meta = {
        "arch":             arch,
        "ticker":           config["data"]["ticker"],
        "window_size":      config["data"]["window_size"],
        "forecast_horizon": config["training"]["forecast_horizon"],
        "features":         config["data"]["features"],
        "close_col_idx":    loaders.close_col_idx,
        "input_dim":        input_dim,
        "hidden_size":      config["models"]["hidden_size"],
        "num_layers":       config["models"]["num_layers"],
        "dropout":          config["models"]["dropout"],
        "tau_constant":     config["models"]["lnn"]["tau_constant"],
        "ode_unfolds":      config["models"]["lnn"]["ode_unfolds"],
        "dt":               config["models"]["lnn"]["dt"],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return metrics


# ── Leaderboard printer ───────────────────────────────────────────────────────

def print_leaderboard(results: list):
    results_sorted = sorted(results, key=lambda r: r.get("test_rmse", float("inf")))

    col = 11
    print(f"\n{'='*72}")
    print(f"  EXPERIMENT RESULTS  (sorted by Test RMSE)")
    print(f"  {'RANK':<6}{'ARCH':<8}{'TEST MAE':>{col}}{'TEST RMSE':>{col}}"
          f"{'TEST MAPE%':>{col}}{'VAL LOSS':>{col}}{'EPOCHS':>{col}}")
    print(f"  {'-'*66}")

    for i, r in enumerate(results_sorted):
        rank = f"#{i+1}" + (" ★" if i == 0 else "")
        print(
            f"  {rank:<6}"
            f"{r.get('arch','?').upper():<8}"
            f"{r.get('test_mae',   'N/A'):>{col}}"
            f"{r.get('test_rmse',  'N/A'):>{col}}"
            f"{r.get('test_mape',  'N/A'):>{col}}"
            f"{r.get('best_val_loss','N/A'):>{col}}"
            f"{r.get('epochs_trained','N/A'):>{col}}"
        )

    print(f"{'='*72}")
    best = results_sorted[0]
    print(f"\n  Best model: {best['arch'].upper()}  "
          f"(RMSE={best.get('test_rmse')}  MAPE={best.get('test_mape')}%)")
    return best["arch"]


# ── Promote helper ────────────────────────────────────────────────────────────

def promote(arch: str, config: dict):
    base_dir = config["artifacts"]["base_dir"]
    best_dir = config["artifacts"]["best_dir"]
    src      = os.path.join(base_dir, arch)

    os.makedirs(best_dir, exist_ok=True)
    for fname in ["model.pth", "scaler.pkl", "meta.json", "metrics.json"]:
        shutil.copy2(os.path.join(src, fname), os.path.join(best_dir, fname))

    print(f"\n  Promoted {arch.upper()} -> {best_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = load_config(args.config)

    # apply overrides
    if args.horizon: config["training"]["forecast_horizon"] = args.horizon
    if args.epochs:  config["training"]["epochs"]           = args.epochs
    if args.lr:      config["training"]["lr"]               = args.lr
    if args.ticker:  config["data"]["ticker"]               = args.ticker

    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ticker  = config["data"]["ticker"]
    horizon = config["training"]["forecast_horizon"]

    print(f"\n{'='*72}")
    print(f"  EXPERIMENT RUN")
    print(f"  Models  : {args.models}")
    print(f"  Ticker  : {ticker}")
    print(f"  Horizon : {horizon} days")
    print(f"  Device  : {device}")
    print(f"{'='*72}")

    # build data ONCE — all models share the same splits + scaler
    from pipeline.data_pipeline import build_dataloaders
    loaders = build_dataloaders(config, ticker=ticker, horizon=horizon)

    results   = []
    skipped   = []
    failed    = []
    t_total   = time.time()

    for arch in args.models:

        # skip-trained check
        metrics_path = os.path.join(config["artifacts"]["base_dir"], arch, "metrics.json")
        if args.skip_trained and os.path.exists(metrics_path):
            print(f"\n  [{arch.upper()}]  already trained — skipping  "
                  f"(remove --skip-trained to retrain)")
            with open(metrics_path) as f:
                results.append(json.load(f))
            skipped.append(arch)
            continue

        print(f"\n\n{'#'*72}")
        print(f"  TRAINING  {arch.upper()}  "
              f"({args.models.index(arch)+1}/{len(args.models)})")
        print(f"{'#'*72}")

        t_arch = time.time()
        try:
            metrics = run_one(arch, config, loaders, device)
            results.append(metrics)
            elapsed = round(time.time() - t_arch, 1)
            print(f"\n  [{arch.upper()}]  done in {elapsed}s  "
                  f"RMSE={metrics.get('test_rmse')}")
        except Exception as e:
            print(f"\n  [{arch.upper()}]  FAILED: {e}")
            failed.append(arch)

    # ── summary
    total_time = round(time.time() - t_total, 1)
    print(f"\n\nTotal experiment time: {total_time}s")
    if skipped: print(f"Skipped (already trained): {skipped}")
    if failed:  print(f"Failed: {failed}")

    if not results:
        print("No results to compare.")
        return

    best_arch = print_leaderboard(results)

    if args.promote:
        promote(best_arch, config)
        print(f"  FastAPI server will now serve {best_arch.upper()}\n")
    else:
        print(f"\n  To promote best model, run:")
        print(f"    python promote.py --model {best_arch}")
        print(f"  Or re-run with --promote flag\n")


if __name__ == "__main__":
    main()
