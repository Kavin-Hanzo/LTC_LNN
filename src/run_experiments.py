"""
run_experiments.py  —  Run all 4 models sequentially and compare results.

This script:
  1. Trains each architecture one by one  (same data, same config)
  2. Evaluates each on the held-out test set
  3. Prints a leaderboard
  4. Optionally auto-promotes the best model to artifacts/best/

Usage:
    # Run all 4 models with defaults from config.yaml
    python run_experiments.py

    # Run specific models only
    python run_experiments.py --models lstm gru

    # Override training mode
    python run_experiments.py --mode mimo --horizon 30 --promote
    python run_experiments.py --mode autoreg

    # Override ticker / horizon for all runs
    python run_experiments.py --ticker TSLA --horizon 14 --promote

    # Skip already-trained models (resume interrupted run)
    python run_experiments.py --skip-trained --promote

    # Auto-promote best at the end
    python run_experiments.py --promote
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
    parser.add_argument("--ticker",  type=str,   default=None,
                        help="Stock ticker (overrides config)")
    parser.add_argument("--mode",    type=str,   default=None,
                        choices=["mimo", "autoreg"],
                        help="Training mode: mimo | autoreg (overrides config)")
    parser.add_argument("--horizon", type=int,   default=None,
                        help="Forecast horizon in days (overrides config)")
    parser.add_argument("--epochs",  type=int,   default=None,
                        help="Max epochs per model (overrides config)")
    parser.add_argument("--lr",      type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--config",  type=str,   default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--promote", action="store_true",
                        help="Auto-promote best model to artifacts/best/ when done")
    parser.add_argument("--skip-trained", action="store_true",
                        help="Skip models that already have a metrics.json")
    return parser.parse_args()


# ── Per-model train + evaluate ────────────────────────────────────────────────

def run_one(
    arch:          str,
    config:        dict,
    loaders,
    device:        torch.device,
    training_mode: str,
    horizon:       int,
) -> dict:
    """Train + evaluate a single architecture. Returns metrics dict."""

    input_dim  = len(loaders.feature_cols)
    n_features = input_dim

    # output_dim: MIMO = horizon, autoreg = 1
    output_dim = horizon if training_mode == "mimo" else 1
    config["training"]["forecast_horizon"] = output_dim
    model = build_model(arch, config, input_dim=input_dim)
    config["training"]["forecast_horizon"] = horizon    # restore

    trainer = Trainer(
        model         = model,
        train_loader  = loaders.train,
        val_loader    = loaders.val,
        config        = config,
        arch          = arch,
        device        = device,
        training_mode = training_mode,
    )
    train_result = trainer.run()

    # reload best checkpoint before eval
    ckpt = torch.load(train_result["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    evaluator = Evaluator(
        model                  = model,
        test_loader            = loaders.test,
        scaler                 = loaders.scaler,
        close_col_idx          = loaders.close_col_idx,
        n_features             = n_features,
        arch                   = arch,
        config                 = config,
        device                 = device,
        original_close_col_idx = loaders.original_close_col_idx,
    )
    metrics = evaluator.run()

    # patch training info
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
        "arch":                   arch,
        "ticker":                 config["data"]["ticker"],
        "training_mode":          training_mode,
        "window_size":            config["data"]["window_size"],
        "forecast_horizon":       horizon,
        "output_dim":             output_dim,
        "features":               loaders.feature_cols,
        "all_features":           loaders.all_feature_cols,
        "boruta_used":            loaders.boruta_used,
        "close_col_idx":          loaders.close_col_idx,
        "original_close_col_idx": loaders.original_close_col_idx,
        "input_dim":              input_dim,
        "hidden_size":            config["models"]["hidden_size"],
        "num_layers":             config["models"]["num_layers"],
        "dropout":                config["models"]["dropout"],
        "tau_constant":           config["models"]["lnn"]["tau_constant"],
        "ode_unfolds":            config["models"]["lnn"]["ode_unfolds"],
        "dt":                     config["models"]["lnn"]["dt"],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return metrics


# ── Leaderboard ───────────────────────────────────────────────────────────────

def print_leaderboard(results: list, training_mode: str):
    results_sorted = sorted(results, key=lambda r: r.get("test_rmse", float("inf")))

    col = 10
    print(f"\n{'='*82}")
    print(f"  EXPERIMENT RESULTS  mode={training_mode.upper()}  (sorted by RMSE)")
    print(f"  {'RANK':<6}{'ARCH':<8}{'MAE':>{col}}{'RMSE':>{col}}"
          f"{'MAPE%':>{col}}{'R2':>{col}}{'VAL LOSS':>{col}}{'EPOCHS':>{col}}")
    print(f"  {'-'*76}")

    for i, r in enumerate(results_sorted):
        rank = f"#{i+1}" + (" ★" if i == 0 else "")
        print(
            f"  {rank:<6}"
            f"{r.get('arch','?').upper():<8}"
            f"{r.get('test_mae',       'N/A'):>{col}}"
            f"{r.get('test_rmse',      'N/A'):>{col}}"
            f"{r.get('test_mape',      'N/A'):>{col}}"
            f"{r.get('test_r2',        'N/A'):>{col}}"
            f"{r.get('best_val_loss',  'N/A'):>{col}}"
            f"{r.get('epochs_trained', 'N/A'):>{col}}"
        )

    print(f"{'='*82}")

    # MIMO: also print per-step RMSE for each arch if available
    if training_mode == "mimo":
        has_per_step = [r for r in results_sorted if r.get("per_step_metrics")]
        if has_per_step:
            print(f"\n  PER-STEP RMSE  (step_1 = next day, step_N = last day)")
            print(f"  {'ARCH':<8}", end="")
            steps = list(has_per_step[0]["per_step_metrics"].keys())
            for s in steps[:10]:   # show up to 10 steps
                print(f"  {s:>10}", end="")
            print()
            print(f"  {'-'*min(8+len(steps[:10])*12, 78)}")
            for r in has_per_step:
                print(f"  {r.get('arch','?').upper():<8}", end="")
                for s in steps[:10]:
                    v = r["per_step_metrics"][s].get("rmse", "N/A")
                    print(f"  {v:>10}", end="")
                print()
            print()

    best = results_sorted[0]
    print(f"\n  Best: {best['arch'].upper()}  "
          f"RMSE=${best.get('test_rmse')}  MAPE={best.get('test_mape')}%")
    return best["arch"]


# ── Promote ───────────────────────────────────────────────────────────────────

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

    # apply CLI overrides
    if args.mode:    config["training"]["training_mode"]    = args.mode
    if args.horizon: config["training"]["forecast_horizon"] = args.horizon
    if args.epochs:  config["training"]["epochs"]           = args.epochs
    if args.lr:      config["training"]["lr"]               = args.lr
    if args.ticker:  config["data"]["ticker"]               = args.ticker

    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ticker        = config["data"]["ticker"]
    horizon       = config["training"]["forecast_horizon"]
    training_mode = config["training"].get("training_mode", "mimo")
    output_dim    = horizon if training_mode == "mimo" else 1

    print(f"\n{'='*72}")
    print(f"  EXPERIMENT RUN")
    print(f"  Models  : {args.models}")
    print(f"  Ticker  : {ticker}")
    print(f"  Mode    : {training_mode.upper()}")
    print(f"  Horizon : {horizon} days  |  Output dim: {output_dim}")
    print(f"  Device  : {device}")
    print(f"{'='*72}")

    # build data ONCE — all models share the same splits + scaler
    loaders = build_dataloaders(config, ticker=ticker, horizon=horizon)

    results = []
    skipped = []
    failed  = []
    t_total = time.time()

    for arch in args.models:

        # skip-trained check
        metrics_path = os.path.join(config["artifacts"]["base_dir"], arch, "metrics.json")
        if args.skip_trained and os.path.exists(metrics_path):
            print(f"\n  [{arch.upper()}]  already trained — skipping")
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
            metrics = run_one(arch, config, loaders, device, training_mode, horizon)
            results.append(metrics)
            elapsed = round(time.time() - t_arch, 1)
            print(f"\n  [{arch.upper()}]  done in {elapsed}s  "
                  f"RMSE={metrics.get('test_rmse')}")
        except Exception as e:
            print(f"\n  [{arch.upper()}]  FAILED: {e}")
            failed.append(arch)

    # summary
    total_time = round(time.time() - t_total, 1)
    print(f"\n\nTotal time: {total_time}s")
    if skipped: print(f"Skipped: {skipped}")
    if failed:  print(f"Failed:  {failed}")

    if not results:
        print("No results to compare.")
        return

    best_arch = print_leaderboard(results, training_mode)

    if args.promote:
        promote(best_arch, config)
        print(f"  FastAPI will now serve {best_arch.upper()}\n")
    else:
        print(f"\n  To promote:  python promote.py --model {best_arch}\n")


if __name__ == "__main__":
    main()