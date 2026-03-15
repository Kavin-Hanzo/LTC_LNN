"""
train.py  —  CLI entrypoint for training a single model.

Usage examples:
    python train.py --model lstm
    python train.py --model lnn  --ticker TSLA
    python train.py --model gru  --ticker AAPL --horizon 14 --epochs 50
    python train.py --model rnn  --lr 0.0005

Args:
    --model    rnn | lstm | gru | lnn          (required)
    --ticker   e.g. AAPL                       (default: from config.yaml)
    --horizon  forecast days e.g. 30           (default: from config.yaml)
    --epochs   override config epochs
    --lr       override config learning rate
    --config   path to config.yaml             (default: ./config.yaml)
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import joblib

from pipeline.data_pipeline import build_dataloaders, load_config
from pipeline.trainer import Trainer
from pipeline.evaluate import Evaluator
from models import build_model, SUPPORTED_MODELS


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── CLI args ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train a single forecasting model.")
    parser.add_argument("--model",   required=True, choices=SUPPORTED_MODELS,
                        help="Model architecture to train")
    parser.add_argument("--ticker",  type=str, default=None,
                        help="Stock ticker (overrides config)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Forecast horizon in days (overrides config)")
    parser.add_argument("--epochs",  type=int, default=None,
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--lr",      type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--config",  type=str, default="config.yaml",
                        help="Path to config.yaml")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = load_config(args.config)

    # ── apply CLI overrides
    if args.horizon:
        config["training"]["forecast_horizon"] = args.horizon
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["lr"] = args.lr

    ticker = args.ticker or config["data"]["ticker"]

    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Model   : {args.model.upper()}")
    print(f"  Ticker  : {ticker}")
    print(f"  Horizon : {config['training']['forecast_horizon']} days")
    print(f"  Device  : {device}")
    print(f"{'='*60}")

    # ── 1. Data
    loaders    = build_dataloaders(config, ticker=ticker)
    input_dim  = len(loaders.feature_cols)   # post-Boruta dim (may be < original)
    n_features = input_dim

    # ── 2. Model
    model = build_model(args.model, config, input_dim=input_dim)
    print(f"\n  Params  : {model.count_parameters():,}")
    print(f"  Summary : {model.model_summary()}")

    # ── 3. Train
    trainer = Trainer(
        model        = model,
        train_loader = loaders.train,
        val_loader   = loaders.val,
        config       = config,
        arch         = args.model,
        device       = device,
    )
    train_result = trainer.run()

    # ── 4. Evaluate on test set (load best checkpoint first)
    best_ckpt = torch.load(train_result["checkpoint_path"], map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])

    evaluator = Evaluator(
        model                  = model,
        test_loader            = loaders.test,
        scaler                 = loaders.scaler,
        close_col_idx          = loaders.close_col_idx,
        n_features             = n_features,
        arch                   = args.model,
        config                 = config,
        device                 = device,
        original_close_col_idx = loaders.original_close_col_idx,
    )
    metrics = evaluator.run()

    # ── 5. Patch metrics with training info
    metrics["epochs_trained"] = train_result["epochs_trained"]
    metrics["best_val_loss"]  = round(train_result["best_val_loss"], 6)
    metrics_path = os.path.join(config["artifacts"]["base_dir"], args.model, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── 6. Save scaler + meta alongside model
    out_dir = os.path.join(config["artifacts"]["base_dir"], args.model)

    scaler_path = os.path.join(out_dir, "scaler.pkl")
    joblib.dump(loaders.scaler, scaler_path)
    print(f"\n  Scaler  → {scaler_path}")

    # input_dim after Boruta may differ from original feature count
    final_input_dim = len(loaders.feature_cols)

    meta = {
        "arch":             args.model,
        "ticker":           ticker,
        "window_size":      config["data"]["window_size"],
        "forecast_horizon": config["training"]["forecast_horizon"],
        "features":             loaders.feature_cols,
        "all_features":         loaders.all_feature_cols,
        "boruta_used":          loaders.boruta_used,
        "close_col_idx":        loaders.close_col_idx,
        "original_close_col_idx": loaders.original_close_col_idx,
        "input_dim":        final_input_dim,           # matches model input layer
        "hidden_size":      config["models"]["hidden_size"],
        "num_layers":       config["models"]["num_layers"],
        "dropout":          config["models"]["dropout"],
        "tau_constant":     config["models"]["lnn"]["tau_constant"],
        "ode_unfolds":      config["models"]["lnn"]["ode_unfolds"],
        "dt":               config["models"]["lnn"]["dt"],
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta    → {meta_path}")

    print(f"\n{'='*60}")
    print(f"  Training complete for {args.model.upper()}")
    print(f"  Artifacts in: {out_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()