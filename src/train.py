"""
train.py  —  CLI entrypoint for training a single model.

Training modes:
    mimo    Direct Multi-Horizon (MIMO) — model outputs all H steps simultaneously.
            output_dim = forecast_horizon. Loss = MSE across full vector.
            No recursive drift. Recommended for most use cases.

    autoreg Single-step autoregressive — model outputs 1 step.
            output_dim = 1. Any horizon supported at inference by rolling.
            Simpler but errors compound over long horizons.

Usage examples:
    python train.py --model lstm --ticker AAPL
    python train.py --model lstm --ticker AAPL --mode mimo --horizon 30
    python train.py --model lnn  --ticker TSLA --mode autoreg
    python train.py --model gru  --ticker AAPL --epochs 50 --lr 0.0005
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
                        help="Model architecture: rnn | lstm | gru | lnn")
    parser.add_argument("--ticker",  type=str, default=None,
                        help="Stock ticker (overrides config.yaml)")
    parser.add_argument("--mode",    type=str, default=None,
                        choices=["mimo", "autoreg"],
                        help="Training mode: mimo (default) | autoreg. "
                             "Overrides config.training.training_mode")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Forecast horizon in days (overrides config)")
    parser.add_argument("--epochs",  type=int, default=None,
                        help="Max training epochs (overrides config)")
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
    if args.mode:
        config["training"]["training_mode"] = args.mode
    if args.horizon:
        config["training"]["forecast_horizon"] = args.horizon
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["lr"] = args.lr

    # resolve final values
    ticker        = args.ticker or config["data"]["ticker"]
    training_mode = config["training"].get("training_mode", "mimo")
    horizon       = config["training"]["forecast_horizon"]

    # MIMO: output_dim = forecast_horizon
    # autoreg: output_dim = 1
    output_dim = horizon if training_mode == "mimo" else 1
    config["training"]["forecast_horizon"] = output_dim \
        if training_mode == "autoreg" else horizon
    # keep horizon correct in config for dataset construction
    config["training"]["forecast_horizon"] = horizon

    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Model   : {args.model.upper()}")
    print(f"  Ticker  : {ticker}")
    print(f"  Mode    : {training_mode.upper()}")
    print(f"  Horizon : {horizon} days")
    print(f"  Out dim : {output_dim} neuron(s)")
    print(f"  Device  : {device}")
    print(f"{'='*60}")

    # ── 1. Data
    loaders    = build_dataloaders(config, ticker=ticker, horizon=horizon)
    input_dim  = len(loaders.feature_cols)
    n_features = input_dim

    # ── 2. Build model with correct output_dim
    # Temporarily override forecast_horizon in config for build_model()
    config["training"]["forecast_horizon"] = output_dim
    model = build_model(args.model, config, input_dim=input_dim)
    # Restore for DataLoaders consistency
    config["training"]["forecast_horizon"] = horizon

    print(f"\n  Params  : {model.count_parameters():,}")
    print(f"  Summary : {model.model_summary()}")

    # ── 3. Train
    # Trainer uses DataLoaders whose y shape is (batch, horizon)
    # For MIMO: pred=(batch, horizon) vs y=(batch, horizon) — full vector loss
    # For autoreg: pred=(batch, 1) vs y=(batch, horizon) — only y[:,0] matters
    # The Trainer's MSE loss handles this correctly:
    #   MIMO:   criterion(pred, y)          both (batch, H)
    #   autoreg:criterion(pred, y[:, :1])   both (batch, 1)
    out_dir = os.path.join(config["artifacts"]["base_dir"], args.model)
    os.makedirs(out_dir, exist_ok=True)

    trainer = Trainer(
        model        = model,
        train_loader = loaders.train,
        val_loader   = loaders.val,
        config       = config,
        arch         = args.model,
        device       = device,
        training_mode= training_mode,
    )
    train_result = trainer.run()

    # ── 4. Evaluate on test set
    best_ckpt = torch.load(train_result["checkpoint_path"], map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])

    # restore horizon in config for evaluator
    config["training"]["forecast_horizon"] = horizon

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

    # ── 5. Patch metrics
    metrics["epochs_trained"] = train_result["epochs_trained"]
    metrics["best_val_loss"]  = round(train_result["best_val_loss"], 6)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── 6. Save scaler + meta
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    joblib.dump(loaders.scaler, scaler_path)
    print(f"\n  Scaler  → {scaler_path}")

    meta = {
        "arch":                   args.model,
        "ticker":                 ticker,
        "training_mode":          training_mode,
        "window_size":            config["data"]["window_size"],
        "forecast_horizon":       horizon,      # original horizon (dataset target width)
        "output_dim":             output_dim,   # actual model output neurons
        "features":               loaders.feature_cols,
        "all_features":           loaders.all_feature_cols,
        "boruta_used":            loaders.boruta_used,
        "close_col_idx":          loaders.close_col_idx,
        "original_close_col_idx": loaders.original_close_col_idx,
        "input_dim":              len(loaders.feature_cols),
        "hidden_size":            config["models"]["hidden_size"],
        "num_layers":             config["models"]["num_layers"],
        "dropout":                config["models"]["dropout"],
        "tau_constant":           config["models"]["lnn"]["tau_constant"],
        "ode_unfolds":            config["models"]["lnn"]["ode_unfolds"],
        "dt":                     config["models"]["lnn"]["dt"],
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta    → {meta_path}")

    print(f"\n{'='*60}")
    print(f"  Done: {args.model.upper()}  {training_mode.upper()}  {ticker}")
    print(f"  Artifacts: {out_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()