# optimized_ohlcv_gpu_parallel.py
# Optimized version of quick_test_model_ohlcv.py for GPU and parallel processing.
# Includes DataParallel for multi-GPU, num_workers for data loading, pin_memory, and AMP for mixed precision.
# Usage example:
#   python optimized_ohlcv_gpu_parallel.py --stage train --feature-set ohlcv --variant all --mode short

import argparse
import os
import time
from copy import deepcopy
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from collections import defaultdict
from config import CFG
from data.fetcher import fetch_all, aligned_closes
from data.indicators import compute_all, FEATURE_COLS as IND_FEATURE_COLS
from data.scaler import chronological_split, scale_splits
from data.dataset import build_datasets
from models import build_model, SUPPORTED_MODELS
from models.trainer import train, load_checkpoint
from evaluation.metrics import compute, aggregate, improvement_table
from experiments import e1_clustering, e3_generalization, e4_interpretability
from visualization import plotter
from vectors.stock2vec import build as build_vectors, zero_vector

_VARIANTS = ["baseline", "onehot", "stock2vec"]
_SECTOR_IDX = {
    "AAPL": 0, "MSFT": 0, "GOOGL": 0, "META": 0,
    "IBM":  1,
    "NVDA": 2,
}
_N_SECTORS = 3
_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume", "close_return"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimized quick test model training/evaluation on OHLCV-scaled features plus stock vector variants with GPU and parallel processing."
    )
    parser.add_argument(
        "--stage",
        choices=["train", "evaluate", "experiment", "compare", "all"],
        default="all",
        help="Which quick-test stage to run."
    )
    parser.add_argument(
        "--feature-set",
        choices=["ohlcv", "indicators"],
        default="ohlcv",
        help="Which feature set to use for model input."
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=CFG.data.tickers,
        help="Ticker list used for training / evaluation."
    )
    parser.add_argument(
        "--years",
        type=int,
        default=CFG.data.history_years[0],
        help="History window in years for raw data and vector construction."
    )
    parser.add_argument(
        "--raw-dir",
        default=CFG.data.raw_dir,
        help="Raw OHLCV directory."
    )
    parser.add_argument(
        "--vectors-dir",
        default=CFG.vectors.vectors_dir,
        help="Directory to write Stock2Vec CSV outputs."
    )
    parser.add_argument(
        "--models-dir",
        default=CFG.training.models_dir,
        help="Directory for model checkpoints."
    )
    parser.add_argument(
        "--plots-dir",
        default=CFG.experiments.plots_dir,
        help="Directory for experiment plots."
    )
    parser.add_argument(
        "--results-dir",
        default=CFG.experiments.results_dir,
        help="Directory for experiment results."
    )
    parser.add_argument(
        "--label",
        default="optimized_ohlcv",
        help="Suffix for saved outputs and experiment labels."
    )
    parser.add_argument(
        "--mode",
        choices=list(CFG.training.modes.keys()),
        default="short",
        help="Train/eval mode: short or long."
    )
    parser.add_argument(
        "--variant",
        nargs="*",
        choices=_VARIANTS + ["all"],
        default=["stock2vec"],
        help="Which model identity variant(s) to run."
    )
    parser.add_argument(
        "--experiment",
        nargs="*",
        choices=["e1", "e3", "e4", "all"],
        default=["all"],
        help="Which experiment(s) to run."
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to use for evaluation."
    )
    parser.add_argument(
        "--arch",
        choices=SUPPORTED_MODELS,
        default=CFG.model.arch,
        help="Model architecture to use."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CFG.training.epochs,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CFG.training.batch_size,
        help="Batch size for training and evaluation."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=CFG.training.lr,
        help="Learning rate for training."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=CFG.training.weight_decay,
        help="Weight decay for training optimizer."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=CFG.training.patience,
        help="Early stopping patience."
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Lookback length for dataset construction. If omitted, inferred from --mode."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Horizon length for dataset construction. If omitted, inferred from --mode."
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=CFG.training.train_ratio,
        help="Train-split ratio."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=CFG.training.val_ratio,
        help="Validation-split ratio."
    )
    parser.add_argument(
        "--scaler-method",
        choices=["standard", "minmax"],
        default=CFG.scaler.method,
        help="Scaler method."
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refetch raw data from Yahoo Finance."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers for DataLoader parallel processing."
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use Automatic Mixed Precision (AMP) for faster training."
    )
    return parser.parse_args()


def get_variant_list(args):
    if "all" in args.variant:
        return _VARIANTS.copy()
    return sorted(set(args.variant), key=_VARIANTS.index)


def onehot_map(tickers):
    return {
        t: np.eye(_N_SECTORS, dtype=np.float32)[_SECTOR_IDX.get(t, 0)]
        for t in tickers
    }


def build_ohlcv_features(raw):
    processed = {}
    for ticker, df in raw.items():
        df_feat = df.copy()
        df_feat["close_return"] = df_feat["Close"].pct_change().clip(-0.30, 0.30)
        df_feat = df_feat.dropna(subset=["close_return"]).copy()
        processed[ticker] = (df_feat, _OHLCV_COLS)
    return processed


def prepare_data(args):
    CFG.make_dirs()
    raw = fetch_all(args.tickers, args.years, args.raw_dir,
                    force_refresh=args.force_refresh)

    if args.feature_set == "ohlcv":
        processed = build_ohlcv_features(raw)
        print(f"[Optimized OHLCV] using {len(_OHLCV_COLS)} scaled OHLCV features")
    else:
        processed = compute_all(raw, CFG.indicators)
        print(f"[Optimized Indicators] using {len(IND_FEATURE_COLS)} technical features")

    aligned = aligned_closes(raw)
    n_tr = int(len(aligned) * args.train_ratio)
    vectors, pca, variance = build_vectors(
        aligned.iloc[:n_tr],
        n_components=CFG.vectors.n_components,
        save_dir=args.vectors_dir,
        label=args.label,
    )

    scaled_splits = {}
    for ticker, (df, cols) in processed.items():
        tr, va, te = chronological_split(df, args.train_ratio, args.val_ratio)
        scaled_splits[ticker] = scale_splits(tr, va, te, cols, args.scaler_method)

    # Extract scalers for saving with model checkpoints
    scalers = {t: scaled_splits[t][3] for t in scaled_splits}

    return raw, processed, aligned, vectors, scaled_splits, scalers


def make_model(args, identity_dim, use_identity):
    cfg = deepcopy(CFG.model)
    cfg.arch = args.arch
    cfg.use_identity = use_identity
    cfg.identity_dim = identity_dim
    model = build_model(cfg, len(_OHLCV_COLS) if args.feature_set == "ohlcv" else len(IND_FEATURE_COLS), args.horizon)
    # Wrap with DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"[Optimized] Using DataParallel with {torch.cuda.device_count()} GPUs")
    return model


def train_variant(model, train_ds, val_ds, args, variant, label, scalers=None, device=None):
    checkpoint = os.path.join(args.models_dir, f"{variant}_{label}.pt")
    cfg_train = deepcopy(CFG.training)
    cfg_train.batch_size = args.batch_size
    cfg_train.epochs = args.epochs
    cfg_train.lr = args.lr
    cfg_train.weight_decay = args.weight_decay
    cfg_train.patience = args.patience

    # Optimized DataLoader
    pin_memory = device.type == 'cuda' if device else False
    tr_loader = torch.utils.data.DataLoader(train_ds,
                                            batch_size=cfg_train.batch_size,
                                            shuffle=True, num_workers=args.num_workers,
                                            pin_memory=pin_memory,
                                            persistent_workers=args.num_workers > 0)
    va_loader = torch.utils.data.DataLoader(val_ds,
                                            batch_size=cfg_train.batch_size * 2,
                                            shuffle=False, num_workers=args.num_workers,
                                            pin_memory=pin_memory,
                                            persistent_workers=args.num_workers > 0)

    # Override trainer to use optimized loaders and AMP
    history = train_optimized(model, tr_loader, va_loader, cfg_train, device, checkpoint, use_amp=args.use_amp)
    
    # Save scalers as pickle files for inference
    if scalers is not None:
        os.makedirs(args.models_dir, exist_ok=True)
        for ticker, scaler in scalers.items():
            scaler_path = os.path.join(args.models_dir,
                                       f"scaler_{ticker}_{variant}_{label}.pkl")
            scaler.save(scaler_path)
            print(f"  [Saved scaler] {scaler_path}")
        # Keep a copy inside the checkpoint too, for convenience
        ckpt = torch.load(checkpoint, map_location="cpu")
        ckpt["scalers"] = scalers
        torch.save(ckpt, checkpoint)
        print(f"  [Saved scalers] {checkpoint}")
    
    return history, checkpoint


def train_optimized(model, tr_loader, va_loader, cfg_training, device, checkpoint_path, use_amp=False):
    """
    Optimized training loop with AMP support.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg_training.lr,
                                 weight_decay=cfg_training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    stopper = EarlyStopping(patience=cfg_training.patience)
    history = {"train_loss": [], "val_loss": []}

    scaler = torch.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    print(f"\n  [Train Optimized] device={device}  epochs={cfg_training.epochs}  "
          f"patience={cfg_training.patience}  AMP={use_amp}  workers={tr_loader.num_workers}")
    print(f"  {'Ep':>5}  {'Train':>10}  {'Val':>10}  {'LR':>9}")
    print("  " + "─" * 40)

    t0 = time.time()
    for ep in range(1, cfg_training.epochs + 1):
        tr = _one_epoch_optimized(model, tr_loader, optimizer, criterion, device, True, scaler)
        va = _one_epoch_optimized(model, va_loader, optimizer, criterion, device, False, scaler)
        scheduler.step(va)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  {ep:5d}  {tr:10.6f}  {va:10.6f}  {lr_now:9.2e}")

        if stopper.step(va, model):
            print(f"  Early stopping at epoch {ep}")
            stopper.restore(model)
            break

    if checkpoint_path:
        torch.save({
            "model_state": model.state_dict(),
            "config": cfg_training,
            "history": history,
        }, checkpoint_path)
        print(f"  [Saved] {checkpoint_path}")

    print(".1f")
    return history


def _one_epoch_optimized(model, loader, optimizer, criterion, device, train, scaler=None):
    model.train(train)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            idn = batch["identity"].to(device, non_blocking=True) if hasattr(model, 'use_identity') and model.use_identity else None
            y = batch["y"].to(device, non_blocking=True)
            if train:
                optimizer.zero_grad()
                if scaler:
                    with torch.amp.autocast(device_type=device.type):
                        pred = model(x, idn)
                        loss = criterion(pred, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(x, idn)
                    loss = criterion(pred, y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            else:
                if scaler:
                    with torch.amp.autocast(device_type=device.type):
                        pred = model(x, idn)
                        loss = criterion(pred, y)
                else:
                    pred = model(x, idn)
                    loss = criterion(pred, y)
            total += loss.item()
            n += 1
    return total / max(n, 1)


class EarlyStopping:
    def __init__(self, patience: int = 15):
        self.patience = patience
        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - 1e-6:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


def evaluate_variant(model, dataset, args, device):
    print("\n[Optimized OHLCV] Running evaluation")
    model.to(device)
    # Optimized DataLoader for evaluation
    pin_memory = device.type == 'cuda'
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size * 2,
                                         shuffle=False, num_workers=args.num_workers,
                                         pin_memory=pin_memory,
                                         persistent_workers=args.num_workers > 0)
    per_stock, agg, preds = evaluate_optimized(model, loader, device)
    return per_stock, agg, preds


def evaluate_optimized(model, loader, device):
    """Optimized evaluation loop with parallel DataLoader."""
    model.eval()
    buckets = defaultdict(lambda: {"y_true": [], "y_pred": [], "close_refs": []})

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            idn = batch["identity"].to(device, non_blocking=True) if hasattr(model, 'use_identity') and model.use_identity else None
            y_true = batch["y"].cpu().numpy()
            close_ref = batch["close_ref"].cpu().numpy()
            tickers = batch["ticker"]

            y_pred = model(x, idn).cpu().numpy()
            for i, t in enumerate(tickers):
                buckets[t]["y_true"].append(y_true[i])
                buckets[t]["y_pred"].append(y_pred[i])
                buckets[t]["close_refs"].append(close_ref[i])

    predictions = {
        t: {
            "y_true": np.stack(d["y_true"]),
            "y_pred": np.stack(d["y_pred"]),
            "close_refs": np.array(d["close_refs"]),
        }
        for t, d in buckets.items()
    }

    per_stock = {
        t: compute(arrays["y_true"], arrays["y_pred"], arrays["close_refs"], ticker=t)
        for t, arrays in predictions.items()
    }
    agg = aggregate(per_stock)

    return per_stock, agg, predictions


def compare_variants(args, prepared, device):
    raw, processed, aligned, vectors, scaled_splits, scalers = prepared
    variants = get_variant_list(args)
    print(f"Comparing variants: {variants}")
    results = {}
    all_per_stock = {}
    all_preds = {}

    for variant in variants:
        print(f"\n--- {variant.upper()} ---")
        if variant == "baseline":
            id_map = {t: zero_vector(CFG.vectors.n_components) for t in processed.keys()}
            identity_dim = CFG.vectors.n_components
            use_identity = False
        elif variant == "onehot":
            id_map = onehot_map(processed.keys())
            identity_dim = _N_SECTORS
            use_identity = True
        else:
            id_map = vectors
            identity_dim = CFG.vectors.n_components
            use_identity = True

        ret_idx = (_OHLCV_COLS.index("close_return") if args.feature_set == "ohlcv"
                   else IND_FEATURE_COLS.index("close_return"))
        tr_ds, va_ds, te_ds = build_datasets(
            processed, id_map, scaled_splits,
            list(processed.keys()), args.lookback, args.horizon, ret_idx,
        )

        model = make_model(args, identity_dim, use_identity)
        history, ckpt = train_variant(model, tr_ds, va_ds, args, variant, args.label, scalers, device)
        per_stock, agg, preds = evaluate_variant(model, te_ds, args, device)

        results[variant] = {
            "history": history,
            "checkpoint": ckpt,
            "metrics": agg,
            "per_stock": per_stock,
        }
        all_per_stock[variant] = per_stock
        all_preds[variant] = preds

        mlbl = f"{args.mode}_{args.label}"
        for ticker, arrays in preds.items():
            plotter.cumulative_return(
                arrays["y_true"], arrays["y_pred"], arrays["close_refs"],
                ticker, variant, args.plots_dir, mlbl,
            )

        os.makedirs(args.results_dir, exist_ok=True)
        metrics_df = pd.DataFrame(per_stock).T
        metrics_df.to_csv(os.path.join(args.results_dir, f"metrics_{variant}_{args.label}.csv"))
        pd.Series(agg).to_csv(os.path.join(args.results_dir, f"agg_metrics_{variant}_{args.label}.csv"), header=False)

    if "baseline" in all_per_stock and "stock2vec" in all_per_stock:
        imp_df = improvement_table(all_per_stock["baseline"], all_per_stock["stock2vec"])
        imp_csv = os.path.join(args.results_dir, f"improvement_{args.label}.csv")
        imp_df.to_csv(imp_csv)
        print(f"\n[Saved] {imp_csv}")

    print("\nVariant comparison complete.")
    return results


def run_experiments(args, prepared):
    raw, processed, aligned, vectors, scaled_splits = prepared
    chosen = args.experiment
    if not chosen:
        print("No experiment selected. Use --experiment e1/e3/e4 or all.")
        return {}

    if "all" in chosen:
        chosen = ["e1", "e3", "e4"]
    chosen = sorted(set(chosen))
    print(f"Running experiments: {chosen}")
    outputs = {}

    if "e1" in chosen:
        outputs["e1"] = e1_clustering.run(vectors, args.plots_dir, args.results_dir, args.label)

    if "e3" in chosen:
        loo_results = leave_one_out(processed, vectors, scaled_splits, args)
        outputs["e3"] = e3_generalization.run(loo_results, vectors, {}, args.plots_dir, args.results_dir, args.label)

    if "e4" in chosen:
        outputs["e4"] = e4_interpretability.run(vectors, aligned.iloc[:int(len(aligned) * args.train_ratio)],
                                                 args.plots_dir, args.results_dir, args.label)

    return outputs


def leave_one_out(processed, vectors, scaled_splits, args):
    results = []
    tickers = list(processed.keys())
    for held in tickers:
        train_tickers = [t for t in tickers if t != held]
        if len(train_tickers) < 2:
            continue
        print(f"\n[E3 Optimized] leave-one-out held-out: {held}")

        ret_idx = (_OHLCV_COLS.index("close_return") if args.feature_set == "ohlcv"
                   else IND_FEATURE_COLS.index("close_return"))
        tr_ds, va_ds, _ = build_datasets(processed, vectors, scaled_splits,
                                        train_tickers, args.lookback, args.horizon, ret_idx)
        _, _, te_ds = build_datasets(processed, vectors, scaled_splits,
                                     [held], args.lookback, args.horizon, ret_idx)

        model = make_model(args, CFG.vectors.n_components, True)
        cfg_train = deepcopy(CFG.training)
        cfg_train.batch_size = args.batch_size
        cfg_train.epochs = args.epochs
        cfg_train.lr = args.lr
        cfg_train.weight_decay = args.weight_decay
        cfg_train.patience = args.patience
        # Optimized training for LOO
        pin_memory = CFG.resolve_device().type == 'cuda'
        tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=cfg_train.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
        va_loader = torch.utils.data.DataLoader(va_ds, batch_size=cfg_train.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
        train_optimized(model, tr_loader, va_loader, cfg_train, CFG.resolve_device(), checkpoint_path=None, use_amp=args.use_amp)

        per_stock, _, _ = evaluate_variant(model, te_ds, args, CFG.resolve_device())
        if held in per_stock:
            metrics = per_stock[held]
            results.append({
                "ticker": held,
                "zero_shot_rmse": metrics["rmse_ret"],
                "zero_shot_rawdev": metrics["raw_dev_mean"],
            })
    return results


def main():
    args = parse_args()
    if args.lookback is None or args.horizon is None:
        mode_cfg = CFG.training.modes[args.mode]
        if args.lookback is None:
            args.lookback = mode_cfg["lookback"]
        if args.horizon is None:
            args.horizon = mode_cfg["horizon"]

    CFG.make_dirs()
    device = CFG.resolve_device()
    print(f"[Optimized] Using device: {device}")
    if device.type == 'cuda':
        print(f"[Optimized] CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")
    variants = get_variant_list(args)
    prepared = None
    if args.stage in ["train", "evaluate", "compare", "experiment", "all"]:
        print("Preparing data for optimized OHLCV quick tests...")
        prepared = prepare_data(args)

    if args.stage in ["train", "compare", "all"]:
        print(f"Training variant(s): {variants}")
        compare_variants(args, prepared, device)

    if args.stage == "evaluate":
        raw, processed, aligned, vectors, scaled_splits, scalers = prepared
        for variant in variants:
            if variant == "baseline":
                id_map = {t: zero_vector(CFG.vectors.n_components) for t in processed.keys()}
                identity_dim = CFG.vectors.n_components
                use_identity = False
            elif variant == "onehot":
                id_map = onehot_map(processed.keys())
                identity_dim = _N_SECTORS
                use_identity = True
            else:
                id_map = vectors
                identity_dim = CFG.vectors.n_components
                use_identity = True

            ret_idx = (_OHLCV_COLS.index("close_return") if args.feature_set == "ohlcv"
                       else IND_FEATURE_COLS.index("close_return"))
            tr_ds, va_ds, te_ds = build_datasets(processed, id_map, scaled_splits,
                                                list(processed.keys()), args.lookback, args.horizon,
                                                ret_idx)
            ds = {"train": tr_ds, "val": va_ds, "test": te_ds}[args.split]

            model = make_model(args, identity_dim, use_identity)
            model_path = os.path.join(args.models_dir, f"{variant}_{args.label}.pt")
            if os.path.exists(model_path):
                ckpt = load_checkpoint(model, model_path)
                print(f"Loaded checkpoint {model_path}")
                if "scalers" in ckpt:
                    print(f"  Scalers available for inference: {list(ckpt['scalers'].keys())}")
                evaluate_variant(model, ds, args, device)
            else:
                print(f"Checkpoint not found for {variant}: {model_path}. Train first or set --stage train.")

    if args.stage in ["experiment", "all"]:
        outputs = run_experiments(args, prepared)
        if outputs:
            print("Experiment outputs:")
            pprint(outputs)

    print("\n✅ optimized_ohlcv_gpu_parallel complete.")


if __name__ == "__main__":
    main()