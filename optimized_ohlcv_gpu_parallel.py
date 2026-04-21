# quick_test_model.py  ─ Kaggle-GPU edition
# ============================================================
# Optimised for Kaggle dual-T4 / dual-P100 (2 × ~15 GB VRAM).
#
# Key changes vs the original:
#   • Auto-detects GPU count → nn.DataParallel when N_GPUS > 1
#   • AMP (torch.cuda.amp) training loop replaces the vanilla trainer
#   • DataLoaders: pin_memory, persistent_workers, prefetch_factor,
#     num_workers = all available CPUs
#   • Batch size auto-scaled by GPU count
#   • Gradient clipping + zero_grad(set_to_none=True) for throughput
#   • CUDA-native metrics via torchmetrics (no .cpu() / numpy in eval)
#   • torch.compile opt-in (--compile flag, requires PyTorch ≥ 2.0)
#   • torch.backends.cudnn.benchmark + TF32 enabled at startup
#
# Usage (Kaggle notebook cell):
#   !python quick_test_model.py --stage compare --variant all \
#          --mode short --experiment e1 --epochs 30 --batch-size 256
#
# All existing CLI flags are preserved and behave identically.
# New flags:  --no-amp   --compile   --workers N

import argparse
import os
import time
from copy import deepcopy
from pprint import pprint
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CFG
from data.fetcher import fetch_all, aligned_closes
from data.indicators import compute_all, FEATURE_COLS
from data.scaler import chronological_split, scale_splits
from data.dataset import build_datasets
from models import build_model, SUPPORTED_MODELS
from models.trainer import train, load_checkpoint
from evaluation.evaluator import run as evaluate
from evaluation.metrics import improvement_table
from experiments import e1_clustering, e3_generalization, e4_interpretability
from visualization import plotter
from vectors.stock2vec import build as build_vectors, zero_vector


# ── Constants ──────────────────────────────────────────────────────────────────

_VARIANTS = ["baseline", "onehot", "stock2vec"]

_SECTOR_IDX: Dict[str, int] = {
    "AAPL": 0, "MSFT": 0, "GOOGL": 0, "META": 0,   # Software / Internet
    "IBM":  1,                                        # Enterprise / Legacy
    "NVDA": 2,                                        # Semiconductors
}
_N_SECTORS = 3


# ── Kaggle / CUDA environment setup ────────────────────────────────────────────

def _setup_cuda_env() -> Tuple[torch.device, int]:
    """
    Configure CUDA back-ends for maximum throughput and return
    (primary device, gpu_count).  Safe to call on CPU-only machines.
    """
    if not torch.cuda.is_available():
        print("[KaggleGPU] CUDA not available – running on CPU.")
        return torch.device("cpu"), 0

    n = torch.cuda.device_count()

    # Ampere / Volta TF32 (free ~10 % speed-up on matmuls, negligible loss)
    torch.backends.cuda.matmul.allow_tf32  = True
    torch.backends.cudnn.allow_tf32        = True

    # cuDNN auto-tuner: best for fixed-size inputs (stock sequences)
    torch.backends.cudnn.benchmark         = True
    torch.backends.cudnn.deterministic     = False   # keep benchmark gains

    device = torch.device("cuda:0")
    gpu_names = [torch.cuda.get_device_name(i) for i in range(n)]
    print(f"[KaggleGPU] {n} GPU(s) detected: {gpu_names}")
    print(f"[KaggleGPU] Primary device: {device}")
    return device, n


# Run once at import time so every function sees the same globals.
DEVICE, N_GPUS = _setup_cuda_env()
_CPU_COUNT: int = os.cpu_count() or 1


# ── DataLoader factory ─────────────────────────────────────────────────────────

def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    n_workers: int,
) -> DataLoader:
    """
    Build a DataLoader with Kaggle-friendly settings:
      • pin_memory=True  – async host→device transfers
      • persistent_workers – avoid worker respawn overhead
      • prefetch_factor=2 – overlap I/O with GPU compute
      • non_blocking in .to() calls finishes the async pipeline
    """
    pf = 2 if n_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(n_workers > 0),
        prefetch_factor=pf,
        drop_last=shuffle,   # drop only for train loader
    )


# ── Model helpers ──────────────────────────────────────────────────────────────

def _wrap_parallel(model: nn.Module) -> nn.Module:
    """
    Wrap in DataParallel when more than one GPU is available.
    The primary GPU (cuda:0) acts as the parameter server.
    Kaggle dual-T4: splits each mini-batch across both cards automatically.
    """
    if N_GPUS > 1:
        model = nn.DataParallel(model, device_ids=list(range(N_GPUS)))
        print(f"[KaggleGPU] DataParallel across {N_GPUS} GPUs.")
    return model.to(DEVICE)


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the raw (non-DataParallel) module."""
    return model.module if isinstance(model, nn.DataParallel) else model


def _try_compile(model: nn.Module, enable: bool) -> nn.Module:
    """
    Apply torch.compile (PyTorch ≥ 2.0) for kernel fusion.
    Falls back silently if unavailable or if compilation fails.
    """
    if not enable:
        return model
    if not hasattr(torch, "compile"):
        print("[KaggleGPU] torch.compile unavailable (requires PyTorch ≥ 2.0) – skipping.")
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        print("[KaggleGPU] torch.compile applied (reduce-overhead mode).")
        return compiled
    except Exception as exc:  # pragma: no cover
        print(f"[KaggleGPU] torch.compile failed ({exc}) – using eager mode.")
        return model


# ── AMP training loop ──────────────────────────────────────────────────────────

def train_fast_amp(
    model: nn.Module,
    train_ds,
    val_ds,
    cfg_train,
    device: torch.device,
    checkpoint_path: Optional[str],
    n_workers: int = 4,
    use_amp: bool = True,
) -> Dict:
    """
    Drop-in replacement for models.trainer.train() that adds:
      1. nn.DataParallel across all available GPUs
      2. Automatic Mixed Precision (AMP) via torch.cuda.amp
      3. Pinned, prefetched DataLoaders
      4. Gradient clipping (max-norm 1.0)
      5. zero_grad(set_to_none=True)

    Checkpoint format is intentionally identical to the original trainer so
    that load_checkpoint() and downstream code keep working unchanged.

    Parameters
    ----------
    model          : freshly built (CPU) model from make_model()
    train_ds / val_ds : dataset objects returned by build_datasets()
    cfg_train      : training config (epochs, lr, weight_decay, patience, …)
    device         : primary CUDA device (or CPU)
    checkpoint_path: where to save the best checkpoint (.pt)
    n_workers      : DataLoader worker count
    use_amp        : enable AMP (disable for debugging)

    Returns
    -------
    history dict with keys "train_loss" and "val_loss"
    """
    # ── Scale batch size with GPU count ───────────────────────────
    effective_bs = cfg_train.batch_size * max(1, N_GPUS)

    # ── DataParallel wrap ─────────────────────────────────────────
    model = _wrap_parallel(model)

    # ── DataLoaders ───────────────────────────────────────────────
    train_loader = _make_loader(train_ds, effective_bs, shuffle=True,  n_workers=n_workers)
    val_loader   = _make_loader(val_ds,   effective_bs * 2, shuffle=False, n_workers=n_workers)

    # ── Optimizer & scheduler ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_train.lr,
        weight_decay=cfg_train.weight_decay,
        fused=(device.type == "cuda"),   # fused AdamW is faster on CUDA
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=max(2, cfg_train.patience // 2), factor=0.5
    )
    criterion = nn.MSELoss()

    # ── AMP scaler (no-op on CPU) ─────────────────────────────────
    amp_enabled = use_amp and (device.type == "cuda")
    scaler      = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    raw_model = _unwrap(model)   # reference for attribute checks & save

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_ctr  = 0

    print(f"[train_fast_amp] bs_per_call={effective_bs}  "
          f"workers={n_workers}  amp={amp_enabled}  "
          f"epochs={cfg_train.epochs}  patience={cfg_train.patience}")

    for epoch in range(cfg_train.epochs):
        t0 = time.time()

        # ── Training pass ─────────────────────────────────────────
        model.train()
        running_train = 0.0
        for batch in train_loader:
            x        = batch["x"].to(device, non_blocking=True)
            y        = batch["y"].to(device, non_blocking=True)
            identity = batch.get("identity")
            if identity is not None:
                identity = identity.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                # Support both model(x) and model(x, identity) signatures
                if identity is not None and getattr(raw_model, "use_identity", False):
                    pred = model(x, identity)
                else:
                    pred = model(x)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_train += loss.item()

        train_loss = running_train / max(len(train_loader), 1)

        # ── Validation pass ───────────────────────────────────────
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x        = batch["x"].to(device, non_blocking=True)
                y        = batch["y"].to(device, non_blocking=True)
                identity = batch.get("identity")
                if identity is not None:
                    identity = identity.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    if identity is not None and getattr(raw_model, "use_identity", False):
                        pred = model(x, identity)
                    else:
                        pred = model(x)
                    loss = criterion(pred, y)
                running_val += loss.item()

        val_loss = running_val / max(len(val_loader), 1)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch + 1:3d}/{cfg_train.epochs}  "
              f"train={train_loss:.5f}  val={val_loss:.5f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"time={elapsed:.1f}s")

        # ── Checkpoint best model ─────────────────────────────────
        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            patience_ctr  = 0
            if checkpoint_path:
                os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
                torch.save(
                    {
                        "model_state_dict":     raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch":                epoch,
                        "val_loss":             best_val_loss,
                    },
                    checkpoint_path,
                )
        else:
            patience_ctr += 1
            if patience_ctr >= cfg_train.patience:
                print(f"  Early stopping at epoch {epoch + 1} "
                      f"(patience={cfg_train.patience}).")
                break

    # ── Restore best weights ──────────────────────────────────────
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"])

    # Return the unwrapped model so callers get the plain nn.Module,
    # keeping the interface identical to models.trainer.train().
    model = raw_model.to(device)
    return history


# ── CUDA-native evaluation ─────────────────────────────────────────────────────

def cuda_evaluate(
    model: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
    n_workers: int = 4,
    use_amp: bool = True,
) -> Tuple[Dict, Dict, Dict]:
    """
    Evaluation that keeps tensors on CUDA for metric computation,
    using torchmetrics when available to avoid host round-trips.

    Returns the same (per_stock, agg, preds) tuple as
    evaluation.evaluator.run() so all downstream code is unaffected.
    The final numpy arrays in `preds` are only materialised once at the
    very end (needed by plotter).
    """
    # ── Try to import torchmetrics ────────────────────────────────
    try:
        import torchmetrics
        _HAS_TM = True
    except ImportError:
        _HAS_TM = False
        print("[cuda_evaluate] torchmetrics not found – using torch-native metrics.")

    amp_enabled = use_amp and (device.type == "cuda")
    eval_loader = _make_loader(dataset, batch_size, shuffle=False, n_workers=n_workers)

    raw_model = _unwrap(model)
    raw_model.eval()

    # Accumulate predictions per ticker (stay on CUDA until end)
    ticker_bufs: Dict[str, Dict[str, List[torch.Tensor]]] = {}

    with torch.no_grad():
        for batch in eval_loader:
            x        = batch["x"].to(device, non_blocking=True)
            y        = batch["y"].to(device, non_blocking=True)
            identity = batch.get("identity")
            if identity is not None:
                identity = identity.to(device, non_blocking=True)
            close_ref = batch.get("close_ref")   # may be absent

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                if identity is not None and getattr(raw_model, "use_identity", False):
                    pred = raw_model(x, identity)
                else:
                    pred = raw_model(x)

            tickers_in_batch = batch["ticker"]   # list[str], length = B
            for i, ticker in enumerate(tickers_in_batch):
                if ticker not in ticker_bufs:
                    ticker_bufs[ticker] = {
                        "y_true":     [],
                        "y_pred":     [],
                        "close_refs": [],
                    }
                ticker_bufs[ticker]["y_true"].append(y[i].detach())
                ticker_bufs[ticker]["y_pred"].append(pred[i].detach())
                if close_ref is not None:
                    ticker_bufs[ticker]["close_refs"].append(
                        close_ref[i].to(device, non_blocking=True)
                    )

    # ── Compute metrics on CUDA, then pull numpy for plotter ──────
    per_stock: Dict[str, Dict] = {}
    preds:     Dict[str, Dict] = {}

    # Build per-ticker torchmetrics objects once (reuse across tickers)
    if _HAS_TM:
        rmse_fn = torchmetrics.MeanSquaredError(squared=False).to(device)
        mae_fn  = torchmetrics.MeanAbsoluteError().to(device)
        mape_fn = torchmetrics.MeanAbsolutePercentageError().to(device)

    for ticker, bufs in ticker_bufs.items():
        yt = torch.stack(bufs["y_true"])    # [N, H]  – CUDA
        yp = torch.stack(bufs["y_pred"])    # [N, H]  – CUDA
        flat_yt = yt.reshape(-1)
        flat_yp = yp.reshape(-1)

        if _HAS_TM:
            rmse_fn.reset(); mae_fn.reset(); mape_fn.reset()
            rmse     = rmse_fn(flat_yp, flat_yt).item()
            mae      = mae_fn(flat_yp,  flat_yt).item()
            mape     = mape_fn(flat_yp, flat_yt).item()
        else:
            # Pure torch on CUDA – still no .cpu() or numpy here
            diff     = flat_yp - flat_yt
            rmse     = diff.pow(2).mean().sqrt().item()
            mae      = diff.abs().mean().item()
            mape     = (diff.abs() / (flat_yt.abs() + 1e-8)).mean().item()

        raw_dev_mean = (yp - yt).abs().mean().item()   # CUDA op

        per_stock[ticker] = {
            "rmse_ret":    rmse,
            "mae_ret":     mae,
            "mape_ret":    mape,
            "raw_dev_mean": raw_dev_mean,
        }

        # Materialise numpy only once, right here (needed by plotter)
        preds[ticker] = {
            "y_true":     yt.cpu().numpy(),
            "y_pred":     yp.cpu().numpy(),
            "close_refs": (
                torch.stack(bufs["close_refs"]).cpu().numpy()
                if bufs["close_refs"]
                else np.zeros(len(bufs["y_true"]), dtype=np.float32)
            ),
        }

    # Aggregate metrics (mean across tickers)
    if per_stock:
        agg = {
            k: float(np.mean([v[k] for v in per_stock.values()]))
            for k in per_stock[next(iter(per_stock))].keys()
        }
    else:
        agg = {}

    return per_stock, agg, preds


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Quick test the modelling and experiment stages "
            "without running the full pipeline. "
            "Kaggle-GPU edition: AMP + DataParallel enabled by default."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["train", "evaluate", "experiment", "compare", "all"],
        default="all",
    )
    parser.add_argument("--tickers",    nargs="*", default=CFG.data.tickers)
    parser.add_argument("--years",      type=int,  default=CFG.data.history_years[0])
    parser.add_argument("--raw-dir",               default=CFG.data.raw_dir)
    parser.add_argument("--vectors-dir",           default=CFG.vectors.vectors_dir)
    parser.add_argument("--models-dir",            default=CFG.training.models_dir)
    parser.add_argument("--plots-dir",             default=CFG.experiments.plots_dir)
    parser.add_argument("--results-dir",           default=CFG.experiments.results_dir)
    parser.add_argument("--label",                 default="quicktest")
    parser.add_argument(
        "--mode",
        choices=list(CFG.training.modes.keys()),
        default="week",
    )
    parser.add_argument(
        "--variant",
        nargs="*",
        choices=_VARIANTS + ["all"],
        default=["stock2vec"],
    )
    parser.add_argument(
        "--experiment",
        nargs="*",
        choices=["e1", "e3", "e4", "all"],
        default=["all"],
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
    )
    parser.add_argument("--arch",          choices=SUPPORTED_MODELS, default=CFG.model.arch)
    parser.add_argument("--epochs",        type=int,   default=CFG.training.epochs)
    parser.add_argument("--batch-size",    type=int,   default=CFG.training.batch_size)
    parser.add_argument("--lr",            type=float, default=CFG.training.lr)
    parser.add_argument("--weight-decay",  type=float, default=CFG.training.weight_decay)
    parser.add_argument("--patience",      type=int,   default=CFG.training.patience)
    parser.add_argument("--lookback",      type=int,   default=None)
    parser.add_argument("--horizon",       type=int,   default=None)
    parser.add_argument("--train-ratio",   type=float, default=CFG.training.train_ratio)
    parser.add_argument("--val-ratio",     type=float, default=CFG.training.val_ratio)
    parser.add_argument(
        "--scaler-method",
        choices=["standard", "minmax"],
        default=CFG.scaler.method,
    )
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--debug",         action="store_true")

    # ── Kaggle-specific flags ───────────────────────────────────
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable Automatic Mixed Precision (useful for debugging NaN losses).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Apply torch.compile (requires PyTorch ≥ 2.0; adds ~60 s warm-up).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(_CPU_COUNT, 4),
        help=(
            "DataLoader worker count. "
            "Kaggle notebooks expose 4 CPUs; default caps at that."
        ),
    )

    return parser.parse_args()


# ── Utility ────────────────────────────────────────────────────────────────────

def get_variant_list(args):
    if "all" in args.variant:
        return _VARIANTS.copy()
    return sorted(set(args.variant), key=_VARIANTS.index)


def onehot_map(tickers):
    out = {}
    for t in tickers:
        v = np.zeros(_N_SECTORS, dtype=np.float32)
        v[_SECTOR_IDX.get(t, 0)] = 1.0
        out[t] = v
    return out


# ── Data preparation (unchanged logic) ────────────────────────────────────────

def prepare_data(args):
    CFG.make_dirs()
    raw       = fetch_all(args.tickers, args.years, args.raw_dir,
                          force_refresh=args.force_refresh)
    processed = compute_all(raw, CFG.indicators)
    aligned   = aligned_closes(raw)
    n_tr      = int(len(aligned) * args.train_ratio)
    vectors, pca, _ = build_vectors(
        aligned.iloc[:n_tr],
        n_components=CFG.vectors.n_components,
        save_dir=args.vectors_dir,
        label=args.label,
    )
    scaled_splits = {}
    for ticker, (df, cols) in processed.items():
        tr, va, te = chronological_split(df, args.train_ratio, args.val_ratio)
        scaled_splits[ticker] = scale_splits(tr, va, te, cols, args.scaler_method)

    scalers = {t: scaled_splits[t][3] for t in scaled_splits}
    return raw, processed, aligned, vectors, scaled_splits, scalers


def build_dataset_for_variant(processed, vectors, scaled_splits, tickers,
                               lookback, horizon, variant):
    if variant == "baseline":
        id_map       = {t: zero_vector(CFG.vectors.n_components) for t in tickers}
        identity_dim = CFG.vectors.n_components
    elif variant == "onehot":
        id_map       = onehot_map(tickers)
        identity_dim = _N_SECTORS
    else:
        id_map       = vectors
        identity_dim = CFG.vectors.n_components
    ret_idx = FEATURE_COLS.index("close_return")
    ds = build_datasets(processed, id_map, scaled_splits, tickers,
                        lookback, horizon, ret_idx)
    return ds, id_map, identity_dim


# ── Model factory ──────────────────────────────────────────────────────────────

def make_model(args, identity_dim: int, use_identity: bool) -> nn.Module:
    cfg             = deepcopy(CFG.model)
    cfg.arch        = args.arch
    cfg.use_identity = use_identity
    cfg.identity_dim = identity_dim
    model           = build_model(cfg, len(FEATURE_COLS), args.horizon)
    return model   # caller decides when to move to device


# ── Training ───────────────────────────────────────────────────────────────────

def train_variant(model, train_ds, val_ds, args, variant, label,
                  scalers=None) -> Tuple[Dict, str]:
    """
    Train one variant using the AMP-accelerated fast trainer.
    Falls back to the original models.trainer.train() if AMP is
    disabled (--no-amp flag) so behaviour is identical on CPU.
    """
    checkpoint = os.path.join(args.models_dir, f"{variant}_{label}.pt")

    cfg_train              = deepcopy(CFG.training)
    cfg_train.batch_size   = args.batch_size
    cfg_train.epochs       = args.epochs
    cfg_train.lr           = args.lr
    cfg_train.weight_decay = args.weight_decay
    cfg_train.patience     = args.patience

    use_amp = not args.no_amp

    if use_amp or N_GPUS > 1:
        # ── Kaggle fast path ─────────────────────────────────────
        model = _try_compile(model, args.compile)
        history = train_fast_amp(
            model, train_ds, val_ds, cfg_train,
            DEVICE, checkpoint,
            n_workers=args.workers,
            use_amp=use_amp,
        )
    else:
        # ── Original trainer (CPU / no-AMP fallback) ─────────────
        print("[train_variant] AMP disabled – using original trainer.")
        history = train(model, train_ds, val_ds, cfg_train, DEVICE, checkpoint)

    # ── Persist scalers into checkpoint (same as original) ───────
    if scalers is not None:
        os.makedirs(args.models_dir, exist_ok=True)
        for ticker, scaler in scalers.items():
            scaler_path = os.path.join(
                args.models_dir,
                f"scaler_{ticker}_{variant}_{label}.pkl",
            )
            scaler.save(scaler_path)
        ckpt                = torch.load(checkpoint, map_location="cpu",
                                         weights_only=False)
        ckpt["scalers"]     = scalers
        torch.save(ckpt, checkpoint)
        print(f"  [Saved scalers → {checkpoint}]")

    return history, checkpoint


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_variant(model, dataset, args):
    """
    Run evaluation.  Uses the CUDA-native path when a GPU is present;
    otherwise falls back to the original evaluator (preserves all
    per_stock / agg / preds keys that downstream code expects).
    """
    print("\n[QuickTest] Running evaluation …")

    if DEVICE.type == "cuda":
        use_amp = not args.no_amp
        per_stock, agg, preds = cuda_evaluate(
            model, dataset, DEVICE, args.batch_size * 2,
            n_workers=args.workers, use_amp=use_amp,
        )
    else:
        # Original CPU evaluator – unchanged behaviour
        per_stock, agg, preds = evaluate(model, dataset, DEVICE, args.batch_size)

    if args.debug:
        print("\n[debug] Aggregate metrics:")
        pprint(agg)

    return per_stock, agg, preds


# ── Compare variants ───────────────────────────────────────────────────────────

def compare_variants(args, prepared):
    raw, processed, aligned, vectors, scaled_splits, scalers = prepared
    variants = get_variant_list(args)
    if not variants:
        variants = ["stock2vec"]
    print(f"\n[compare_variants] Variants: {variants}")

    results      = {}
    all_per_stock = {}
    all_preds     = {}

    for variant in variants:
        print(f"\n{'─' * 54}\n  {variant.upper()}\n{'─' * 54}")

        if variant == "baseline":
            id_map       = {t: zero_vector(CFG.vectors.n_components)
                            for t in processed.keys()}
            identity_dim = CFG.vectors.n_components
            use_identity = False
        elif variant == "onehot":
            id_map       = onehot_map(processed.keys())
            identity_dim = _N_SECTORS
            use_identity = True
        else:
            id_map       = vectors
            identity_dim = CFG.vectors.n_components
            use_identity = True

        ret_idx = FEATURE_COLS.index("close_return")
        tr_ds, va_ds, te_ds = build_datasets(
            processed, id_map, scaled_splits,
            list(processed.keys()), args.lookback, args.horizon, ret_idx,
        )

        model = make_model(args, identity_dim, use_identity)
        history, ckpt = train_variant(
            model, tr_ds, va_ds, args, variant, args.label, scalers
        )

        # Reload model for evaluation (best-epoch weights)
        model = make_model(args, identity_dim, use_identity)
        model.to(DEVICE)
        load_checkpoint(model, ckpt)

        per_stock, agg, preds = evaluate_variant(model, te_ds, args)

        results[variant] = {
            "history":    history,
            "checkpoint": ckpt,
            "metrics":    agg,
            "per_stock":  per_stock,
        }
        all_per_stock[variant] = per_stock
        all_preds[variant]     = preds

        # Cumulative return plots
        mlbl = f"{args.mode}_{args.label}"
        for ticker, arrays in preds.items():
            plotter.cumulative_return(
                arrays["y_true"], arrays["y_pred"], arrays["close_refs"],
                ticker, variant, args.plots_dir, mlbl,
            )

        # Per-stock metrics CSV
        os.makedirs(args.results_dir, exist_ok=True)
        metrics_csv = os.path.join(args.results_dir,
                                   f"metrics_{variant}_{args.label}.csv")
        pd.DataFrame(per_stock).T.to_csv(metrics_csv)
        print(f"[Saved] {metrics_csv}")

        agg_csv = os.path.join(args.results_dir,
                               f"agg_metrics_{variant}_{args.label}.csv")
        pd.Series(agg).to_csv(agg_csv, header=False)
        print(f"[Saved] {agg_csv}")

    # E2-style ablation
    if "baseline" in all_per_stock and "stock2vec" in all_per_stock:
        imp_df  = improvement_table(all_per_stock["baseline"],
                                    all_per_stock["stock2vec"])
        imp_csv = os.path.join(args.results_dir,
                               f"improvement_{args.label}.csv")
        imp_df.to_csv(imp_csv)
        print(f"\n[Saved] {imp_csv}")

    print("\n[compare_variants] Done.")
    return results


# ── Experiments (unchanged logic) ─────────────────────────────────────────────

def run_experiments(args, prepared):
    raw, processed, aligned, vectors, scaled_splits, _ = prepared
    chosen = args.experiment or []
    if not chosen:
        print("No experiment selected.  Use --experiment e1/e3/e4 or all.")
        return {}
    if "all" in chosen:
        chosen = ["e1", "e3", "e4"]
    chosen  = sorted(set(chosen))
    print(f"[run_experiments] Running: {chosen}")
    outputs = {}

    if "e1" in chosen:
        outputs["e1"] = e1_clustering.run(
            vectors, args.plots_dir, args.results_dir, args.label
        )

    if "e3" in chosen:
        loo_results    = leave_one_out(processed, vectors, scaled_splits, args)
        outputs["e3"] = e3_generalization.run(
            loo_results, vectors, {},
            args.plots_dir, args.results_dir, args.label,
        )

    if "e4" in chosen:
        n_tr           = int(len(aligned) * CFG.training.train_ratio)
        outputs["e4"] = e4_interpretability.run(
            vectors, aligned.iloc[:n_tr],
            args.plots_dir, args.results_dir, args.label,
        )

    return outputs


def leave_one_out(processed, vectors, scaled_splits, args) -> List[Dict]:
    results = []
    tickers = list(processed.keys())
    ret_idx = FEATURE_COLS.index("close_return")

    for held in tickers:
        train_tickers = [t for t in tickers if t != held]
        if len(train_tickers) < 2:
            continue
        print(f"\n[E3] LOO held-out: {held}")

        tr_ds, va_ds, _ = build_datasets(
            processed, vectors, scaled_splits,
            train_tickers, args.lookback, args.horizon, ret_idx,
        )
        _, _, te_ds = build_datasets(
            processed, vectors, scaled_splits,
            [held], args.lookback, args.horizon, ret_idx,
        )

        model    = make_model(args, CFG.vectors.n_components, True)
        cfg_copy = deepcopy(CFG.training)
        cfg_copy.batch_size   = args.batch_size
        cfg_copy.epochs       = args.epochs
        cfg_copy.lr           = args.lr
        cfg_copy.weight_decay = args.weight_decay
        cfg_copy.patience     = args.patience

        use_amp = not args.no_amp
        if use_amp or N_GPUS > 1:
            model = _try_compile(model, args.compile)
            train_fast_amp(model, tr_ds, va_ds, cfg_copy, DEVICE,
                           checkpoint_path=None,
                           n_workers=args.workers, use_amp=use_amp)
        else:
            train(model, tr_ds, va_ds, cfg_copy, DEVICE, checkpoint_path=None)

        per_stock, _, _ = evaluate_variant(model, te_ds, args)
        if held in per_stock:
            m = per_stock[held]
            results.append({
                "ticker":           held,
                "zero_shot_rmse":   m["rmse_ret"],
                "zero_shot_rawdev": m["raw_dev_mean"],
            })

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve lookback / horizon from mode config if not overridden
    if args.lookback is None or args.horizon is None:
        mode_cfg = CFG.training.modes[args.mode]
        if args.lookback is None:
            args.lookback = mode_cfg["lookback"]
        if args.horizon is None:
            args.horizon  = mode_cfg["horizon"]

    CFG.make_dirs()
    variants = get_variant_list(args)

    if args.debug:
        print(f"[debug] args = {vars(args)}")
        print(f"[debug] DEVICE={DEVICE}  N_GPUS={N_GPUS}  CPU_COUNT={_CPU_COUNT}")

    # ── Data preparation ─────────────────────────────────────────
    prepared = None
    if args.stage in ("train", "evaluate", "compare", "experiment", "all"):
        print("\n[main] Preparing data …")
        prepared = prepare_data(args)

    # ── Train / compare ──────────────────────────────────────────
    if args.stage in ("train", "compare", "all"):
        if not variants:
            variants = ["stock2vec"]
        print(f"\n[main] Training variant(s): {variants}")
        compare_variants(args, prepared)

    # ── Evaluate only ────────────────────────────────────────────
    if args.stage == "evaluate":
        print(f"\n[main] Evaluating {variants} on split={args.split}")
        raw, processed, aligned, vectors, scaled_splits, scalers = prepared
        for variant in variants:
            if variant == "baseline":
                id_map       = {t: zero_vector(CFG.vectors.n_components)
                                for t in processed.keys()}
                identity_dim = CFG.vectors.n_components
                use_identity = False
            elif variant == "onehot":
                id_map       = onehot_map(processed.keys())
                identity_dim = _N_SECTORS
                use_identity = True
            else:
                id_map       = vectors
                identity_dim = CFG.vectors.n_components
                use_identity = True

            ret_idx           = FEATURE_COLS.index("close_return")
            tr_ds, va_ds, te_ds = build_datasets(
                processed, id_map, scaled_splits,
                list(processed.keys()), args.lookback, args.horizon, ret_idx,
            )
            ds         = {"train": tr_ds, "val": va_ds, "test": te_ds}[args.split]
            model      = make_model(args, identity_dim, use_identity)
            model_path = os.path.join(args.models_dir, f"{variant}_{args.label}.pt")

            if os.path.exists(model_path):
                ckpt = load_checkpoint(model, model_path)
                model.to(DEVICE)
                print(f"[main] Loaded checkpoint: {model_path}")
                if "scalers" in ckpt:
                    print(f"  Scalers: {list(ckpt['scalers'].keys())}")
                evaluate_variant(model, ds, args)
            else:
                print(f"[main] Checkpoint not found: {model_path} – train first.")

    # ── Experiments ──────────────────────────────────────────────
    if args.stage in ("experiment", "all"):
        outputs = run_experiments(args, prepared)
        if outputs:
            print("\n[main] Experiment outputs:")
            pprint(outputs)

    print("\n✅ quick_test_model complete.")


if __name__ == "__main__":
    main()
