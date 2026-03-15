"""
pipeline/evaluate.py

Evaluator — runs inference on the held-out test set and computes:
  - MAE   Mean Absolute Error          (in real $ prices)
  - RMSE  Root Mean Squared Error      (in real $ prices)
  - MAPE  Mean Absolute Percentage Error
  - R2    Coefficient of Determination

Autoregressive mode (output_dim=1):
  Model predicts only the next single step from each window.
  Metrics are computed on next-step predictions vs actual next-step prices.
  This is the cleanest evaluation — no error compounding, directly comparable
  across all 4 architectures.

Outputs per arch:
  artifacts/{arch}/metrics.json       <- all metrics + eval_mode
  artifacts/{arch}/predictions.npz   <- truths (N,1) + preds (N,1)
  artifacts/{arch}/trend_plot.png     <- predicted vs actual next-step prices
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from pipeline.data_pipeline import inverse_scale_close


# -- metric functions ----------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-9))


# -- Visualization -------------------------------------------------------------

def plot_predictions(
    truths:    np.ndarray,
    preds:     np.ndarray,
    arch:      str,
    out_path:  str,
    n_samples: int = 200,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  [plot]  matplotlib not installed -- skipping trend plot.")
        return

    y_true = truths.reshape(-1)
    y_pred = preds.reshape(-1)

    if len(y_true) > n_samples:
        idx    = np.linspace(0, len(y_true) - 1, n_samples, dtype=int)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    x = np.arange(len(y_true))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0f0f0f")

    ax = axes[0]
    ax.set_facecolor("#1a1a2e")
    ax.plot(x, y_true, color="#00d4ff", lw=1.8, label="Actual",    alpha=0.95)
    ax.plot(x, y_pred, color="#ff6b35", lw=1.8, label="Predicted", alpha=0.9,
            linestyle="--")
    ax.fill_between(x, y_true, y_pred, alpha=0.07, color="#ffffff")
    ax.set_title(
        f"{arch.upper()} -- Predicted vs Actual Close Price (next-step, test set)",
        color="white", fontsize=13, pad=10
    )
    ax.set_ylabel("Price ($)", color="white")
    ax.tick_params(colors="white")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)
    ax.grid(True, color="#333", lw=0.5, alpha=0.6)

    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    residuals  = y_pred - y_true
    bar_colors = ["#00c851" if r >= 0 else "#ff4444" for r in residuals]
    ax2.bar(x, residuals, color=bar_colors, alpha=0.75, width=1.0)
    ax2.axhline(0, color="#888", lw=0.8)
    ax2.set_ylabel("Residual ($)", color="white")
    ax2.set_xlabel("Test sample index", color="white")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#444")
    ax2.grid(True, color="#333", lw=0.5, alpha=0.4)

    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [plot]  trend plot saved -> {out_path}")


# -- Evaluator -----------------------------------------------------------------

class Evaluator:

    def __init__(
        self,
        model:                  nn.Module,
        test_loader:            DataLoader,
        scaler:                 MinMaxScaler,
        close_col_idx:          int,
        n_features:             int,
        arch:                   str,
        config:                 dict,
        device:                 torch.device,
        original_close_col_idx: int = None,   # pre-Boruta Close index for inverse scaling
    ):
        self.model                  = model.to(device)
        self.test_loader            = test_loader
        self.scaler                 = scaler
        self.close_col_idx          = close_col_idx
        self.n_features             = n_features
        self.arch                   = arch
        self.config                 = config
        self.device                 = device
        # if original_close_col_idx not provided, fall back to close_col_idx
        self.original_close_col_idx = original_close_col_idx if original_close_col_idx is not None \
                                      else close_col_idx

        self.out_dir = os.path.join(config["artifacts"]["base_dir"], arch)
        os.makedirs(self.out_dir, exist_ok=True)

    def run(self) -> dict:
        """
        Collect next-step test predictions, inverse-scale, compute metrics.

        With output_dim=1 (autoregressive mode):
          pred[i]  = model prediction for next day given window[i]
          truth[i] = actual next-day Close price

        Returns:
            dict with test_mae, test_rmse, test_mape, test_r2
        """
        self.model.eval()
        all_preds  = []
        all_truths = []

        with torch.no_grad():
            for X, y in self.test_loader:
                X    = X.to(self.device)
                pred = self.model(X)                        # (batch, 1)
                all_preds.append(pred[:, 0].cpu().numpy())  # next-step pred
                all_truths.append(y[:, 0].numpy())          # next-step truth

        preds_scaled  = np.concatenate(all_preds)           # (N,)
        truths_scaled = np.concatenate(all_truths)          # (N,)

        # inverse scale to real $ prices
        # scaler was fitted on original features — always use original_close_col_idx
        preds_real = inverse_scale_close(
            preds_scaled.reshape(-1, 1),
            self.scaler,
            self.close_col_idx,
            original_close_idx = self.original_close_col_idx,
        ).reshape(-1)
        truths_real = inverse_scale_close(
            truths_scaled.reshape(-1, 1),
            self.scaler,
            self.close_col_idx,
            original_close_idx = self.original_close_col_idx,
        ).reshape(-1)

        # metrics on real $ prices
        metrics = {
            "arch":           self.arch,
            "eval_mode":      "single_step",
            "test_mae":       round(mae( truths_real, preds_real), 4),
            "test_rmse":      round(rmse(truths_real, preds_real), 4),
            "test_mape":      round(mape(truths_real, preds_real), 4),
            "test_r2":        round(r2(  truths_real, preds_real), 4),
            "epochs_trained": None,   # patched in by train.py
            "best_val_loss":  None,   # patched in by train.py
        }

        metrics_path = os.path.join(self.out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        npz_path = os.path.join(self.out_dir, "predictions.npz")
        np.savez(
            npz_path,
            truths = truths_real.reshape(-1, 1),
            preds  = preds_real.reshape(-1, 1),
        )

        plot_path = os.path.join(self.out_dir, "trend_plot.png")
        plot_predictions(
            truths_real.reshape(-1, 1),
            preds_real.reshape(-1, 1),
            self.arch, plot_path
        )

        print(f"\n[Evaluator]  arch={self.arch.upper()}  (single-step eval)")
        print(f"  MAE   = ${metrics['test_mae']}")
        print(f"  RMSE  = ${metrics['test_rmse']}")
        print(f"  MAPE  = {metrics['test_mape']}%")
        print(f"  R2    = {metrics['test_r2']}")
        print(f"  Saved -> {metrics_path}")

        return metrics