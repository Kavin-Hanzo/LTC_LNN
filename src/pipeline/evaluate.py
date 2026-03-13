"""
pipeline/evaluate.py

Evaluator — runs inference on the held-out test set and computes:
  - MAE   Mean Absolute Error          (in real $ prices)
  - RMSE  Root Mean Squared Error      (in real $ prices)
  - MAPE  Mean Absolute Percentage Error
  - R2    Coefficient of Determination (1.0 = perfect, <0 = worse than mean)

All metrics are on INVERSE-SCALED prices (real dollar values, not 0-1).

Outputs per arch:
  artifacts/{arch}/metrics.json       <- all metrics
  artifacts/{arch}/predictions.npz   <- truths_real + preds_real arrays
  artifacts/{arch}/trend_plot.png     <- predicted vs actual trend line chart
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


# ── metric functions ──────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² = 1 - SS_res / SS_tot
    1.0  = perfect prediction
    0.0  = model predicts the mean
    < 0  = worse than predicting the mean
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-9))


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_predictions(
    truths:   np.ndarray,
    preds:    np.ndarray,
    arch:     str,
    out_path: str,
    n_samples: int = 200,
):
    """
    Plot predicted vs actual Close prices on the test set.

    Strategy: for each test sample the model predicts `horizon` steps ahead.
    We take step-1 prediction from each sample (next-day forecast) and
    plot it against the actual next-day price. This gives a clean
    time-aligned trend line comparison.

    Args:
        truths:    (N_samples, horizon) — real prices
        preds:     (N_samples, horizon) — predicted prices
        arch:      model name for plot title
        out_path:  where to save the PNG
        n_samples: how many test samples to show (subsample if larger)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  [plot]  matplotlib not installed — skipping trend plot.")
        return

    # ── use step-1 predictions (most interpretable trend line)
    step = 0
    y_true_line = truths[:, step]
    y_pred_line = preds[:,  step]

    # subsample if too many points
    if len(y_true_line) > n_samples:
        idx = np.linspace(0, len(y_true_line) - 1, n_samples, dtype=int)
        y_true_line = y_true_line[idx]
        y_pred_line = y_pred_line[idx]

    x = np.arange(len(y_true_line))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#ffffff")

    # ── top panel: trend lines
    ax = axes[0]
    ax.set_facecolor("#ffffff")
    ax.plot(x, y_true_line, color="#00d4ff", linewidth=1.5,
            label="Actual Close",    alpha=0.9)
    ax.plot(x, y_pred_line, color="#ff6b35", linewidth=1.5,
            label="Predicted Close", alpha=0.9, linestyle="--")
    ax.fill_between(x, y_true_line, y_pred_line,
                    alpha=0.08, color="#ffffff")

    ax.set_title(f"{arch.upper()} — Predicted vs Actual Close Price (test set, step+1)",
                 color="black", fontsize=13, pad=12)
    ax.set_ylabel("Price ($)", color="black")
    ax.tick_params(colors="black")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#ffffff", labelcolor="black", fontsize=10)
    ax.grid(True, color="#333", linewidth=0.5, alpha=0.6)

    # ── bottom panel: residuals
    ax2 = axes[1]
    ax2.set_facecolor("#ffffff")
    residuals = y_pred_line - y_true_line
    colors = ["#00c851" if r >= 0 else "#ff4444" for r in residuals]
    ax2.bar(x, residuals, color=colors, alpha=0.7, width=1.0)
    ax2.axhline(0, color="#888", linewidth=0.8)
    ax2.set_ylabel("Residual ($)", color="black")
    ax2.set_xlabel("Test sample index", color="black")
    ax2.tick_params(colors="black")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")
    ax2.grid(True, color="#333", linewidth=0.5, alpha=0.4)

    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [plot]  trend plot saved -> {out_path}")


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:

    def __init__(
        self,
        model:          nn.Module,
        test_loader:    DataLoader,
        scaler:         MinMaxScaler,
        close_col_idx:  int,
        n_features:     int,
        arch:           str,
        config:         dict,
        device:         torch.device,
    ):
        self.model         = model.to(device)
        self.test_loader   = test_loader
        self.scaler        = scaler
        self.close_col_idx = close_col_idx
        self.n_features    = n_features
        self.arch          = arch
        self.config        = config
        self.device        = device

        self.out_dir = os.path.join(config["artifacts"]["base_dir"], arch)
        os.makedirs(self.out_dir, exist_ok=True)

    def run(self) -> dict:
        """
        Collect all test predictions, inverse-scale, compute metrics,
        save metrics.json + predictions.npz + trend_plot.png.

        Returns:
            dict with test_mae, test_rmse, test_mape, test_r2
        """
        self.model.eval()
        all_preds  = []
        all_truths = []

        with torch.no_grad():
            for X, y in self.test_loader:
                X    = X.to(self.device)
                pred = self.model(X)                   # (batch, horizon)
                all_preds.append(pred.cpu().numpy())
                all_truths.append(y.numpy())

        preds_scaled  = np.vstack(all_preds)           # (N_samples, horizon)
        truths_scaled = np.vstack(all_truths)

        # ── inverse scale -> real dollar prices
        preds_real  = inverse_scale_close(
            preds_scaled,  self.scaler, self.close_col_idx, self.n_features
        )
        truths_real = inverse_scale_close(
            truths_scaled, self.scaler, self.close_col_idx, self.n_features
        )

        # ── metrics (all on real $ prices)
        metrics = {
            "arch":           self.arch,
            "test_mae":       round(mae( truths_real, preds_real), 4),
            "test_rmse":      round(rmse(truths_real, preds_real), 4),
            "test_mape":      round(mape(truths_real, preds_real), 4),
            "test_r2":        round(r2(  truths_real, preds_real), 4),
            "epochs_trained": None,   # patched in by train.py
            "best_val_loss":  None,   # patched in by train.py
        }

        # ── save metrics.json
        metrics_path = os.path.join(self.out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # ── save raw prediction arrays (useful for custom analysis)
        npz_path = os.path.join(self.out_dir, "predictions.npz")
        np.savez(npz_path, truths=truths_real, preds=preds_real)

        # ── trend plot
        plot_path = os.path.join(self.out_dir, "trend_plot.png")
        plot_predictions(truths_real, preds_real, self.arch, plot_path)

        print(f"\n[Evaluator]  arch={self.arch.upper()}")
        print(f"  MAE   = ${metrics['test_mae']}")
        print(f"  RMSE  = ${metrics['test_rmse']}")
        print(f"  MAPE  = {metrics['test_mape']}%")
        print(f"  R²    = {metrics['test_r2']}")
        print(f"  Saved -> {metrics_path}")

        return metrics
