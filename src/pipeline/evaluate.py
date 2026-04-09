"""
pipeline/evaluate.py

Evaluator — supports two training modes:

  MIMO (Direct Multi-Horizon):
    model output shape: (batch, forecast_horizon)
    y shape:            (batch, forecast_horizon)
    Loss was computed across the full horizon vector during training.
    Metrics here are computed per-step AND averaged across all steps.
    Saves predictions.npz shape: (N_samples, forecast_horizon)

  Autoreg (Single-step autoregressive):
    model output shape: (batch, 1)
    y shape:            (batch, forecast_horizon) — only step 0 used
    Metrics computed on next-step only.
    Saves predictions.npz shape: (N_samples, 1)

All metrics are on INVERSE-SCALED prices (real dollar values).

Outputs per arch:
  artifacts/{arch}/metrics.json
  artifacts/{arch}/predictions.npz
  artifacts/{arch}/trend_plot.png
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


# ── Metric functions ──────────────────────────────────────────────────────────

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


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_predictions(
    truths:    np.ndarray,
    preds:     np.ndarray,
    arch:      str,
    out_path:  str,
    mode:      str = "mimo",
    max_samples: int = None,
):
    """
    Plot predicted vs actual Close prices across ALL test samples.

    For MIMO:   plots step-1 (next day) prediction vs actual.
    For autoreg: plots the single predicted step vs actual.

    Args:
        max_samples: if set, subsample to this many points evenly.
                     Default None = plot all samples.
                     Only set this for very large datasets (e.g. minute-level).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  [plot]  matplotlib not installed -- skipping.")
        return

    y_true = truths[:, 0].reshape(-1)
    y_pred = preds[:,  0].reshape(-1)

    total_samples = len(y_true)

    # only subsample if explicitly requested AND dataset exceeds the cap
    if max_samples and total_samples > max_samples:
        idx    = np.linspace(0, total_samples - 1, max_samples, dtype=int)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        sample_note = f" (subsampled {max_samples}/{total_samples})"
    else:
        sample_note = f" ({total_samples} samples)"

    x = np.arange(len(y_true))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#ffffff")

    ax = axes[0]
    ax.set_facecolor("#ffffff")
    ax.plot(x, y_true, color="#00d4ff", lw=1.8, label="Actual",    alpha=0.95)
    ax.plot(x, y_pred, color="#ff6b35", lw=1.8, label="Predicted", alpha=0.9,
            linestyle="--")
    ax.fill_between(x, y_true, y_pred, alpha=0.07, color="#ffffff")
    mode_label = "MIMO step+1" if mode == "mimo" else "autoreg step+1"
    ax.set_title(
        f"{arch.upper()} — Predicted vs Actual Close Price  "
        f"({mode_label}, test set{sample_note})",
        color="black", fontsize=13, pad=10
    )
    ax.set_ylabel("Price ($)", color="black")
    ax.tick_params(colors="black")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.legend(facecolor="#ffffff", labelcolor="black", fontsize=10)
    ax.grid(True, color="#333", lw=0.5, alpha=0.6)

    ax2 = axes[1]
    ax2.set_facecolor("#ffffff")
    residuals  = y_pred - y_true
    bar_colors = ["#00c851" if r >= 0 else "#ff4444" for r in residuals]
    ax2.bar(x, residuals, color=bar_colors, alpha=0.75, width=1.0)
    ax2.axhline(0, color="#888", lw=0.8)
    ax2.set_ylabel("Residual ($)", color="black")
    ax2.set_xlabel("Test sample index", color="black")
    ax2.tick_params(colors="black")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#444")
    ax2.grid(True, color="#333", lw=0.5, alpha=0.4)

    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [plot]  saved -> {out_path}")


# ── Evaluator ─────────────────────────────────────────────────────────────────

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
        original_close_col_idx: int = None,
    ):
        self.model                  = model.to(device)
        self.test_loader            = test_loader
        self.scaler                 = scaler
        self.close_col_idx          = close_col_idx
        self.n_features             = n_features
        self.arch                   = arch
        self.config                 = config
        self.device                 = device
        self.original_close_col_idx = original_close_col_idx \
                                      if original_close_col_idx is not None \
                                      else close_col_idx

        self.mode    = config.get("training", {}).get("training_mode", "mimo")
        self.out_dir = os.path.join(config["artifacts"]["base_dir"], arch)
        os.makedirs(self.out_dir, exist_ok=True)

    # ── MIMO evaluation ───────────────────────────────────────────────────────

    def _run_mimo(self) -> dict:
        """
        MIMO: model outputs (batch, H) — full horizon vector.
        Metrics computed:
          - per-step: MAE/RMSE at each of the H steps
          - overall:  averaged across all steps and all samples
        """
        self.model.eval()
        all_preds  = []
        all_truths = []

        with torch.no_grad():
            for X, y in self.test_loader:
                X    = X.to(self.device)
                pred = self.model(X)                     # (batch, H)
                all_preds.append(pred.cpu().numpy())
                all_truths.append(y.numpy())

        preds_scaled  = np.vstack(all_preds)             # (N, H)
        truths_scaled = np.vstack(all_truths)            # (N, H)

        H = preds_scaled.shape[1]

        # inverse scale each horizon step independently
        preds_real  = inverse_scale_close(
            preds_scaled,
            self.scaler,
            self.close_col_idx,
            original_close_idx = self.original_close_col_idx,
        )                                                 # (N, H)
        truths_real = inverse_scale_close(
            truths_scaled,
            self.scaler,
            self.close_col_idx,
            original_close_idx = self.original_close_col_idx,
        )                                                 # (N, H)

        # ── overall metrics (flattened across all steps)
        overall = {
            "test_mae":  round(mae( truths_real, preds_real), 4),
            "test_rmse": round(rmse(truths_real, preds_real), 4),
            "test_mape": round(mape(truths_real, preds_real), 4),
            "test_r2":   round(r2(  truths_real, preds_real), 4),
        }

        # ── per-step metrics (step_1 = next day, step_H = last horizon day)
        per_step = {}
        for h in range(H):
            per_step[f"step_{h+1}"] = {
                "mae":  round(mae( truths_real[:, h], preds_real[:, h]), 4),
                "rmse": round(rmse(truths_real[:, h], preds_real[:, h]), 4),
                "mape": round(mape(truths_real[:, h], preds_real[:, h]), 4),
                "r2":   round(r2(  truths_real[:, h], preds_real[:, h]), 4),
            }

        return preds_real, truths_real, overall, per_step

    # ── Autoreg evaluation ────────────────────────────────────────────────────

    def _run_autoreg(self) -> dict:
        """
        Autoreg: model outputs (batch, 1) — single next step.
        Metrics computed on next-step prediction only.
        """
        self.model.eval()
        all_preds  = []
        all_truths = []

        with torch.no_grad():
            for X, y in self.test_loader:
                X    = X.to(self.device)
                pred = self.model(X)                     # (batch, 1)
                all_preds.append(pred[:, 0].cpu().numpy())
                all_truths.append(y[:, 0].numpy())

        preds_scaled  = np.concatenate(all_preds)        # (N,)
        truths_scaled = np.concatenate(all_truths)       # (N,)

        preds_real = inverse_scale_close(
            preds_scaled.reshape(-1, 1),
            self.scaler, self.close_col_idx,
            original_close_idx = self.original_close_col_idx,
        ).reshape(-1)
        truths_real = inverse_scale_close(
            truths_scaled.reshape(-1, 1),
            self.scaler, self.close_col_idx,
            original_close_idx = self.original_close_col_idx,
        ).reshape(-1)

        overall = {
            "test_mae":  round(mae( truths_real, preds_real), 4),
            "test_rmse": round(rmse(truths_real, preds_real), 4),
            "test_mape": round(mape(truths_real, preds_real), 4),
            "test_r2":   round(r2(  truths_real, preds_real), 4),
        }

        # reshape to (N, 1) for consistent downstream shape
        return (
            preds_real.reshape(-1, 1),
            truths_real.reshape(-1, 1),
            overall,
            {},   # no per-step for single-step model
        )

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Dispatch to MIMO or autoreg evaluation based on config.training_mode.
        Saves metrics.json, predictions.npz, trend_plot.png.

        Returns:
            dict with test_mae, test_rmse, test_mape, test_r2,
                  per_step_metrics (MIMO only),
                  eval_mode
        """
        if self.mode == "mimo":
            preds_real, truths_real, overall, per_step = self._run_mimo()
            eval_mode = "mimo"
        else:
            preds_real, truths_real, overall, per_step = self._run_autoreg()
            eval_mode = "autoreg_single_step"

        metrics = {
            "arch":             self.arch,
            "eval_mode":        eval_mode,
            "epochs_trained":   None,   # patched in by train.py
            "best_val_loss":    None,   # patched in by train.py
            **overall,
        }
        if per_step:
            metrics["per_step_metrics"] = per_step

        # save metrics.json
        metrics_path = os.path.join(self.out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # save predictions.npz  (N, H) or (N, 1)
        npz_path = os.path.join(self.out_dir, "predictions.npz")
        np.savez(npz_path, truths=truths_real, preds=preds_real)

        # trend plot
        plot_path = os.path.join(self.out_dir, "trend_plot.png")
        plot_predictions(truths_real, preds_real, self.arch, plot_path, mode=self.mode)

        print(f"\n[Evaluator]  arch={self.arch.upper()}  mode={eval_mode}")
        print(f"  MAE   = ${metrics['test_mae']}")
        print(f"  RMSE  = ${metrics['test_rmse']}")
        print(f"  MAPE  = {metrics['test_mape']}%")
        print(f"  R2    = {metrics['test_r2']}")
        if per_step:
            print(f"  Per-step RMSE:")
            for step, m in per_step.items():
                print(f"    {step}: ${m['rmse']}")
        print(f"  Saved -> {metrics_path}")

        return metrics