"""
visualize.py  —  Standalone visualization tool for trained models.

Loads predictions.npz saved by evaluate.py and produces:
  1. Trend plot     — predicted vs actual close price (step+1 or all horizons)
  2. Horizon plot   — error vs forecast step (how accuracy degrades over time)
  3. Scatter plot   — predicted vs actual scatter with perfect-fit diagonal
  4. Compare plot   — overlay all trained models on the same chart

Usage:
    # Plot a single model
    python visualize.py --model lstm

    # Plot specific horizon step (0 = next day, 6 = 7th day ahead)
    python visualize.py --model lstm --step 0

    # Plot error across all horizon steps
    python visualize.py --model lstm --horizon-error

    # Compare all trained models on one chart
    python visualize.py --compare

    # Save to custom path
    python visualize.py --model lstm --out my_plot.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# ── matplotlib setup ──────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.gridspec as gridspec
except ImportError:
    print("ERROR: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

from models import SUPPORTED_MODELS

# ── dark theme constants ──────────────────────────────────────────────────────
BG       = "#ffffff"
PANEL_BG = "#ffffff"
ACTUAL   = "#00d4ff"
COLORS   = ["#ff6b35", "#00c851", "#ffd700", "#c77dff"]
GRID_C   = "#333333"
TEXT_C   = "black"


# ── data loading ──────────────────────────────────────────────────────────────

def load_predictions(arch: str, base_dir: str = "artifacts") -> tuple:
    """Load truths and preds arrays saved by Evaluator."""
    path = os.path.join(base_dir, arch, "predictions.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No predictions found for '{arch}' at {path}.\n"
            f"Run:  python train.py --model {arch}  first."
        )
    data = np.load(path)
    return data["truths"], data["preds"]   # both (N_samples, horizon)


# ── Plot 1: Trend line ────────────────────────────────────────────────────────

def plot_trend(
    truths: np.ndarray,
    preds:  np.ndarray,
    arch:   str,
    step:   int = 0,
    n_samples: int = 200,
    out_path: str = None,
):
    """
    Predicted vs actual Close price trend line for a given forecast step.
    Bottom panel shows residuals as a bar chart.
    """
    y_true = truths[:, step]
    y_pred = preds[:,  step]

    if len(y_true) > n_samples:
        idx    = np.linspace(0, len(y_true) - 1, n_samples, dtype=int)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    x = np.arange(len(y_true))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor(BG)

    # trend panel
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    ax.plot(x, y_true, color=ACTUAL,    lw=1.8, label="Actual",    alpha=0.95)
    ax.plot(x, y_pred, color=COLORS[0], lw=1.8, label="Predicted", alpha=0.9,
            linestyle="--")
    ax.fill_between(x, y_true, y_pred, alpha=0.07, color=TEXT_C)
    ax.set_title(
        f"{arch.upper()} — Predicted vs Actual  |  step+{step+1} forecast",
        color=TEXT_C, fontsize=13, pad=10
    )
    ax.set_ylabel("Price ($)", color=TEXT_C)
    ax.tick_params(colors=TEXT_C)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_C, fontsize=10)
    ax.grid(True, color=GRID_C, lw=0.5, alpha=0.6)

    # residuals panel
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    residuals = y_pred - y_true
    bar_colors = ["#00c851" if r >= 0 else "#ff4444" for r in residuals]
    ax2.bar(x, residuals, color=bar_colors, alpha=0.75, width=1.0)
    ax2.axhline(0, color="#888", lw=0.8)
    ax2.set_ylabel("Residual ($)", color=TEXT_C)
    ax2.set_xlabel("Test sample index", color=TEXT_C)
    ax2.tick_params(colors=TEXT_C)
    for sp in ax2.spines.values(): sp.set_edgecolor("#444")
    ax2.grid(True, color=GRID_C, lw=0.5, alpha=0.4)

    plt.tight_layout(pad=1.5)
    out = out_path or f"artifacts/{arch}/trend_plot_step{step}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved -> {out}")


# ── Plot 2: Horizon error ─────────────────────────────────────────────────────

def plot_horizon_error(
    truths: np.ndarray,
    preds:  np.ndarray,
    arch:   str,
    out_path: str = None,
):
    """
    RMSE and MAE per forecast step — shows how accuracy degrades
    as we forecast further into the future.
    """
    horizon = truths.shape[1]
    rmse_per_step = []
    mae_per_step  = []

    for s in range(horizon):
        diff = truths[:, s] - preds[:, s]
        rmse_per_step.append(float(np.sqrt(np.mean(diff ** 2))))
        mae_per_step.append(float(np.mean(np.abs(diff))))

    steps = np.arange(1, horizon + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL_BG)

    ax.plot(steps, rmse_per_step, color=COLORS[0], lw=2.0, marker="o",
            markersize=4, label="RMSE")
    ax.plot(steps, mae_per_step,  color=ACTUAL,    lw=2.0, marker="s",
            markersize=4, label="MAE")
    ax.fill_between(steps, mae_per_step, rmse_per_step,
                    alpha=0.08, color=TEXT_C)

    ax.set_title(f"{arch.upper()} — Forecast Error by Horizon Step",
                 color=TEXT_C, fontsize=13, pad=10)
    ax.set_xlabel("Forecast step (days ahead)", color=TEXT_C)
    ax.set_ylabel("Error ($)", color=TEXT_C)
    ax.tick_params(colors=TEXT_C)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1f"))
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_C, fontsize=10)
    ax.grid(True, color=GRID_C, lw=0.5, alpha=0.6)

    plt.tight_layout(pad=1.5)
    out = out_path or f"artifacts/{arch}/horizon_error.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved -> {out}")


# ── Plot 3: Scatter ───────────────────────────────────────────────────────────

def plot_scatter(
    truths: np.ndarray,
    preds:  np.ndarray,
    arch:   str,
    step:   int = 0,
    out_path: str = None,
):
    """
    Predicted vs actual scatter plot with perfect-fit diagonal.
    Tight cluster along diagonal = good model.
    """
    y_true = truths[:, step].ravel()
    y_pred = preds[:,  step].ravel()

    # subsample for readability
    if len(y_true) > 500:
        idx    = np.random.choice(len(y_true), 500, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    lo = min(y_true.min(), y_pred.min()) * 0.98
    hi = max(y_true.max(), y_pred.max()) * 1.02

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL_BG)

    ax.scatter(y_true, y_pred, color=COLORS[0], alpha=0.4, s=18, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], color=ACTUAL, lw=1.5, linestyle="--",
            label="Perfect fit")

    ax.set_title(f"{arch.upper()} — Actual vs Predicted Scatter  (step+{step+1})",
                 color=TEXT_C, fontsize=12, pad=10)
    ax.set_xlabel("Actual Price ($)",    color=TEXT_C)
    ax.set_ylabel("Predicted Price ($)", color=TEXT_C)
    ax.tick_params(colors=TEXT_C)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_C, fontsize=10)
    ax.grid(True, color=GRID_C, lw=0.5, alpha=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    plt.tight_layout(pad=1.5)
    out = out_path or f"artifacts/{arch}/scatter_step{step}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved -> {out}")


# ── Plot 4: Multi-model comparison ───────────────────────────────────────────

def plot_compare(
    base_dir:  str = "artifacts",
    step:      int = 0,
    n_samples: int = 200,
    out_path:  str = None,
):
    """
    Overlay predicted trend lines for all trained models on the same chart.
    Uses the actual prices from the first available model as ground truth.
    """
    available = []
    for arch in SUPPORTED_MODELS:
        path = os.path.join(base_dir, arch, "predictions.npz")
        if os.path.exists(path):
            available.append(arch)

    if not available:
        print("No trained models found. Run training first.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL_BG)

    truths_ref = None

    for i, arch in enumerate(available):
        truths, preds = load_predictions(arch, base_dir)
        y_pred = preds[:, step]

        # subsample consistently
        n = min(len(y_pred), n_samples)
        idx = np.linspace(0, len(y_pred) - 1, n, dtype=int)
        y_pred = y_pred[idx]

        if truths_ref is None:
            truths_ref = truths[:, step][idx]
            x = np.arange(len(truths_ref))
            ax.plot(x, truths_ref, color=ACTUAL, lw=2.0,
                    label="Actual", alpha=0.95, zorder=5)

        ax.plot(x[:len(y_pred)], y_pred, color=COLORS[i % len(COLORS)],
                lw=1.5, label=arch.upper(), alpha=0.85, linestyle="--")

    ax.set_title(f"Model Comparison — Predicted vs Actual  |  step+{step+1}",
                 color=TEXT_C, fontsize=13, pad=10)
    ax.set_xlabel("Test sample index", color=TEXT_C)
    ax.set_ylabel("Price ($)",         color=TEXT_C)
    ax.tick_params(colors=TEXT_C)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_C, fontsize=10)
    ax.grid(True, color=GRID_C, lw=0.5, alpha=0.5)

    plt.tight_layout(pad=1.5)
    out = out_path or os.path.join(base_dir, "compare_plot.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved -> {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize model predictions.")
    parser.add_argument("--model",         type=str, choices=SUPPORTED_MODELS,
                        help="Model to visualize")
    parser.add_argument("--step",          type=int, default=0,
                        help="Forecast step index to plot (0 = next day)")
    parser.add_argument("--horizon-error", action="store_true",
                        help="Plot RMSE/MAE across all horizon steps")
    parser.add_argument("--scatter",       action="store_true",
                        help="Plot predicted vs actual scatter")
    parser.add_argument("--compare",       action="store_true",
                        help="Overlay all trained models on one chart")
    parser.add_argument("--all",           action="store_true",
                        help="Generate all plots for the given model")
    parser.add_argument("--out",           type=str, default=None,
                        help="Custom output path for the plot PNG")
    parser.add_argument("--base-dir",      type=str, default="artifacts")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.compare:
        print("\n[visualize]  compare plot ...")
        plot_compare(base_dir=args.base_dir, step=args.step, out_path=args.out)
        return

    if not args.model:
        print("ERROR: --model required (or use --compare)")
        print(f"  Choices: {SUPPORTED_MODELS}")
        sys.exit(1)

    print(f"\n[visualize]  arch={args.model.upper()}  step={args.step}")
    truths, preds = load_predictions(args.model, args.base_dir)
    print(f"  predictions shape: truths={truths.shape}  preds={preds.shape}")

    if args.all or (not args.horizon_error and not args.scatter):
        print("  -> trend plot ...")
        plot_trend(truths, preds, args.model, step=args.step,
                   out_path=args.out if not args.all else None)

    if args.all or args.horizon_error:
        print("  -> horizon error plot ...")
        plot_horizon_error(truths, preds, args.model)

    if args.all or args.scatter:
        print("  -> scatter plot ...")
        plot_scatter(truths, preds, args.model, step=args.step)

    print()


if __name__ == "__main__":
    main()
