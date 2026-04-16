# experiments/e2_ablation.py
# Experiment 2 — prove the identity vector improves prediction accuracy.
#
# Compares three model variants:
#   Baseline   no company context
#   OneHot     one-hot sector encoding (naive context)
#   Stock2Vec  4D PCA identity vector (our approach)
#
# Produces:
#   Grouped RMSE / RawDev / DirAcc bar charts per stock × model
#   Loss curves (Baseline vs Stock2Vec)
#   Predicted vs actual price overlay (3 representative stocks)
#   Per-horizon raw deviation growth chart
#   Metric table CSV + improvement table CSV

import os
from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation.metrics import metrics_table, improvement_table, returns_to_prices


def run(results:               Dict[str, Dict],
        predictions_per_model: Dict[str, Dict],
        histories:             Dict[str, Dict],
        plots_dir:             str,
        results_dir:           str,
        label:                 str = "",
        showcase:              Optional[List[str]] = None) -> Optional[pd.DataFrame]:

    os.makedirs(plots_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    sns.set_style("whitegrid")
    sfx     = f"_{label}" if label else ""
    tickers = sorted(next(iter(results.values())).keys())
    show    = showcase or tickers[:3]

    print(f"\n{'═'*55}")
    print(f"  E2 — Ablation Study  [{label}]")
    print(f"{'═'*55}")

    # ── Bar charts ────────────────────────────────────────────────
    _bar_chart(results, "rmse_ret",      "RMSE (return space)",
               f"E2 — RMSE per Stock × Model  [{label}]",
               plots_dir, f"e2_rmse{sfx}.png", lower=True)
    _bar_chart(results, "raw_dev_mean",  "Mean Raw Deviation ($)",
               f"E2 — Raw Price Deviation  [{label}]",
               plots_dir, f"e2_rawdev{sfx}.png", lower=True)
    _bar_chart(results, "dir_acc",       "Direction Accuracy (%)",
               f"E2 — Direction Accuracy  [{label}]",
               plots_dir, f"e2_diracc{sfx}.png", lower=False)

    # ── Loss curves ───────────────────────────────────────────────
    if histories:
        _loss_curves(histories, plots_dir, sfx)

    # ── Predicted vs actual ───────────────────────────────────────
    _pred_vs_actual(predictions_per_model, show, plots_dir, sfx)

    # ── Per-horizon deviation ─────────────────────────────────────
    for t in show:
        _horizon_deviation(predictions_per_model, t, plots_dir, sfx)

    # ── Tables ────────────────────────────────────────────────────
    mt = metrics_table(results)
    mt.to_csv(os.path.join(results_dir, f"e2_metrics{sfx}.csv"))
    print("\n  Metrics:\n" + mt.to_string())

    imp = None
    if "Baseline" in results and "Stock2Vec" in results:
        imp = improvement_table(results["Baseline"], results["Stock2Vec"])
        imp.to_csv(os.path.join(results_dir, f"e2_improvement{sfx}.csv"))
        print("\n  Improvement (Baseline → Stock2Vec):\n" + imp.to_string())

    print(f"\n  [E2] plots → {plots_dir}\n")
    return imp


# ── Private helpers ───────────────────────────────────────────────

_PALETTE = {"Baseline": "#4E79A7", "OneHot": "#F28E2B", "Stock2Vec": "#59A14F"}


def _bar_chart(results, metric, ylabel, title, plots_dir, fname, lower):
    models  = list(results.keys())
    tickers = sorted(next(iter(results.values())).keys())
    x, w    = np.arange(len(tickers)), 0.22
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, m in enumerate(models):
        vals  = [results[m][t][metric] for t in tickers]
        color = _PALETTE.get(m, "#888888")
        bars  = ax.bar(x + i * w, vals, w, label=m,
                       color=color, edgecolor="white", alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.3f}", ha="center", fontsize=7.5)

    ax.set_xticks(x + w)
    ax.set_xticklabels(tickers, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.99, 0.97, "↓ lower is better" if lower else "↑ higher is better",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=8, color="gray")
    _save(fig, plots_dir, fname)


def _loss_curves(histories, plots_dir, sfx):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"E2 — Training Loss Curves", fontsize=13, fontweight="bold")
    for ax, split in zip(axes, ["train_loss", "val_loss"]):
        for m, hist in histories.items():
            if split in hist and hist[split]:
                ax.plot(hist[split], label=m,
                        color=_PALETTE.get(m, "#888"), lw=2)
        ax.set_title(split.replace("_", " ").title())
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    _save(fig, plots_dir, f"e2_loss_curves{sfx}.png")


def _pred_vs_actual(ppm, tickers_show, plots_dir, sfx):
    fig, axes = plt.subplots(len(tickers_show), 1,
                             figsize=(15, 5 * len(tickers_show)))
    if len(tickers_show) == 1:
        axes = [axes]
    fig.suptitle("E2 — Predicted vs Actual Price", fontsize=13, fontweight="bold")

    for ax, t in zip(axes, tickers_show):
        ref_m = next(iter(ppm))
        if t not in ppm[ref_m]:
            continue
        N    = min(200, len(ppm[ref_m][t]["y_true"]))
        refs = ppm[ref_m][t]["close_refs"][-N:]
        act  = refs * (1 + ppm[ref_m][t]["y_true"][-N:, 0])
        ax.plot(act, color="black", lw=2, label="Actual", zorder=5)

        for m, per_stock in ppm.items():
            if t not in per_stock:
                continue
            pred = per_stock[t]["close_refs"][-N:] * (1 + per_stock[t]["y_pred"][-N:, 0])
            ax.plot(pred, color=_PALETTE.get(m, "#888"),
                    lw=1.5, ls="--", alpha=0.85, label=m)

        ax.set_title(f"{t} — next-step price (last {N} samples)",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Price ($)"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Sample index")
    _save(fig, plots_dir, f"e2_pred_vs_actual{sfx}.png")


def _horizon_deviation(ppm, ticker, plots_dir, sfx):
    fig, ax = plt.subplots(figsize=(10, 5))
    for m, per_stock in ppm.items():
        if ticker not in per_stock:
            continue
        a   = per_stock[ticker]
        pt  = returns_to_prices(a["y_true"], a["close_refs"])
        pp  = returns_to_prices(a["y_pred"], a["close_refs"])
        dev = np.abs(pt - pp).mean(axis=0)
        ax.plot(range(1, len(dev) + 1), dev, label=m,
                color=_PALETTE.get(m, "#888"), lw=2, marker="o", ms=5)
    ax.set_xlabel("Forecast Horizon (days ahead)")
    ax.set_ylabel("Mean Abs Price Deviation ($)")
    ax.set_title(f"E2 — Raw Deviation Growth  |  {ticker}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    _save(fig, plots_dir, f"e2_rawdev_horizon_{ticker}{sfx}.png")


def _save(fig, plots_dir, fname, dpi=130):
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, fname), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [E2] → {fname}")
