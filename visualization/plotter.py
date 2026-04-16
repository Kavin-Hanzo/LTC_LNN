# visualization/plotter.py
# Shared plotting utilities used across the pipeline.
#
# Functions:
#   indicator_dashboard    — 4-panel OHLCV + indicator chart per stock
#   cumulative_return      — actual vs predicted price + raw deviation panel
#   research_summary       — one-page summary of all 4 experiment results

import os
from typing import Dict, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation.metrics import returns_to_prices


def _save(fig, path: str, dpi: int = 130) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ── Indicator dashboard ───────────────────────────────────────────

def indicator_dashboard(df: pd.DataFrame,
                        ticker: str,
                        plots_dir: str,
                        label: str = "") -> str:
    sfx  = f"_{label}" if label else ""
    x    = df["Date"] if "Date" in df.columns else df.index

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1, 1]})
    fig.suptitle(f"{ticker} — Technical Dashboard  [{label}]",
                 fontsize=14, fontweight="bold")

    # Close + BB %B
    ax1 = axes[0]
    if "Close" in df.columns:
        ax1.plot(x, df["Close"], color="black", lw=1.5, label="Close")
    if "bb_pct_b" in df.columns:
        ax1.twinx().plot(x, df["bb_pct_b"], color="blue",
                         lw=1, alpha=0.5, ls="--", label="BB %B")
    ax1.set_ylabel("Price ($)"); ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3); ax1.set_title("Close Price + Bollinger %B")

    # RSI
    ax2 = axes[1]
    if "rsi" in df.columns:
        ax2.plot(x, df["rsi"], color="purple", lw=1.2, label="RSI (14)")
        ax2.axhline(70, color="red",   ls="--", alpha=0.5)
        ax2.axhline(30, color="green", ls="--", alpha=0.5)
        ax2.fill_between(x, df["rsi"], 70, where=(df["rsi"] >= 70),
                         color="red", alpha=0.15, interpolate=True)
        ax2.fill_between(x, df["rsi"], 30, where=(df["rsi"] <= 30),
                         color="green", alpha=0.15, interpolate=True)
    ax2.set_ylabel("RSI"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # MACD
    ax3 = axes[2]
    if "macd_norm" in df.columns:
        colors3 = ["green" if v >= 0 else "red" for v in df["macd_norm"]]
        ax3.bar(x, df["macd_norm"], color=colors3, alpha=0.6)
        ax3.set_ylabel("MACD / Close"); ax3.grid(alpha=0.3)

    # ADX
    ax4 = axes[3]
    if "adx" in df.columns:
        ax4.plot(x, df["adx"], color="darkred", lw=1.2, label="ADX (14)")
        ax4.axhline(25, color="black", ls=":", alpha=0.5)
        ax4.fill_between(x, df["adx"], 25, where=(df["adx"] >= 25),
                         color="darkred", alpha=0.15, interpolate=True)
    ax4.set_ylabel("ADX"); ax4.set_xlabel("Date")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    plt.subplots_adjust(hspace=0.08, top=0.95)
    path = os.path.join(plots_dir, f"dashboard_{ticker}{sfx}.png")
    return _save(fig, path)


# ── Actual vs predicted + raw deviation ──────────────────────────

def cumulative_return(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      close_refs: np.ndarray,
                      ticker: str,
                      model_name: str,
                      plots_dir: str,
                      label: str = "") -> str:
    sfx  = f"_{label}" if label else ""
    act  = close_refs * (1 + y_true[:, 0])
    pred = close_refs * (1 + y_pred[:, 0])
    dev  = np.abs(act - pred)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(f"{ticker} — Actual vs Predicted  |  {model_name}  [{label}]",
                 fontsize=13, fontweight="bold")

    axes[0].plot(act,  color="black",   lw=2,   label="Actual")
    axes[0].plot(pred, color="#E15759", lw=1.5, ls="--", alpha=0.85,
                 label="Predicted")
    axes[0].set_ylabel("Price ($)")
    axes[0].set_title("Next-step price")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].fill_between(range(len(dev)), dev, color="#F28E2B", alpha=0.6)
    axes[1].plot(dev, color="#F28E2B", lw=1.2)
    axes[1].axhline(dev.mean(), color="red", ls="--", lw=1.5,
                    label=f"Mean = ${dev.mean():.2f}")
    axes[1].set_ylabel("Raw Deviation ($)")
    axes[1].set_xlabel("Sample index")
    axes[1].set_title("Raw Price Deviation from Actual")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    path = os.path.join(plots_dir,
                        f"cumret_{ticker}_{model_name}{sfx}.png")
    return _save(fig, path)


# ── Research summary (one-page) ───────────────────────────────────

def research_summary(e1_summary: dict,
                     e2_imp_df:  Optional[pd.DataFrame],
                     e3_df:      Optional[pd.DataFrame],
                     plots_dir:  str,
                     label:      str = "") -> str:
    sfx = f"_{label}" if label else ""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Stock2Vec Research Summary  [{label}]\n"
        "4D PCA Identity Vectors for Multi-Stock MIMO LSTM Prediction",
        fontsize=14, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel A — E1 separation
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.bar(["Intra-sector", "Inter-sector"],
             [e1_summary.get("intra_dist_mean", 0),
              e1_summary.get("inter_dist_mean", 0)],
             color=["#59A14F", "#E15759"], edgecolor="white")
    sep = e1_summary.get("separation_ratio", 0)
    sil = e1_summary.get("silhouette", "N/A")
    ax_a.set_title(f"E1 — Separation\nRatio={sep}x  Sil={sil}",
                   fontsize=10, fontweight="bold")
    ax_a.set_ylabel("Avg Euclidean Dist"); ax_a.grid(axis="y", alpha=0.3)

    # Panel B — E2 improvement
    ax_b = fig.add_subplot(gs[0, 1])
    if e2_imp_df is not None and "RMSE_imp(%)" in e2_imp_df.columns:
        vals  = e2_imp_df["RMSE_imp(%)"]
        clrs  = ["#59A14F" if v >= 0 else "#E15759" for v in vals]
        ax_b.barh(e2_imp_df.index, vals, color=clrs, edgecolor="white")
        ax_b.axvline(0, color="black", lw=0.8)
        ax_b.set_title("E2 — RMSE Improvement\n(+) = Stock2Vec better",
                       fontsize=10, fontweight="bold")
        ax_b.set_xlabel("% improvement"); ax_b.grid(axis="x", alpha=0.3)
    else:
        ax_b.text(0.5, 0.5, "E2 N/A", ha="center", va="center",
                  transform=ax_b.transAxes)

    # Panel C — E3 scatter
    ax_c = fig.add_subplot(gs[0, 2])
    if e3_df is not None and len(e3_df) > 0:
        ax_c.scatter(e3_df["centroid_dist"], e3_df["rmse_gap"],
                     s=150, c="#4E79A7", zorder=5)
        for _, row in e3_df.iterrows():
            ax_c.annotate(row["ticker"],
                          (row["centroid_dist"], row["rmse_gap"]),
                          textcoords="offset points", xytext=(5, 3), fontsize=8)
        ax_c.axhline(0, color="black", lw=0.8, ls="--")
        ax_c.set_title("E3 — Generalisation Gap", fontsize=10, fontweight="bold")
        ax_c.set_xlabel("Dist to centroid")
        ax_c.set_ylabel("RMSE gap (zero–full)")
        ax_c.grid(alpha=0.3)
    else:
        ax_c.text(0.5, 0.5, "E3 N/A", ha="center", va="center",
                  transform=ax_c.transAxes)

    # Panel D — conclusions text
    ax_d = fig.add_subplot(gs[1, :])
    ax_d.axis("off")
    avg_imp = (e2_imp_df["RMSE_imp(%)"].mean()
               if e2_imp_df is not None else float("nan"))
    corr_e3 = (np.corrcoef(e3_df["centroid_dist"],
                            e3_df["zero_shot_rmse"])[0, 1]
               if (e3_df is not None and len(e3_df) >= 3)
               else float("nan"))
    lines = [
        "RESEARCH CONCLUSIONS",
        "",
        f"E1 Clustering     : {e1_summary.get('verdict', '—')}",
        f"E2 Ablation       : Stock2Vec avg RMSE improvement = {avg_imp:.1f}%",
        f"E3 Generalisation : Distance–RMSE Pearson ρ = {corr_e3:.3f}",
        "E4 Interpret.     : PC1 ≈ Volatility / broad-market factor",
    ]
    ax_d.text(0.05, 0.95, "\n".join(lines),
              transform=ax_d.transAxes, fontsize=11,
              verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4ff",
                        edgecolor="#4E79A7", alpha=0.9))

    path = os.path.join(plots_dir, f"research_summary{sfx}.png")
    return _save(fig, path)
