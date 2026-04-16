# experiments/e3_generalization.py
# Experiment 3 — cross-stock zero-shot generalization.
#
# Protocol: leave-one-out.
# For each ticker T:
#   Train Stock2Vec model on all tickers EXCEPT T
#   At inference feed T's Stock2Vec vector (not seen during training)
#   Compare zero-shot RMSE vs full-training RMSE
#
# Hypothesis: zero-shot RMSE correlates with Euclidean distance
# between T's vector and the training-set centroid.
# If the correlation is positive and strong → closer vectors generalise better.

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def run(loo_results:        List[Dict],
        vectors:            Dict[str, np.ndarray],
        full_train_metrics: Dict[str, Dict],
        plots_dir:          str,
        results_dir:        str,
        label:              str = "") -> pd.DataFrame:
    """
    loo_results: [{ticker, zero_shot_rmse, zero_shot_rawdev}, ...]
    """
    os.makedirs(plots_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    sns.set_style("whitegrid")
    sfx = f"_{label}" if label else ""

    print(f"\n{'═'*55}")
    print(f"  E3 — Cross-Stock Generalisation  [{label}]")
    print(f"{'═'*55}")

    all_tickers = list(vectors.keys())
    rows = []
    for item in loo_results:
        t       = item["ticker"]
        train_v = [vectors[k] for k in all_tickers if k != t]
        centroid = np.mean(train_v, axis=0)
        dist    = float(np.linalg.norm(vectors[t] - centroid))
        full_r  = full_train_metrics.get(t, {}).get("rmse_ret", np.nan)
        zero_r  = item["zero_shot_rmse"]
        rows.append({
            "ticker":           t,
            "centroid_dist":    round(dist,   4),
            "zero_shot_rmse":   round(zero_r, 6),
            "full_train_rmse":  round(full_r, 6),
            "rmse_gap":         round(zero_r - full_r, 6),
            "zero_shot_rawdev": round(item.get("zero_shot_rawdev", np.nan), 2),
        })
        print(f"  {t:8s}  dist={dist:.4f}  "
              f"zero={zero_r:.6f}  full={full_r:.6f}  "
              f"gap={zero_r-full_r:+.6f}")

    df = pd.DataFrame(rows).sort_values("centroid_dist")
    df.to_csv(os.path.join(results_dir, f"e3_results{sfx}.csv"), index=False)

    # ── Plot 1: distance vs RMSE scatter ─────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(df.centroid_dist, df.zero_shot_rmse,
                s=180, zorder=5, color="#4E79A7")
    for _, row in df.iterrows():
        ax1.annotate(row.ticker, (row.centroid_dist, row.zero_shot_rmse),
                     textcoords="offset points", xytext=(7, 4), fontsize=10)
    if len(df) >= 3:
        cf = np.polyfit(df.centroid_dist, df.zero_shot_rmse, 1)
        xl = np.linspace(df.centroid_dist.min(), df.centroid_dist.max(), 100)
        ax1.plot(xl, np.polyval(cf, xl), "r--", lw=1.5, alpha=0.7, label="Trend")
        ax1.legend(fontsize=9)
    ax1.set_xlabel("Embedding Distance to Training Centroid")
    ax1.set_ylabel("Zero-Shot RMSE")
    ax1.set_title(f"E3 — Closer Vector → Better Generalisation  [{label}]",
                  fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)
    _save(fig1, plots_dir, f"e3_dist_vs_rmse{sfx}.png")

    # ── Plot 2: zero-shot vs full-training RMSE ───────────────────
    x, w = np.arange(len(df)), 0.35
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(x - w/2, df.full_train_rmse, w,
            color="#4E79A7", label="Full Training", alpha=0.9)
    ax2.bar(x + w/2, df.zero_shot_rmse,  w,
            color="#F28E2B", label="Zero-Shot",     alpha=0.9)
    ax2.set_xticks(x); ax2.set_xticklabels(df.ticker, fontsize=10)
    ax2.set_ylabel("RMSE (return space)")
    ax2.set_title(f"E3 — Zero-Shot vs Full Training  [{label}]",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
    _save(fig2, plots_dir, f"e3_zeroshot_vs_full{sfx}.png")

    # ── Correlation verdict ───────────────────────────────────────
    corr = (np.corrcoef(df.centroid_dist, df.zero_shot_rmse)[0, 1]
            if len(df) >= 3 else np.nan)
    verdict = ("✅ Supported" if not np.isnan(corr) and corr > 0.5
               else "⚠  Inconclusive")
    print(f"\n  Distance–RMSE ρ = {corr:.3f}   →  {verdict}\n")
    return df


def _save(fig, plots_dir, fname, dpi=130):
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, fname), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [E3] → {fname}")
