# experiments/e4_interpretability.py
# Experiment 4 — map each PC dimension to a known financial factor.
#
# Produces:
#   Radar fingerprint chart (all stocks overlaid)
#   Scatter: each PC loading vs each financial proxy
#   Correlation heatmap: loadings × proxies
#   Ranked CSVs per PC dimension
#
# Financial proxies derived from training-split prices:
#   realised_vol   annualised daily return std-dev  (≈ volatility / beta)
#   momentum_1y    1-year total return              (≈ growth factor)
#   mean_return    annualised mean daily return

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_SECTOR = {
    "AAPL":  "Software/Internet", "MSFT":  "Software/Internet",
    "GOOGL": "Software/Internet", "META":  "Software/Internet",
    "IBM":   "Enterprise/Legacy", "NVDA":  "Semiconductors",
}
_COLOR = {
    "AAPL": "#4E79A7", "MSFT": "#4E79A7",
    "GOOGL": "#4E79A7", "META": "#4E79A7",
    "IBM": "#E15759", "NVDA": "#F28E2B",
}


def _proxies(aligned_closes: pd.DataFrame) -> pd.DataFrame:
    ret   = aligned_closes.pct_change().dropna()
    rows  = {}
    for t in aligned_closes.columns:
        r = ret[t].dropna()
        rows[t] = {
            "realised_vol": r.std() * np.sqrt(252),
            "momentum_1y":  float((1 + r).prod() - 1),
            "mean_return":  r.mean() * 252,
        }
    return pd.DataFrame(rows).T


def run(vectors:        Dict[str, np.ndarray],
        aligned_closes: pd.DataFrame,
        plots_dir:      str,
        results_dir:    str,
        label:          str = "") -> dict:

    os.makedirs(plots_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    sns.set_style("whitegrid")
    sfx = f"_{label}" if label else ""

    print(f"\n{'═'*55}")
    print(f"  E4 — Interpretability  [{label}]")
    print(f"{'═'*55}")

    tickers = sorted(vectors.keys())
    n_dim   = len(next(iter(vectors.values())))
    pc_cols = [f"PC{i+1}" for i in range(n_dim)]
    load_df = pd.DataFrame({t: vectors[t] for t in tickers},
                           index=pc_cols).T
    prx_df  = _proxies(aligned_closes)

    # ── 1. Radar fingerprint ──────────────────────────────────────
    angles = np.linspace(0, 2 * np.pi, n_dim, endpoint=False).tolist()
    angles += angles[:1]
    fig1, ax1 = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax1.set_theta_offset(np.pi / 2); ax1.set_theta_direction(-1)
    ax1.set_xticks(angles[:-1]); ax1.set_xticklabels(pc_cols, fontsize=11)
    for t in tickers:
        v = vectors[t].tolist() + [vectors[t][0]]
        c = _COLOR.get(t, "#888")
        ax1.plot(angles, v, lw=2, color=c, label=t)
        ax1.fill(angles, v, alpha=0.04, color=c)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax1.set_title(f"E4 — PC Fingerprints  [{label}]",
                  fontsize=12, fontweight="bold", pad=20)
    _save(fig1, plots_dir, f"e4_radar{sfx}.png")

    # ── 2. Scatter: PC loading vs each proxy ─────────────────────
    proxy_cols = [c for c in prx_df.columns]
    fig2, axes2 = plt.subplots(n_dim, len(proxy_cols),
                               figsize=(5 * len(proxy_cols), 4.5 * n_dim))
    if n_dim == 1:
        axes2 = [axes2]
    fig2.suptitle(f"E4 — PC Loading vs Financial Proxy  [{label}]",
                  fontsize=12, fontweight="bold")
    common = [t for t in tickers if t in prx_df.index]
    for r, pc in enumerate(pc_cols):
        for c, prc in enumerate(proxy_cols):
            ax = axes2[r][c] if n_dim > 1 else axes2[c]
            xs = prx_df.loc[common, prc].values.astype(float)
            ys = load_df.loc[common, pc].values.astype(float)
            colors = [_COLOR.get(t, "#888") for t in common]
            ax.scatter(xs, ys, c=colors, s=150, zorder=5,
                       edgecolors="white", lw=1.5)
            for i, t in enumerate(common):
                ax.annotate(t, (xs[i], ys[i]),
                            textcoords="offset points",
                            xytext=(5, 3), fontsize=8)
            if len(xs) >= 3:
                corr = np.corrcoef(xs, ys)[0, 1]
                ax.text(0.05, 0.92, f"ρ={corr:.3f}",
                        transform=ax.transAxes, fontsize=9,
                        color="navy", fontweight="bold")
            ax.set_xlabel(prc.replace("_", " ").title(), fontsize=9)
            ax.set_ylabel(pc, fontsize=9)
            ax.grid(alpha=0.25)
    _save(fig2, plots_dir, f"e4_pc_vs_proxy{sfx}.png")

    # ── 3. Correlation heatmap ────────────────────────────────────
    combined = pd.concat([load_df.loc[common], prx_df.loc[common]], axis=1)
    corr_mat = combined.corr().loc[pc_cols, proxy_cols]
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr_mat, annot=True, fmt=".3f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, ax=ax3,
                linewidths=0.5, annot_kws={"size": 10},
                cbar_kws={"label": "Pearson ρ"})
    ax3.set_title(f"E4 — Loadings × Proxy Correlation  [{label}]",
                  fontsize=12, fontweight="bold")
    ax3.tick_params(axis="x", rotation=30, labelsize=9)
    ax3.tick_params(axis="y", rotation=0,  labelsize=9)
    _save(fig3, plots_dir, f"e4_corr_heatmap{sfx}.png")

    # ── Save ranked tables ────────────────────────────────────────
    for pc in pc_cols:
        ranked = load_df[[pc]].join(prx_df).sort_values(pc, ascending=False)
        ranked.to_csv(os.path.join(results_dir, f"e4_ranked_{pc}{sfx}.csv"))

    # ── Print hints ───────────────────────────────────────────────
    if common:
        rho = np.corrcoef(prx_df.loc[common, "realised_vol"],
                          load_df.loc[common, "PC1"])[0, 1]
        print(f"  PC1 × realised_vol ρ = {rho:.3f}  "
              f"({'≈ Market/Vol factor ✅' if abs(rho) > 0.5 else 'weak correlation'})")
    print(f"  [E4] plots → {plots_dir}\n")

    return {"load_df": load_df, "proxies": prx_df}


def _save(fig, plots_dir, fname, dpi=130):
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, fname), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [E4] → {fname}")
