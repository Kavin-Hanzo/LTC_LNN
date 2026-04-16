# experiments/e1_clustering.py
# Experiment 1 — prove the Stock2Vec vectors capture sector identity.
#
# Evidence:
#   1. t-SNE 2D scatter  — same-sector stocks should cluster visually
#   2. Cosine similarity heatmap
#   3. Euclidean distance bar chart  — intra vs inter-sector gap
#   4. PC loading bars per dimension
#   5. Silhouette score

import os
from typing import Dict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Sub-sector labels and colours for US IT universe
_SECTOR = {
    "AAPL":  "Software/Internet",
    "MSFT":  "Software/Internet",
    "GOOGL": "Software/Internet",
    "META":  "Software/Internet",
    "IBM":   "Enterprise/Legacy",
    "NVDA":  "Semiconductors",
}
_COLOR = {
    "AAPL": "#4E79A7", "MSFT": "#4E79A7",
    "GOOGL": "#4E79A7", "META": "#4E79A7",
    "IBM":  "#E15759",
    "NVDA": "#F28E2B",
}


def run(vectors: Dict[str, np.ndarray],
        plots_dir:   str,
        results_dir: str,
        label:       str = "") -> dict:

    os.makedirs(plots_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    sns.set_style("whitegrid")
    sfx = f"_{label}" if label else ""

    tickers = sorted(vectors.keys())
    V       = np.stack([vectors[t] for t in tickers])   # (N, D)
    n       = len(tickers)

    print(f"\n{'═'*55}")
    print(f"  E1 — Clustering Proof  [{label}]")
    print(f"{'═'*55}")

    # ── 1. t-SNE ─────────────────────────────────────────────────
    sk_ver  = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    tsne_kw = dict(n_components=2, perplexity=max(2, n - 2),
                   learning_rate="auto", init="pca", random_state=42)
    tsne_kw["max_iter" if sk_ver >= (1, 2) else "n_iter"] = 3000
    coords  = TSNE(**tsne_kw).fit_transform(V)

    fig, ax = plt.subplots(figsize=(9, 7))
    seen_sectors = {}
    for i, t in enumerate(tickers):
        c = _COLOR.get(t, "#888888")
        s = _SECTOR.get(t, "Unknown")
        ax.scatter(coords[i, 0], coords[i, 1],
                   c=c, s=280, zorder=5, edgecolors="white", lw=2)
        ax.annotate(t, coords[i], textcoords="offset points",
                    xytext=(9, 5), fontsize=10, fontweight="bold")
        seen_sectors[s] = c
    patches = [mpatches.Patch(facecolor=c, label=s)
               for s, c in sorted(seen_sectors.items())]
    ax.legend(handles=patches, title="Sub-Sector", fontsize=9, framealpha=0.9)
    ax.set_title(f"E1 — t-SNE Projection of Stock2Vec  [{label}]",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.25)
    _save(fig, plots_dir, f"e1_tsne{sfx}.png")

    # ── 2. Cosine similarity heatmap ─────────────────────────────
    cos = pd.DataFrame(cosine_similarity(V), index=tickers, columns=tickers)
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    sns.heatmap(cos, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=-1, vmax=1, ax=ax2, square=True, linewidths=0.4,
                annot_kws={"size": 9},
                cbar_kws={"label": "Cosine Similarity"})
    ax2.set_title(f"E1 — Cosine Similarity  [{label}]",
                  fontsize=12, fontweight="bold")
    ax2.tick_params(axis="x", rotation=30, labelsize=9)
    ax2.tick_params(axis="y", rotation=0,  labelsize=9)
    _save(fig2, plots_dir, f"e1_cosine{sfx}.png")

    # ── 3. Euclidean pair distances ───────────────────────────────
    euc    = euclidean_distances(V)
    pairs, dists, intra = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = tickers[i], tickers[j]
            same   = _SECTOR.get(t1) == _SECTOR.get(t2)
            pairs.append(f"{t1} ↔ {t2}")
            dists.append(euc[i, j])
            intra.append(same)

    pdf = pd.DataFrame({"pair": pairs, "dist": dists,
                        "intra": intra}).sort_values("dist")
    intra_mean = pdf[pdf.intra]["dist"].mean()
    inter_mean = pdf[~pdf.intra]["dist"].mean()
    sep_ratio  = inter_mean / intra_mean if intra_mean > 0 else np.inf

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    bar_colors = ["#59A14F" if s else "#E15759" for s in pdf.intra]
    ax3.barh(pdf.pair, pdf.dist, color=bar_colors, edgecolor="white", lw=0.4)
    ax3.axvline(intra_mean, color="#59A14F", ls="--", lw=2,
                label=f"Avg intra ({intra_mean:.3f})")
    ax3.axvline(inter_mean, color="#E15759", ls="--", lw=2,
                label=f"Avg inter ({inter_mean:.3f})")
    ax3.invert_yaxis()
    ax3.set_title(f"E1 — Pair Distances  separation={sep_ratio:.2f}x  [{label}]",
                  fontsize=12, fontweight="bold")
    ax3.set_xlabel("Euclidean Distance (4D)")
    ax3.legend(handles=[mpatches.Patch(facecolor="#59A14F", label="Intra-sector"),
                        mpatches.Patch(facecolor="#E15759", label="Inter-sector")],
               fontsize=9)
    ax3.tick_params(axis="y", labelsize=8.5)
    _save(fig3, plots_dir, f"e1_distances{sfx}.png")

    # ── 4. PC loading bars ────────────────────────────────────────
    n_dim = V.shape[1]
    fig4, axes4 = plt.subplots(1, n_dim, figsize=(5 * n_dim, 5))
    if n_dim == 1:
        axes4 = [axes4]
    load_df = pd.DataFrame(V, index=tickers,
                           columns=[f"PC{i+1}" for i in range(n_dim)])
    for k, ax_k in enumerate(axes4):
        col = f"PC{k+1}"
        s   = load_df.sort_values(col)
        ax_k.barh(s.index, s[col],
                  color=[_COLOR.get(t, "#888") for t in s.index],
                  edgecolor="white")
        ax_k.axvline(0, color="black", lw=0.8)
        ax_k.set_title(col, fontsize=11, fontweight="bold")
        ax_k.set_xlabel("Loading")
    fig4.suptitle(f"E1/E4 — PC Loadings per Dimension  [{label}]",
                  fontsize=12, fontweight="bold")
    _save(fig4, plots_dir, f"e1_loadings{sfx}.png")

    # ── Silhouette score ──────────────────────────────────────────
    sec_ids = np.array([list(set(_SECTOR.values())).index(
                            _SECTOR.get(t, "Unknown")) for t in tickers])
    try:
        sil = silhouette_score(V, sec_ids) if len(set(sec_ids)) > 1 else None
    except Exception:
        sil = None

    summary = {
        "label":            label,
        "intra_dist_mean":  round(intra_mean, 4),
        "inter_dist_mean":  round(inter_mean, 4),
        "separation_ratio": round(sep_ratio,  3),
        "silhouette":       round(sil, 4) if sil else "N/A",
        "verdict": ("✅ STRONG" if sep_ratio >= 1.5 else "⚠  WEAK"),
    }
    load_df.to_csv(os.path.join(results_dir, f"e1_loadings{sfx}.csv"))
    pd.Series(summary).to_csv(
        os.path.join(results_dir, f"e1_summary{sfx}.csv"), header=False)

    print(f"  Separation ratio : {sep_ratio:.2f}x")
    print(f"  Silhouette score : {sil}")
    print(f"  Verdict          : {summary['verdict']}\n")
    return summary


def _save(fig, plots_dir, fname, dpi=130):
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, fname), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [E1] → {fname}")
