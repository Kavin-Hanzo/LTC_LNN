# vectors/stock2vec.py
# Build a stable 4D identity vector for each stock using PCA
# on the cross-stock RETURN correlation matrix.
#
# Why PCA loadings instead of per-stock indicator statistics?
#   - Indicator stats drift with market conditions.
#   - PCA loadings encode "which market forces does this stock respond to" —
#     a structural property that changes slowly.
#   - Fama-French interpretation:
#       PC1 ≈ broad market factor
#       PC2 ≈ growth vs value
#       PC3 ≈ sub-sector factor
#       PC4 ≈ idiosyncratic residual
#
# CRITICAL: PCA is fit ONLY on the training-split close prices.
#           The test window is never seen during vector construction.

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def build(aligned_closes: pd.DataFrame,
          n_components:   int = 4,
          save_dir:       str = "outputs/vectors",
          label:          str = "") -> Tuple[Dict[str, np.ndarray], PCA, float]:
    """
    Build Stock2Vec embeddings from a (T × N) training-split Close price matrix.

    Returns
    -------
    vectors : {ticker: np.ndarray of shape (n_components,)}
    pca     : fitted PCA object  (used in E4 interpretability)
    var_captured : total fraction of variance explained
    """
    os.makedirs(save_dir, exist_ok=True)
    tickers = list(aligned_closes.columns)

    # Work in return space — removes price-level incomparability
    returns = aligned_closes.pct_change().dropna()
    X       = StandardScaler().fit_transform(returns)     # (T-1, N)

    n_comp = min(n_components, len(tickers) - 1, len(returns) - 1)
    pca    = PCA(n_components=n_comp, random_state=42).fit(X)

    # components_ shape: (n_comp, N)
    # Transpose → each STOCK is one row of length n_comp
    loadings = pca.components_.T                          # (N, n_comp)

    vectors: Dict[str, np.ndarray] = {
        t: loadings[i].astype(np.float32)
        for i, t in enumerate(tickers)
    }
    var_captured = float(np.sum(pca.explained_variance_ratio_))

    # ── Save lookup table ────────────────────────────────────────
    cols    = [f"v{i+1}" for i in range(n_comp)]
    lkp     = pd.DataFrame(loadings, index=tickers, columns=cols)
    lkp.index.name = "ticker"
    fname   = f"stock2vec_{label}.csv" if label else "stock2vec.csv"
    lkp.to_csv(os.path.join(save_dir, fname))

    ev = pd.DataFrame({
        "component":     [f"PC{i+1}" for i in range(n_comp)],
        "explained_var": pca.explained_variance_ratio_,
        "cumulative":    np.cumsum(pca.explained_variance_ratio_),
    })
    ev.to_csv(os.path.join(save_dir,
                           f"pca_variance_{label}.csv"), index=False)

    print(f"[Stock2Vec] {n_comp}D vectors for {len(tickers)} stocks  "
          f"variance_captured={var_captured:.2%}")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {v:.2%}  "
              f"(cum {np.cumsum(pca.explained_variance_ratio_)[i]:.2%})")

    return vectors, pca, var_captured


def zero_vector(n_components: int = 4) -> np.ndarray:
    """Identity-free zero vector — used for the Baseline model."""
    return np.zeros(n_components, dtype=np.float32)
