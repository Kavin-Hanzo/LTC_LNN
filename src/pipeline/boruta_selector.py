"""
pipeline/boruta_selector.py

Boruta Feature Selection — applied AFTER MinMaxScaling on the train split.

How Boruta works:
  1. For each real feature, create a "shadow" copy with shuffled values.
  2. Train a RandomForest on real + shadow features combined.
  3. A real feature is "important" only if its importance consistently
     exceeds the best shadow feature's importance across many iterations.
  4. Features that never beat shadow features are marked as "unimportant"
     and dropped from all splits (train, val, test).

Why apply AFTER scaling:
  - Boruta uses RandomForest feature importances, which are scale-invariant.
  - But applying it on scaled data keeps the pipeline consistent:
    the same column indices used by the scaler are preserved for
    inverse_scale_close() to work correctly.
  - We reindex close_col_idx after dropping columns.

Key constraint:
  - 'Close' column is ALWAYS kept regardless of Boruta result.
    It is the forecast target and must remain in the feature set.

Usage:
    from pipeline.boruta_selector import run_boruta

    train_sc, val_sc, test_sc, selected_cols, new_close_idx = run_boruta(
        train_sc    = train_sc,
        val_sc      = val_sc,
        test_sc     = test_sc,
        feature_cols = config["data"]["features"],
        close_col_idx = close_col_idx,
        config      = config,
    )
"""

from __future__ import annotations

import json
import os
import warnings
from typing import List, Tuple

import numpy as np

# suppress the sklearn/joblib parallel warning that BorutaPy triggers internally
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)


def run_boruta(
    train_sc:      np.ndarray,
    val_sc:        np.ndarray,
    test_sc:       np.ndarray,
    feature_cols:  List[str],
    close_col_idx: int,
    config:        dict,
    save_dir:      str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], int]:
    """
    Run Boruta feature selection on scaled train data.
    Drop unimportant columns from train, val, and test arrays.
    Always keeps the Close column regardless of result.

    Args:
        train_sc:      scaled train array  (N_train, n_features)
        val_sc:        scaled val array    (N_val,   n_features)
        test_sc:       scaled test array   (N_test,  n_features)
        feature_cols:  list of feature names matching column order
        close_col_idx: index of Close in feature_cols
        config:        full config dict (reads config.boruta.*)
        save_dir:      if set, saves boruta_result.json here

    Returns:
        train_sc_sel:    reduced train array
        val_sc_sel:      reduced val array
        test_sc_sel:     reduced test array
        selected_cols:   list of kept feature names
        new_close_idx:   new index of Close in selected_cols
    """
    try:
        from boruta import BorutaPy
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        raise ImportError(
            "Boruta not installed. Run:  pip install boruta\n"
            "Also requires scikit-learn (already in requirements.txt)."
        )

    boruta_cfg   = config.get("boruta", {})
    n_estimators = boruta_cfg.get("n_estimators", 100)
    max_iter     = boruta_cfg.get("max_iter",     100)
    perc         = boruta_cfg.get("perc",         100)
    alpha        = boruta_cfg.get("alpha",         0.05)

    n_features = train_sc.shape[1]
    close_col  = feature_cols[close_col_idx]

    print(f"\n[Boruta]  running on {n_features} features  "
          f"(n_estimators={n_estimators}, max_iter={max_iter}, "
          f"perc={perc}, alpha={alpha})")

    # ── build X (all features) and y (Close column) from train split
    X = train_sc.astype(np.float32)
    y = train_sc[:, close_col_idx].astype(np.float32)

    # ── RandomForest estimator for Boruta
    rf = RandomForestRegressor(
        n_estimators = n_estimators,
        n_jobs       = -1,
        max_depth    = 5,
        random_state = config.get("training", {}).get("seed", 42),
    )

    # ── run Boruta
    selector = BorutaPy(
        estimator   = rf,
        n_estimators= "auto",
        perc        = perc,
        alpha       = alpha,
        max_iter    = max_iter,
        verbose     = 0,
        random_state= config.get("training", {}).get("seed", 42),
    )
    selector.fit(X, y)

    # ── collect results
    support        = selector.support_           # True = confirmed important
    support_weak   = selector.support_weak_      # True = tentatively important
    ranking        = selector.ranking_           # 1 = most important

    # accepted = confirmed important OR tentatively important
    accepted_mask  = support | support_weak

    # ALWAYS keep Close regardless of Boruta decision
    accepted_mask[close_col_idx] = True

    selected_indices = np.where(accepted_mask)[0].tolist()
    selected_cols    = [feature_cols[i] for i in selected_indices]
    rejected_cols    = [feature_cols[i] for i in range(n_features)
                        if not accepted_mask[i]]

    new_close_idx    = selected_cols.index(close_col)

    print(f"  [Boruta]  selected  ({len(selected_cols)}): {selected_cols}")
    if rejected_cols:
        print(f"  [Boruta]  rejected  ({len(rejected_cols)}): {rejected_cols}")
    else:
        print(f"  [Boruta]  all features accepted")

    # ── apply selection to all splits
    train_sc_sel = train_sc[:, selected_indices]
    val_sc_sel   = val_sc[:,   selected_indices]
    test_sc_sel  = test_sc[:,  selected_indices]

    # ── save result for inspection
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        result = {
            "selected_features":  selected_cols,
            "rejected_features":  rejected_cols,
            "new_close_col_idx":  new_close_idx,
            "n_features_before":  n_features,
            "n_features_after":   len(selected_cols),
            "ranking": {
                feature_cols[i]: int(ranking[i])
                for i in range(n_features)
            },
        }
        path = os.path.join(save_dir, "boruta_result.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  [Boruta]  result saved -> {path}")

    return train_sc_sel, val_sc_sel, test_sc_sel, selected_cols, new_close_idx