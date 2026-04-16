# data/dataset.py
# MIMO sliding-window PyTorch Dataset for multi-stock training.
#
# Each sample:
#   x          (lookback, n_features)  — scaled indicator window
#   identity   (identity_dim,)         — Stock2Vec vector for this stock
#   y          (horizon,)              — target normalised close returns
#   close_ref  float                   — raw close price at window end
#                                        (for price-space error reporting)
#   ticker     str                     — which company this sample belongs to

from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset


class MIMODataset(Dataset):
    def __init__(self,
                 scaled_X:    Dict[str, np.ndarray],
                 closes:      Dict[str, np.ndarray],
                 identities:  Dict[str, np.ndarray],
                 tickers:     List[str],
                 lookback:    int,
                 horizon:     int,
                 ret_col_idx: int = 0):
        self.lookback = lookback
        self.horizon  = horizon
        self.tickers  = tickers
        self.samples: List[dict] = []

        for ticker in tickers:
            X   = scaled_X[ticker]       # (T, F)
            cls = closes[ticker]         # (T,)
            idn = identities[ticker]     # (D,)
            n   = len(X)

            for s in range(n - lookback - horizon + 1):
                e = s + lookback
                self.samples.append({
                    "x":         torch.tensor(X[s:e],              dtype=torch.float32),
                    "identity":  torch.tensor(idn,                  dtype=torch.float32),
                    "y":         torch.tensor(X[e:e+horizon, ret_col_idx],
                                              dtype=torch.float32),
                    "close_ref": float(cls[e - 1]),
                    "ticker":    ticker,
                })

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def collate(batch: List[dict]) -> dict:
    return {
        "x":         torch.stack([b["x"]        for b in batch]),
        "identity":  torch.stack([b["identity"]  for b in batch]),
        "y":         torch.stack([b["y"]         for b in batch]),
        "close_ref": torch.tensor([b["close_ref"] for b in batch],
                                   dtype=torch.float32),
        "ticker":    [b["ticker"] for b in batch],
    }


def build_datasets(processed:    Dict,
                   identities:   Dict[str, np.ndarray],
                   scaled_splits: Dict,
                   tickers:      List[str],
                   lookback:     int,
                   horizon:      int,
                   ret_col_idx:  int = 0):
    """
    Build (train_ds, val_ds, test_ds) from pre-computed scaled splits.
    scaled_splits: {ticker: (X_train, X_val, X_test, scaler)}
    processed    : {ticker: (df_with_features, feat_cols)}
    """
    split_data = {"train": {}, "val": {}, "test": {}}

    for ticker in tickers:
        df, _ = processed[ticker]
        X_tr, X_va, X_te, _ = scaled_splits[ticker]
        n_tr = len(X_tr)
        n_va = len(X_va)

        raw_closes = (df["Close"].values
                      if "Close" in df.columns
                      else np.ones(len(df)))

        split_data["train"][ticker] = {"X": X_tr, "closes": raw_closes[:n_tr]}
        split_data["val"][ticker]   = {"X": X_va, "closes": raw_closes[n_tr: n_tr + n_va]}
        split_data["test"][ticker]  = {"X": X_te, "closes": raw_closes[n_tr + n_va:]}

    def _make(split):
        return MIMODataset(
            scaled_X   = {t: split_data[split][t]["X"]      for t in tickers},
            closes     = {t: split_data[split][t]["closes"]  for t in tickers},
            identities = identities,
            tickers    = tickers,
            lookback   = lookback,
            horizon    = horizon,
            ret_col_idx = ret_col_idx,
        )

    tr, va, te = _make("train"), _make("val"), _make("test")
    print(f"  [Dataset] lookback={lookback} horizon={horizon} "
          f"train={len(tr)} val={len(va)} test={len(te)}")
    return tr, va, te
