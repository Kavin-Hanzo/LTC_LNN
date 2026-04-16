# data/scaler.py
# Walk-forward scaler — fit ONLY on the training split, then apply
# the same statistics to val and test. Zero lookahead bias.

import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class WalkForwardScaler:
    """
    Thin wrapper around sklearn scalers that enforces walk-forward semantics:
      1. fit(X_train)
      2. transform(X_train / X_val / X_test)
      3. inverse_transform(X_scaled)
    """

    def __init__(self, method: str = "standard"):
        if method == "standard":
            self._sk = StandardScaler()
        elif method == "minmax":
            self._sk = MinMaxScaler()
        else:
            raise ValueError(f"method must be 'standard' or 'minmax', got '{method}'")
        self.method      = method
        self._fitted     = False
        self.n_features_ = 0

    def fit(self, X: np.ndarray) -> "WalkForwardScaler":
        assert X.ndim == 2, f"Expected 2D array, got shape {X.shape}"
        self._sk.fit(X)
        self._fitted     = True
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check()
        return self._sk.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        self._check()
        return self._sk.inverse_transform(X)

    def inverse_transform_col(self, values: np.ndarray,
                               col_idx: int) -> np.ndarray:
        """Inverse-transform a single column without needing all features."""
        self._check()
        flat  = values.reshape(-1, 1)
        dummy = np.zeros((len(flat), self.n_features_))
        dummy[:, col_idx] = flat[:, 0]
        return self._sk.inverse_transform(dummy)[:, col_idx].reshape(values.shape)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "WalkForwardScaler":
        with open(path, "rb") as f:
            return pickle.load(f)

    def _check(self):
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call .fit(X_train) first.")


def chronological_split(df: pd.DataFrame,
                         train_ratio: float = 0.70,
                         val_ratio:   float = 0.15
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered DataFrame into train / val / test.
    NEVER shuffle — that leaks future data into training.
    """
    n    = len(df)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    return (df.iloc[:n_tr].copy(),
            df.iloc[n_tr : n_tr + n_va].copy(),
            df.iloc[n_tr + n_va:].copy())


def scale_splits(train: pd.DataFrame,
                 val:   pd.DataFrame,
                 test:  pd.DataFrame,
                 feat_cols: list,
                 method:    str = "standard"
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, WalkForwardScaler]:
    """
    Convenience: split DataFrames → scaled numpy arrays + fitted scaler.
    Returns (X_train, X_val, X_test, scaler).
    """
    scaler  = WalkForwardScaler(method=method)
    X_train = scaler.fit_transform(train[feat_cols].values)
    X_val   = scaler.transform(val[feat_cols].values)
    X_test  = scaler.transform(test[feat_cols].values)
    return X_train, X_val, X_test, scaler
