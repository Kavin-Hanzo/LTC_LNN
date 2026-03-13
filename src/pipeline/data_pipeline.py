"""
pipeline/data_pipeline.py

Full data pipeline:
  1. Fetch OHLCV via yfinance
  2. Feature engineering  (RSI, MACD, SMA_20, Volume_Delta + raw OHLCV)
  3. Time-ordered train / val / test split  (NO shuffle)
  4. MinMaxScaler fit on train only  ->  transform all splits
  5. Sliding-window dataset  ->  DataLoaders

Exports used downstream:
  build_dataloaders()      -> DataLoaders  (trainer, evaluator)
  load_config()            -> dict
  inverse_scale_close()    -> np.ndarray   (evaluator, predictor)
  engineer_features()      -> pd.DataFrame (predictor)
  fetch_ohlcv()            -> pd.DataFrame (predictor)
  TimeSeriesDataset        (type reference)
  DataLoaders              (type reference)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── 1. Fetch ──────────────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    """
    Download OHLCV from yfinance.
    Returns DataFrame indexed by Date: Open, High, Low, Close, Volume.
    """
    import yfinance as yf

    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' (period={period}).")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    print(f"  [fetch]    {ticker}  |  {len(df)} rows  "
          f"|  {df.index[0].date()} -> {df.index[-1].date()}")
    return df


# ── 2. Feature engineering ────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return close.ewm(span=fast, adjust=False).mean() \
         - close.ewm(span=slow, adjust=False).mean()


def _sma(close: pd.Series, period: int = 20) -> pd.Series:
    return close.rolling(window=period).mean()


def _volume_delta(volume: pd.Series) -> pd.Series:
    return volume.pct_change().fillna(0)


RAW_COLS = {"Open", "High", "Low", "Close", "Volume"}

FEATURE_BUILDERS = {
    "RSI":          lambda df: _rsi(df["Close"]),
    "MACD":         lambda df: _macd(df["Close"]),
    "SMA_20":       lambda df: _sma(df["Close"]),
    "Volume_Delta": lambda df: _volume_delta(df["Volume"]),
}


def engineer_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """
    Build feature DataFrame from raw OHLCV.
    Supported: Open, High, Low, Close, Volume, RSI, MACD, SMA_20, Volume_Delta
    Drops NaN rows introduced by rolling/EWM warm-up.
    """
    result = pd.DataFrame(index=df.index)

    for feat in feature_list:
        if feat in RAW_COLS:
            result[feat] = df[feat].values
        elif feat in FEATURE_BUILDERS:
            result[feat] = FEATURE_BUILDERS[feat](df).values
        else:
            raise ValueError(
                f"Unknown feature '{feat}'. "
                f"Supported: {sorted(list(RAW_COLS) + list(FEATURE_BUILDERS.keys()))}"
            )

    before = len(result)
    result.dropna(inplace=True)
    dropped = before - len(result)
    if dropped:
        print(f"  [features] dropped {dropped} NaN rows (indicator warm-up)")

    print(f"  [features] cols={list(result.columns)}  rows={len(result)}")
    return result


# ── 3. Time-ordered split ─────────────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    val_split: float,
    test_split: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split — NO shuffle. Layout: [train | val | test]
    If val_split=0.0, val DataFrame is empty (build_dataloaders handles carving).
    """
    n       = len(df)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_val - n_test

    train = df.iloc[:n_train]
    val   = df.iloc[n_train : n_train + n_val]   # empty slice if n_val=0
    test  = df.iloc[n_train + n_val :]

    if n_val > 0:
        print(f"  [split]    train={len(train)}  val={len(val)}  test={len(test)}")
    return train, val, test


# ── 4. Scaling ────────────────────────────────────────────────────────────────
#
#  The scaler is fit ONLY on the train split to prevent data leakage.
#  It is then used to transform val and test.
#
#  inverse_scale_close() is the shared helper used by:
#    - evaluate.py   -> convert scaled predictions back to $ for metrics
#    - predictor.py  -> convert model output back to $ for the API response
#
#  data_pipeline.py itself does NOT call inverse_scale_close — it only
#  produces scaled data. The function lives here because it depends on the
#  scaler's structure (n_features, close_col_idx) which is owned by this module.

def fit_scale(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Fit MinMaxScaler on train only. Transform all three splits.
    Returns (train_arr, val_arr, test_arr, scaler).
    """
    scaler       = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.values)   # fit + transform
    val_scaled   = scaler.transform(val.values)         # transform only
    test_scaled  = scaler.transform(test.values)        # transform only
    print(f"  [scaler]   fitted on train ({len(train)} rows)  "
          f"n_features={train.shape[1]}")
    return train_scaled, val_scaled, test_scaled, scaler


def inverse_scale_close(
    scaled_values: np.ndarray,
    scaler:        MinMaxScaler,
    close_col_idx: int,
    n_features:    int,
) -> np.ndarray:
    """
    Inverse-transform Close price predictions back to original dollar scale.

    This function is NOT called during training — it is used AFTER training
    by evaluate.py (metrics on real $ prices) and predictor.py (API response).

    How it works:
      MinMaxScaler.inverse_transform() needs a full (n, n_features) array.
      We build a zero dummy array, slot the Close predictions into the correct
      column, inverse-transform the whole thing, then extract just that column.

    Args:
        scaled_values:  shape (batch, horizon)  OR  (horizon,)  — scaled 0-1
        scaler:         the fitted MinMaxScaler saved during training
        close_col_idx:  index of the Close column in the feature list
        n_features:     total number of features (= scaler input width)

    Returns:
        Array of same shape as scaled_values, in original dollar prices.
    """
    was_1d = scaled_values.ndim == 1
    if was_1d:
        scaled_values = scaled_values[np.newaxis, :]       # (1, horizon)

    batch, horizon = scaled_values.shape

    # build dummy array: zeros everywhere except the Close column
    dummy = np.zeros((batch * horizon, n_features), dtype=np.float32)
    dummy[:, close_col_idx] = scaled_values.reshape(-1)

    # inverse transform and extract Close column
    inversed = scaler.inverse_transform(dummy)[:, close_col_idx]
    result   = inversed.reshape(batch, horizon)

    return result[0] if was_1d else result


# ── Scaler save/load helpers ──────────────────────────────────────────────────

def save_scaler(scaler: MinMaxScaler, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str) -> MinMaxScaler:
    return joblib.load(path)


# ── 5. Sliding-window Dataset ─────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    Sliding window dataset.

    For each index i:
        X[i] = scaled[i : i+W,  :]                  shape (W, n_features)
        y[i] = scaled[i+W : i+W+N, close_col_idx]   shape (N,)

    W = window_size (lookback), N = forecast_horizon, both set in config.yaml.
    y contains SCALED Close prices — inverse_scale_close() converts them back.
    """

    def __init__(
        self,
        data:             np.ndarray,
        window_size:      int,
        forecast_horizon: int,
        close_col_idx:    int,
    ):
        self.data             = torch.tensor(data, dtype=torch.float32)
        self.window_size      = window_size
        self.forecast_horizon = forecast_horizon
        self.close_col_idx    = close_col_idx
        self.n_samples        = len(data) - window_size - forecast_horizon + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"Not enough data ({len(data)} rows) for "
                f"window_size={window_size} + forecast_horizon={forecast_horizon}. "
                "Use a longer data period or reduce these values."
            )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_end = idx + self.window_size
        y_end = x_end + self.forecast_horizon
        X = self.data[idx  : x_end, :]                        # (W, n_features)
        y = self.data[x_end : y_end, self.close_col_idx]      # (N,)
        return X, y


# ── 6. DataLoaders container ──────────────────────────────────────────────────

@dataclass
class DataLoaders:
    train:          DataLoader
    val:            DataLoader
    test:           DataLoader
    scaler:         MinMaxScaler
    feature_cols:   List[str]
    close_col_idx:  int
    dates_test:     pd.DatetimeIndex   # test set dates, used by visualize.py


# ── 7. Master builder ─────────────────────────────────────────────────────────

def build_dataloaders(
    config:  dict,
    ticker:  str = None,
    horizon: int = None,
) -> DataLoaders:
    """
    End-to-end pipeline. Returns DataLoaders ready for Trainer.

    Split modes:
      val_split > 0  ->  train | val | test   (e.g. 0.70 / 0.15 / 0.15)
      val_split = 0  ->  train+val | test     (e.g. 0.80 / 0.20)
                         val is carved as the last 10% of the train portion
                         so early stopping still works without wasting test data.

    Args:
        config:  full config dict from load_config()
        ticker:  override config.data.ticker  (CLI / run_experiments.py)
        horizon: override config.training.forecast_horizon  (CLI / API)
    """
    ticker       = ticker  or config["data"]["ticker"]
    horizon      = horizon or config["training"]["forecast_horizon"]
    feature_list = config["data"]["features"]
    window_size  = config["data"]["window_size"]
    val_split    = config["training"]["val_split"]
    test_split   = config["training"]["test_split"]
    batch_size   = config["training"]["batch_size"]

    if "Close" not in feature_list:
        raise ValueError(
            "'Close' must be in config.data.features — it is the forecast target."
        )
    close_col_idx = feature_list.index("Close")

    print(f"\n[DataPipeline]  ticker={ticker}  W={window_size}  H={horizon}")

    # steps 1-4
    raw_df  = fetch_ohlcv(ticker, config["data"]["period"])
    feat_df = engineer_features(raw_df, feature_list)

    if val_split > 0:
        # explicit 3-way split: train | val | test
        train_df, val_df, test_df = time_split(feat_df, val_split, test_split)
    else:
        # 80/20 mode: split into train_block | test only
        # then carve last 10% of train_block as val for early stopping
        _, dummy_val, test_df = time_split(feat_df, val_split=0.0, test_split=test_split)
        n_train_block = len(feat_df) - len(test_df)
        n_inner_val   = max(1, int(n_train_block * 0.10))
        train_df      = feat_df.iloc[: n_train_block - n_inner_val]
        val_df        = feat_df.iloc[n_train_block - n_inner_val : n_train_block]
        print(f"  [split]    80/20 mode — inner val carved from train tail")
        print(f"  [split]    train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    train_sc, val_sc, test_sc, scaler = fit_scale(train_df, val_df, test_df)

    # step 5 — datasets
    train_ds = TimeSeriesDataset(train_sc, window_size, horizon, close_col_idx)
    val_ds   = TimeSeriesDataset(val_sc,   window_size, horizon, close_col_idx)
    test_ds  = TimeSeriesDataset(test_sc,  window_size, horizon, close_col_idx)

    print(f"  [dataset]  train={len(train_ds)}  val={len(val_ds)}  "
          f"test={len(test_ds)}  (W={window_size}, H={horizon})")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, drop_last=False)

    return DataLoaders(
        train         = train_loader,
        val           = val_loader,
        test          = test_loader,
        scaler        = scaler,
        feature_cols  = feature_list,
        close_col_idx = close_col_idx,
        dates_test    = test_df.index,
    )