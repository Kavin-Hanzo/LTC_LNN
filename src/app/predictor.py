"""
app/predictor.py

Two inference modes, selected from meta.json training_mode:

  MIMO (Direct Multi-Horizon):
    Single forward pass -> model outputs all H steps simultaneously.
    Shape: (1, W, F) -> (1, H)
    No recursive drift. Any horizon <= trained horizon is supported by slicing.

  Autoreg (Single-step autoregressive):
    Model outputs 1 step -> feed prediction back -> repeat horizon times.
    Shape: (1, W, F) -> (1, 1) repeated horizon times.
    Any horizon supported but error compounds over long horizons.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple
import json
import os

import numpy as np
import pandas as pd
import torch

from app.model_store import ModelStore
from pipeline.data_pipeline import engineer_features, inverse_scale_close


# ── Main entry point ──────────────────────────────────────────────────────────

def run_inference(
    store:   ModelStore,
    ticker:  str,
    horizon: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Dispatch to MIMO or autoreg inference based on how model was trained.

    Args:
        store:   loaded ModelStore singleton
        ticker:  stock ticker symbol
        horizon: number of trading days to forecast

    Returns:
        historical: list of {date, price}  — last W real Close prices
        forecast:   list of {date, price}  — next horizon predicted prices
    """
    import yfinance as yf

    W          = store.window_size
    n_features = len(store.features)
    close_idx  = store.close_col_idx
    mode       = store.meta.get("training_mode", "mimo")

    # ── 1. fetch raw OHLCV
    fetch_days = W + 60
    raw_df = yf.download(
        ticker, period=f"{fetch_days}d",
        auto_adjust=True, progress=False
    )
    if raw_df.empty:
        raise ValueError(f"No data returned from yfinance for '{ticker}'.")

    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    raw_df = raw_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    raw_df.dropna(inplace=True)
    raw_df.index = pd.to_datetime(raw_df.index)
    raw_df.sort_index(inplace=True)

    # ── 2. engineer ALL original features
    all_features = store.meta.get("all_features") or store.features
    feat_df_full = engineer_features(raw_df, all_features)

    if len(feat_df_full) < W:
        raise ValueError(
            f"Not enough data after feature warm-up. "
            f"Need {W} rows, got {len(feat_df_full)}."
        )

    # ── 3. scale using stored scaler (transform only, never fit)
    window_full       = feat_df_full.iloc[-W:]
    window_dates      = window_full.index
    window_scaled_all = store.scaler.transform(window_full.values).astype(np.float32)

    # ── 4. apply Boruta column selection if used during training
    if store.meta.get("boruta_used", False):
        fs_path = os.path.join("feature_selection", f"{ticker}.json")
        if not os.path.exists(fs_path):
            raise FileNotFoundError(
                f"Model was trained with Boruta but '{fs_path}' not found.\n"
                f"Run: python select_features.py --ticker {ticker}"
            )
        with open(fs_path) as f:
            fs_result = json.load(f)
        all_feat_list = fs_result["all_features"]
        sel_feat_list = fs_result["selected_features"]
        selected_idx  = [all_feat_list.index(f) for f in sel_feat_list]
        window_scaled = window_scaled_all[:, selected_idx]
    else:
        window_scaled = window_scaled_all

    # ── 5. forecast
    original_close_idx = store.meta.get("original_close_col_idx", close_idx)

    if mode == "mimo":
        pred_prices = _infer_mimo(
            store, window_scaled, horizon, close_idx, original_close_idx
        )
    else:
        pred_prices = _infer_autoreg(
            store, window_scaled, horizon, close_idx, original_close_idx,
            n_features
        )

    # ── 6. build output
    last_date      = window_dates[-1].to_pydatetime()
    forecast_dates = _next_trading_days(last_date, horizon)

    hist_close = raw_df["Close"].iloc[-W:].values
    historical = [
        {"date": d.strftime("%Y-%m-%d"), "price": round(float(p), 4)}
        for d, p in zip(window_dates, hist_close)
    ]
    forecast = [
        {"date": d.strftime("%Y-%m-%d"), "price": round(float(p), 4)}
        for d, p in zip(forecast_dates, pred_prices)
    ]

    return historical, forecast


# ── MIMO inference ────────────────────────────────────────────────────────────

def _infer_mimo(
    store:              ModelStore,
    window_scaled:      np.ndarray,
    horizon:            int,
    close_idx:          int,
    original_close_idx: int,
) -> np.ndarray:
    """
    Single forward pass — model outputs all trained_horizon steps at once.
    If requested horizon <= trained_horizon: slice output.
    If requested horizon >  trained_horizon: raise clear error.
    """
    trained_horizon = store.forecast_horizon

    if horizon > trained_horizon:
        raise ValueError(
            f"MIMO model was trained for {trained_horizon} steps. "
            f"Requested horizon={horizon} exceeds this. "
            f"Either retrain with a larger forecast_horizon or use horizon <= {trained_horizon}."
        )

    store.model.eval()
    x = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(store.device)

    with torch.no_grad():
        pred_scaled = store.model(x)              # (1, trained_horizon)

    pred_scaled_np = pred_scaled.cpu().numpy()    # (1, trained_horizon)
    pred_scaled_np = pred_scaled_np[:, :horizon]  # (1, horizon) — slice if needed

    pred_prices = inverse_scale_close(
        pred_scaled_np,
        store.scaler,
        close_idx,
        original_close_idx = original_close_idx,
    )
    return pred_prices[0] if pred_prices.ndim == 2 else pred_prices


# ── Autoreg inference ─────────────────────────────────────────────────────────

def _infer_autoreg(
    store:              ModelStore,
    window_scaled:      np.ndarray,
    horizon:            int,
    close_idx:          int,
    original_close_idx: int,
    n_features:         int,
) -> np.ndarray:
    """
    Roll the model forward horizon steps one step at a time.
    Model output_dim=1. Any horizon supported.
    """
    store.model.eval()
    window = window_scaled.copy()
    preds  = []

    with torch.no_grad():
        for _ in range(horizon):
            x    = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(store.device)
            out  = store.model(x)
            pred = float(out[0, 0].cpu())
            preds.append(pred)

            new_row            = window[-1].copy()
            new_row[close_idx] = pred
            window             = np.vstack([window[1:], new_row])

    pred_scaled_arr = np.array(preds).reshape(1, -1)
    pred_prices = inverse_scale_close(
        pred_scaled_arr,
        store.scaler,
        close_idx,
        original_close_idx = original_close_idx,
    )
    return pred_prices[0] if pred_prices.ndim == 2 else pred_prices


# ── Trading day helper ────────────────────────────────────────────────────────

def _next_trading_days(start: datetime, n: int) -> List[datetime]:
    dates  = []
    cursor = start
    while len(dates) < n:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            dates.append(cursor)
    return dates