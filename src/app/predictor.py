"""
app/predictor.py

Inference logic:
  1. Fetch the last W trading days of real Close prices via yfinance
  2. Engineer features + scale using the stored scaler
  3. Run model forward pass
  4. Inverse-scale output back to dollar prices
  5. Return (historical_prices, forecast_dates, forecast_prices)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from app.model_store import ModelStore
from pipeline.data_pipeline import engineer_features, inverse_scale_close


def run_inference(
    store:   ModelStore,
    ticker:  str,
    horizon: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Fetch live data, run model, return historical + forecast price lists.

    Args:
        store:   loaded ModelStore singleton
        ticker:  stock ticker symbol
        horizon: number of days to forecast

    Returns:
        historical: list of {date, price}  — last window_size real prices
        forecast:   list of {date, price}  — next horizon predicted prices
    """
    import yfinance as yf

    W = store.window_size

    # ── fetch enough history to build W-day window after feature engineering
    # features like SMA_20, RSI need ~30 extra rows to warm up
    fetch_days   = W + 60
    period_str   = f"{fetch_days}d"
    raw_df       = yf.download(ticker, period=period_str, auto_adjust=True, progress=False)

    if raw_df.empty:
        raise ValueError(f"No data returned from yfinance for ticker '{ticker}'.")

    # flatten multi-level columns
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)

    raw_df = raw_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    raw_df.dropna(inplace=True)
    raw_df.index = pd.to_datetime(raw_df.index)
    raw_df.sort_index(inplace=True)

    # ── feature engineering
    feat_df = engineer_features(raw_df, store.features)

    # ── take last W rows as the input window
    if len(feat_df) < W:
        raise ValueError(
            f"Not enough data after feature engineering. "
            f"Need {W} rows, got {len(feat_df)}."
        )

    window_df   = feat_df.iloc[-W:]                     # (W, n_features)
    window_dates = window_df.index                       # DatetimeIndex

    # ── scale using stored scaler  (transform only, never fit)
    window_scaled = store.scaler.transform(window_df.values)   # (W, n_features)

    # ── model forward pass
    x = torch.tensor(window_scaled, dtype=torch.float32)
    x = x.unsqueeze(0).to(store.device)                 # (1, W, n_features)

    with torch.no_grad():
        pred_scaled = store.model(x)                    # (1, horizon_trained)

    pred_scaled_np = pred_scaled.cpu().numpy()          # (1, horizon_trained)

    # ── clip to requested horizon (may differ from trained horizon)
    pred_scaled_np = pred_scaled_np[:, :horizon]        # (1, horizon)

    # ── inverse scale -> dollar prices
    n_features   = len(store.features)
    pred_prices  = inverse_scale_close(
        pred_scaled_np, store.scaler, store.close_col_idx, n_features
    )                                                    # (1, horizon) or (horizon,)

    if pred_prices.ndim == 2:
        pred_prices = pred_prices[0]                    # (horizon,)

    # ── build forecast dates (skip weekends naively)
    last_date      = window_dates[-1].to_pydatetime()
    forecast_dates = _next_trading_days(last_date, horizon)

    # ── historical window prices (raw Close, not scaled)
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


# ── Trading day helper ────────────────────────────────────────────────────────

def _next_trading_days(start: datetime, n: int) -> List[datetime]:
    """
    Return next n weekday dates after start.
    Simple approximation — does not account for public holidays.
    """
    dates  = []
    cursor = start
    while len(dates) < n:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:    # Mon-Fri
            dates.append(cursor)
    return dates
