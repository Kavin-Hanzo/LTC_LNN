"""
app/predictor.py

Autoregressive rolling inference:
  1. Fetch the last W trading days of real OHLCV via yfinance
  2. Engineer features + MinMaxScale using the stored scaler
  3. Roll the model forward `horizon` times:
       for each step:
           pred_scaled = model(window)          # (1, W, F) -> (1, 1)
           append pred_scaled to forecast list
           slide window: drop oldest row, append new row
               - Close column  <- pred_scaled (model's own prediction)
               - OHLC columns  <- carry forward last known values (approx)
               - Volume        <- carry forward last known value
               - RSI/MACD/SMA  <- carry forward last known values (approx)
  4. Inverse-scale the full forecast list -> dollar prices
  5. Return (historical, forecast)

Why autoregressive?
  The model output_dim=1 (single step), so any horizon is supported at
  inference time without retraining. Errors compound over long horizons
  but this is inherent to multi-step forecasting regardless of approach.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from app.model_store import ModelStore
from pipeline.data_pipeline import engineer_features, inverse_scale_close


# ── Main inference entry point ────────────────────────────────────────────────

def run_inference(
    store:   ModelStore,
    ticker:  str,
    horizon: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Fetch live data, roll model forward `horizon` steps, return results.

    Args:
        store:   loaded ModelStore singleton
        ticker:  stock ticker symbol
        horizon: number of trading days to forecast (any value, no retraining needed)

    Returns:
        historical: list of {date, price}  — last W real Close prices
        forecast:   list of {date, price}  — next horizon predicted Close prices
    """
    import yfinance as yf

    W          = store.window_size
    n_features = len(store.features)
    close_idx  = store.close_col_idx

    # ── 1. fetch raw OHLCV — extra rows for indicator warm-up
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
    # scaler was fitted on the full original feature set — must transform all columns
    # then drop Boruta-rejected ones AFTER scaling
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
    # shape: (W, n_original_features)

    # ── 4. apply Boruta column selection if used during training
    # reads feature_selection/{ticker}.json — does NOT re-run Boruta algorithm
    if store.meta.get("boruta_used", False):
        import json as _json
        fs_path = os.path.join("feature_selection", f"{ticker}.json")
        if not os.path.exists(fs_path):
            raise FileNotFoundError(
                f"Model was trained with Boruta but no feature selection JSON "
                f"found at '{fs_path}'.\n"
                f"Run:  python select_features.py --ticker {ticker}"
            )
        with open(fs_path) as f:
            fs_result = _json.load(f)

        all_feat_list = fs_result["all_features"]
        sel_feat_list = fs_result["selected_features"]
        selected_idx  = [all_feat_list.index(f) for f in sel_feat_list]
        window_scaled = window_scaled_all[:, selected_idx]
        # shape: (W, n_selected_features) — matches model input layer
    else:
        window_scaled = window_scaled_all
        # shape: (W, n_original_features)

    # ── 4. autoregressive roll
    pred_scaled_list = _roll_forward(
        model         = store.model,
        window_scaled = window_scaled,
        horizon       = horizon,
        close_idx     = close_idx,
        n_features    = n_features,
        device        = store.device,
    )

    # ── 5. inverse scale -> dollar prices
    # scaler was fitted on original features — pass original_close_col_idx
    original_close_idx = store.meta.get("original_close_col_idx", close_idx)
    pred_scaled_arr    = np.array(pred_scaled_list).reshape(1, -1)  # (1, horizon)
    pred_prices        = inverse_scale_close(
        pred_scaled_arr,
        store.scaler,
        close_idx,
        original_close_idx = original_close_idx,
    )
    if pred_prices.ndim == 2:
        pred_prices = pred_prices[0]                                 # (horizon,)

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


# ── Autoregressive rolling core ───────────────────────────────────────────────

def _roll_forward(
    model:         torch.nn.Module,
    window_scaled: np.ndarray,
    horizon:       int,
    close_idx:     int,
    n_features:    int,
    device:        torch.device,
) -> List[float]:
    """
    Roll the model forward `horizon` steps autoregressively.

    At each step:
      - Feed the current W-row window to the model  -> scaled next-step Close
      - Slide the window: drop row[0], append a new row where
          Close column  = model's prediction
          all others    = carried forward from last real row (frozen approximation)

    The "carry forward" approximation for OHLC/Volume/indicators is the
    standard approach for autoregressive forecasting — we only have reliable
    signal for Close. RSI/MACD/SMA will drift but remain in a sensible range
    because they're bounded by the scaler.

    Returns:
        list of horizon scaled Close predictions (floats)
    """
    model.eval()
    window = window_scaled.copy()           # (W, n_features) — don't mutate original
    preds  = []

    with torch.no_grad():
        for _ in range(horizon):
            # (1, W, n_features) forward pass
            x    = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            out  = model(x)                 # (1, 1)  — single step output
            pred = float(out[0, 0].cpu())
            preds.append(pred)

            # ── slide window forward by 1 row
            # new row = last row of window, with Close replaced by prediction
            new_row              = window[-1].copy()     # carry forward all features
            new_row[close_idx]   = pred                  # update Close only
            window               = np.vstack([window[1:], new_row])  # drop oldest

    return preds


# ── Trading day helper ────────────────────────────────────────────────────────

def _next_trading_days(start: datetime, n: int) -> List[datetime]:
    """
    Return next n weekday dates after start.
    Simple Mon-Fri approximation — does not account for public holidays.
    """
    dates  = []
    cursor = start
    while len(dates) < n:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            dates.append(cursor)
    return dates