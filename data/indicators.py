# data/indicators.py
# Computes 7 dimensionless technical features per stock.
#
# RULE: every output is a ratio — no raw price values.
# This removes price-scale bias so $170 AAPL and $875 NVDA
# contribute equally to the loss function.
#
# Final features:
#   close_return   (Close_t / Close_{t-1}) - 1
#   volume_change  (Volume_t / Volume_{t-1}) - 1   clipped ±5
#   bb_pct_b       (Close - BB_lower) / (BB_upper - BB_lower)
#   atr_pct        ATR_14 / Close
#   rsi            RSI_14                           0–100
#   macd_norm      MACDh / Close
#   adx            ADX_14                           0–100

import warnings
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import pandas_ta as ta

warnings.filterwarnings("ignore")

FEATURE_COLS: List[str] = [
    "close_return",
    "volume_change",
    "bb_pct_b",
    "atr_pct",
    "rsi",
    "macd_norm",
    "adx",
]


def compute(df: pd.DataFrame,
            bb_length: int = 20, bb_std: float = 2.0,
            atr_length: int = 14, rsi_length: int = 14,
            macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
            adx_length: int = 14) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute all 7 features on a single OHLCV DataFrame.
    Returns (df_with_features, FEATURE_COLS).
    NaN rows from rolling windows are dropped before returning.
    """
    out = df.copy()

    # close_return
    out["close_return"] = out["Close"].pct_change().clip(-0.30, 0.30)

    # volume_change
    out["volume_change"] = out["Volume"].pct_change().clip(-5.0, 5.0)

    # bb_pct_b  — position inside Bollinger Bands, 0 = lower, 1 = upper
    bb  = ta.bbands(out["Close"], length=bb_length, std=bb_std)
    bbl = [c for c in bb.columns if c.startswith("BBL")][0]
    bbu = [c for c in bb.columns if c.startswith("BBU")][0]
    width = (bb[bbu] - bb[bbl]).replace(0, np.nan)
    out["bb_pct_b"] = (out["Close"] - bb[bbl]) / width

    # atr_pct  — average true range as fraction of price
    atr = ta.atr(out["High"], out["Low"], out["Close"], length=atr_length)
    out["atr_pct"] = (atr / out["Close"]).clip(0, 0.20)

    # rsi
    out["rsi"] = ta.rsi(out["Close"], length=rsi_length)

    # macd_norm  — MACD histogram scaled by close price
    macd   = ta.macd(out["Close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    macdh  = [c for c in macd.columns if c.startswith("MACDh")][0]
    out["macd_norm"] = (macd[macdh] / out["Close"]).clip(-0.10, 0.10)

    # adx
    adx_df = ta.adx(out["High"], out["Low"], out["Close"], length=adx_length)
    adx_c  = [c for c in adx_df.columns if c.startswith("ADX_")][0]
    out["adx"] = adx_df[adx_c]

    out.dropna(subset=FEATURE_COLS, inplace=True)
    out.reset_index(inplace=True)
    if "index" in out.columns:
        out.rename(columns={"index": "Date"}, inplace=True)

    return out, FEATURE_COLS


def compute_all(raw_data: Dict[str, pd.DataFrame],
                cfg) -> Dict[str, Tuple[pd.DataFrame, List[str]]]:
    """
    Run compute() for every ticker in raw_data.
    Returns {ticker: (df_with_features, FEATURE_COLS)}.
    """
    result = {}
    for ticker, df in raw_data.items():
        try:
            df_feat, cols = compute(
                df,
                bb_length=cfg.bb_length, bb_std=cfg.bb_std,
                atr_length=cfg.atr_length, rsi_length=cfg.rsi_length,
                macd_fast=cfg.macd_fast, macd_slow=cfg.macd_slow,
                macd_signal=cfg.macd_signal, adx_length=cfg.adx_length,
            )
            result[ticker] = (df_feat, cols)
            print(f"  [Indicators] {ticker:6s}  {len(df_feat):>5} rows")
        except Exception as e:
            print(f"  [Indicators] {ticker:6s}  ERROR: {e}")
    return result
