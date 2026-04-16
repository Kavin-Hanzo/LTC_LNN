# data/fetcher.py
# Downloads OHLCV from Yahoo Finance and caches to disk.
# Returns {ticker: DataFrame(OHLCV)}.
# No indicator logic. No scaling. Raw prices only.

import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def _cache_path(raw_dir: str, ticker: str, years: int) -> str:
    return os.path.join(raw_dir, f"{ticker}_{years}y.csv")


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end,
                     interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")

    # Flatten MultiIndex columns yfinance sometimes produces
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.ffill(limit=3).dropna()
    return df


def fetch_all(tickers: List[str],
              years: int,
              raw_dir: str,
              force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV for every ticker over `years` years.
    Caches each ticker as a CSV so subsequent runs are instant.

    Returns {ticker: DataFrame} for tickers that loaded successfully.
    """
    os.makedirs(raw_dir, exist_ok=True)
    end   = datetime.today()
    start = end - timedelta(days=int(years * 365.25))
    s_str = start.strftime("%Y-%m-%d")
    e_str = end.strftime("%Y-%m-%d")

    print(f"\n[Fetcher] {s_str} → {e_str}  ({years}y)  tickers={tickers}")

    data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        cache = _cache_path(raw_dir, ticker, years)
        if os.path.exists(cache) and not force_refresh:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            print(f"  [cache]  {ticker:6s}  {len(df):>5} rows")
        else:
            try:
                df = _download(ticker, s_str, e_str)
                df.to_csv(cache)
                print(f"  [live]   {ticker:6s}  {len(df):>5} rows")
            except Exception as exc:
                print(f"  [ERROR]  {ticker:6s}  {exc}")
                continue

        if len(df) < 252:
            print(f"  [SKIP]   {ticker:6s}  only {len(df)} rows — need ≥252")
            continue

        data[ticker] = df

    print(f"[Fetcher] {len(data)}/{len(tickers)} tickers ready\n")
    return data


def aligned_closes(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Return a (T × N) DataFrame of Close prices on a shared date index.
    Used by stock2vec to build the cross-stock return matrix.
    Dates where any ticker is missing are dropped.
    """
    closes = pd.DataFrame({t: df["Close"] for t, df in data.items()})
    return closes.ffill(limit=3).dropna()
