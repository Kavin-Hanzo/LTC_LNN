"""
select_features.py

Standalone Boruta feature selection script — run ONCE per ticker BEFORE training.

What it does:
  1. Fetches historical OHLCV via yfinance
  2. Engineers all features (Open, High, Low, Close, Volume, RSI, MACD, SMA_20)
  3. Does an 80/20 time-ordered split  (uses only train portion for Boruta)
  4. MinMaxScales the train split
  5. Runs Boruta with tunable params  (all in this file, no config.yaml dependency)
  6. Saves result  ->  feature_selection/{TICKER}.json

Output JSON is read by:
  - data_pipeline.py  (training)
  - predictor.py      (inference)

Usage:
    # single ticker
    python select_features.py --ticker AAPL

    # multiple tickers
    python select_features.py --ticker AAPL TSLA GOOGL MSFT

    # tune Boruta params
    python select_features.py --ticker AAPL --n-estimators 200 --max-iter 150 --perc 90

    # use more data
    python select_features.py --ticker AAPL --period 10y

    # force re-run even if JSON already exists
    python select_features.py --ticker AAPL --overwrite

Output:
    feature_selection/
        AAPL.json
        TSLA.json
        ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)

# ── Output directory ──────────────────────────────────────────────────────────

OUTPUT_DIR = "feature_selection"

# ── All features the pipeline supports ───────────────────────────────────────

ALL_FEATURES = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "SMA_20"]
RAW_COLS     = {"Open", "High", "Low", "Close", "Volume"}

# ── Boruta default params (tune here, no config.yaml needed) ─────────────────

DEFAULT_BORUTA_PARAMS = {
    "n_estimators": 100,    # RF trees per iteration — more = more stable, slower
    "max_iter":     100,    # max Boruta iterations
    "perc":         100,    # importance threshold percentile
                            #   100 = strict (must beat MAX of shadow features)
                            #    90 = lenient (must beat 90th percentile of shadows)
    "alpha":        0.05,   # p-value for feature acceptance/rejection
    "max_depth":    5,      # RF tree depth — shallower = faster, less overfit
    "random_state": 42,
}

# ── Feature engineering ───────────────────────────────────────────────────────

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


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build all supported features from raw OHLCV DataFrame."""
    df = pd.DataFrame(index=raw_df.index)
    df["Open"]   = raw_df["Open"].values
    df["High"]   = raw_df["High"].values
    df["Low"]    = raw_df["Low"].values
    df["Close"]  = raw_df["Close"].values
    df["Volume"] = raw_df["Volume"].values
    df["RSI"]    = _rsi(raw_df["Close"]).values
    df["MACD"]   = _macd(raw_df["Close"]).values
    df["SMA_20"] = _sma(raw_df["Close"]).values
    df.dropna(inplace=True)
    return df


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for '{ticker}' (period={period}).")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# ── Core selection ────────────────────────────────────────────────────────────

def run_selection(
    ticker:       str,
    period:       str,
    boruta_params: dict,
) -> dict:
    """
    Full feature selection pipeline for one ticker.

    Returns the result dict that will be saved as JSON.
    """
    try:
        from boruta import BorutaPy
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        print("ERROR: boruta not installed.  Run:  pip install boruta")
        sys.exit(1)

    print(f"\n{'='*58}")
    print(f"  Ticker  : {ticker}")
    print(f"  Period  : {period}")
    print(f"  Params  : {boruta_params}")
    print(f"{'='*58}")

    # ── fetch + engineer
    raw_df  = fetch(ticker, period)
    feat_df = build_features(raw_df)
    print(f"  Rows after feature warm-up : {len(feat_df)}")

    # ── train split only  (80% chronological — no leakage from val/test)
    n_train = int(len(feat_df) * 0.80)
    train_df = feat_df.iloc[:n_train]
    print(f"  Train rows used for Boruta : {len(train_df)}")

    # ── MinMaxScale train
    scaler       = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df.values)  # (N, 8)

    close_col_idx = ALL_FEATURES.index("Close")

    # ── X = all features,  y = Close column  (next-step target)
    X = train_scaled.astype(np.float64)
    y = train_scaled[:, close_col_idx].astype(np.float64)

    # ── RandomForest estimator
    rf = RandomForestRegressor(
        n_estimators = boruta_params["n_estimators"],
        max_depth    = boruta_params["max_depth"],
        n_jobs       = -1,
        random_state = boruta_params["random_state"],
    )

    # ── Boruta
    print(f"\n  Running Boruta  (this may take 1-3 minutes) ...")
    selector = BorutaPy(
        estimator    = rf,
        n_estimators = "auto",
        perc         = boruta_params["perc"],
        alpha        = boruta_params["alpha"],
        max_iter     = boruta_params["max_iter"],
        verbose      = 0,
        random_state = boruta_params["random_state"],
    )
    selector.fit(X, y)

    # ── collect results
    support      = selector.support_        # confirmed important
    support_weak = selector.support_weak_   # tentatively important
    ranking      = selector.ranking_        # 1 = top importance

    # accepted = confirmed + tentative
    accepted_mask = support | support_weak

    # Close ALWAYS kept — it is the forecast target
    accepted_mask[close_col_idx] = True

    selected_features = [f for f, keep in zip(ALL_FEATURES, accepted_mask) if keep]
    rejected_features = [f for f, keep in zip(ALL_FEATURES, accepted_mask) if not keep]
    new_close_idx     = selected_features.index("Close")

    # ── print summary
    print(f"\n  {'Feature':<12}  {'Status':<12}  Rank")
    print(f"  {'-'*36}")
    for feat, rank, accepted in zip(ALL_FEATURES, ranking, accepted_mask):
        status = "SELECTED" if accepted else "rejected"
        forced = "  (forced)" if feat == "Close" and not (support | support_weak)[ALL_FEATURES.index(feat)] else ""
        print(f"  {feat:<12}  {status:<12}  {rank}{forced}")

    print(f"\n  Selected ({len(selected_features)}): {selected_features}")
    if rejected_features:
        print(f"  Rejected ({len(rejected_features)}): {rejected_features}")

    result = {
        "ticker":            ticker,
        "period":            period,
        "generated_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "all_features":      ALL_FEATURES,
        "selected_features": selected_features,
        "rejected_features": rejected_features,
        "close_col_idx":     new_close_idx,
        "n_features_before": len(ALL_FEATURES),
        "n_features_after":  len(selected_features),
        "boruta_params":     boruta_params,
        "ranking": {
            feat: int(rank)
            for feat, rank in zip(ALL_FEATURES, ranking)
        },
        "confirmed":  [f for f, s in zip(ALL_FEATURES, support)      if s],
        "tentative":  [f for f, s in zip(ALL_FEATURES, support_weak) if s],
        "rejected":   [f for f, s in zip(ALL_FEATURES, ~accepted_mask) if s],
    }

    return result


# ── Save ──────────────────────────────────────────────────────────────────────

def save_result(result: dict) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{result['ticker']}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_supported_tickers() -> List[str]:
    """
    Import SUPPORTED_TICKERS from app/main.py.
    This is the single source of truth for which stocks the app supports.
    """
    SUPPORTED_TICKERS = [
        "AAPL", "TSLA", "GOOGL", "MSFT", "AMZN",
        "NVDA", "META", "NFLX",  "AMD",  "INTC",
        "BLK","JPM"
    ]
    # try:
    #     sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    #     from app.main import SUPPORTED_TICKERS
    #     return SUPPORTED_TICKERS
    # except ImportError as e:
    #     print(f"ERROR: Could not import SUPPORTED_TICKERS from app/main.py: {e}")
    #     sys.exit(1)
    return SUPPORTED_TICKERS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Boruta feature selection for one or more stock tickers."
    )

    # ticker source — mutually exclusive
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument(
        "--ticker", nargs="+",
        help="Specific ticker(s) to process  e.g. AAPL TSLA GOOGL"
    )
    ticker_group.add_argument(
        "--all", action="store_true",
        help="Run for ALL tickers defined in app/main.py SUPPORTED_TICKERS"
    )

    parser.add_argument(
        "--period", type=str, default="5y",
        help="yfinance data period  (default: 5y)"
    )
    parser.add_argument(
        "--n-estimators", type=int,
        default=DEFAULT_BORUTA_PARAMS["n_estimators"],
        help=f"RF trees per iteration  (default: {DEFAULT_BORUTA_PARAMS['n_estimators']})"
    )
    parser.add_argument(
        "--max-iter", type=int,
        default=DEFAULT_BORUTA_PARAMS["max_iter"],
        help=f"Max Boruta iterations  (default: {DEFAULT_BORUTA_PARAMS['max_iter']})"
    )
    parser.add_argument(
        "--perc", type=int,
        default=DEFAULT_BORUTA_PARAMS["perc"],
        help=f"Importance threshold percentile  100=strict  90=lenient  "
             f"(default: {DEFAULT_BORUTA_PARAMS['perc']})"
    )
    parser.add_argument(
        "--alpha", type=float,
        default=DEFAULT_BORUTA_PARAMS["alpha"],
        help=f"p-value threshold  (default: {DEFAULT_BORUTA_PARAMS['alpha']})"
    )
    parser.add_argument(
        "--max-depth", type=int,
        default=DEFAULT_BORUTA_PARAMS["max_depth"],
        help=f"RF max tree depth  (default: {DEFAULT_BORUTA_PARAMS['max_depth']})"
    )
    parser.add_argument(
        "--seed", type=int,
        default=DEFAULT_BORUTA_PARAMS["random_state"],
        help=f"Random seed  (default: {DEFAULT_BORUTA_PARAMS['random_state']})"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run even if JSON already exists for this ticker"
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── resolve ticker list
    if args.all:
        tickers = _load_supported_tickers()
        print(f"\n  --all flag: running for all {len(tickers)} supported tickers")
        print(f"  {tickers}")
    else:
        tickers = [t.upper() for t in args.ticker]

    boruta_params = {
        "n_estimators": args.n_estimators,
        "max_iter":     args.max_iter,
        "perc":         args.perc,
        "alpha":        args.alpha,
        "max_depth":    args.max_depth,
        "random_state": args.seed,
    }

    results   = []
    skipped   = []
    failed    = []

    for ticker in tickers:
        out_path = os.path.join(OUTPUT_DIR, f"{ticker}.json")

        if os.path.exists(out_path) and not args.overwrite:
            print(f"\n  [{ticker}]  JSON already exists -> {out_path}")
            print(f"             Use --overwrite to re-run.")
            skipped.append(ticker)
            continue

        try:
            result = run_selection(ticker, args.period, boruta_params)
            path   = save_result(result)
            print(f"\n  Saved -> {path}")
            results.append(result)
        except Exception as e:
            print(f"\n  [{ticker}]  FAILED: {e}")
            failed.append(ticker)

    # ── final summary
    print(f"\n\n{'='*58}")
    print(f"  FEATURE SELECTION SUMMARY")
    print(f"{'='*58}")
    for r in results:
        print(f"  {r['ticker']:<8}  "
              f"{r['n_features_before']} -> {r['n_features_after']} features  "
              f"kept: {r['selected_features']}")
    if skipped:
        print(f"\n  Skipped (already exist): {skipped}")
    if failed:
        print(f"  Failed: {failed}")
    print(f"\n  JSONs saved in:  {OUTPUT_DIR}/")
    print(f"\n  Next step:")
    print(f"    Set boruta.enabled: true in config.yaml")
    print(f"    python train.py --model lstm --ticker <TICKER>")
    print()


if __name__ == "__main__":
    main()