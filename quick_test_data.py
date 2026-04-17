# quick_test_data.py
# Quick test script for the data preparation stages of stock2vec.
# Usage example:
#   python quick_test_data.py --stage all --years 5 --tickers AAPL MSFT --lookback 30 --horizon 5

import argparse
import os
from pprint import pprint

from config import CFG
from data.fetcher import fetch_all, aligned_closes
from data.indicators import compute_all, FEATURE_COLS
from data.scaler import chronological_split, scale_splits
from data.dataset import build_datasets
from vectors.stock2vec import build as build_vectors
from visualization import plotter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick test the data preparation pipeline without training models."
    )
    parser.add_argument(
        "--stage",
        choices=["fetch", "indicators", "vectors", "datasets", "all"],
        default="all",
        help="Which data preparation stage to run."
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=CFG.data.tickers,
        help="List of tickers to fetch and prepare."
    )
    parser.add_argument(
        "--years",
        type=int,
        default=CFG.data.history_years[0],
        help="History window in years for raw data fetch."
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download raw ticker data even if cached CSV exists."
    )
    parser.add_argument(
        "--raw-dir",
        default=CFG.data.raw_dir,
        help="Raw OHLCV cache directory."
    )
    parser.add_argument(
        "--vectors-dir",
        default=CFG.vectors.vectors_dir,
        help="Directory to write Stock2Vec CSV outputs."
    )
    parser.add_argument(
        "--plots-dir",
        default=CFG.experiments.plots_dir,
        help="Directory to write Stock2Vec CSV outputs."
    )
    parser.add_argument(
        "--label",
        default="quicktest",
        help="Optional suffix used for saved Stock2Vec files."
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=CFG.vectors.n_components,
        help="Number of PCA components for Stock2Vec."
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=CFG.training.modes["short"]["lookback"],
        help="Lookback length to use when inspecting dataset shapes."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=CFG.training.modes["short"]["horizon"],
        help="Horizon length to use when inspecting dataset shapes."
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=CFG.training.train_ratio,
        help="Training split ratio for the dataset."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=CFG.training.val_ratio,
        help="Validation split ratio for the dataset."
    )
    parser.add_argument(
        "--scaler-method",
        choices=["standard", "minmax"],
        default=CFG.scaler.method,
        help="Scaler method for feature normalization."
    )
    parser.add_argument(
        "--inspect-samples",
        type=int,
        default=3,
        help="Number of dataset samples to inspect after building the datasets."
    )
    return parser.parse_args()


def run_fetch(args):
    print("\n[QuickTest Data] Stage: fetch")
    raw = fetch_all(args.tickers, args.years, args.raw_dir,
                    force_refresh=args.force_refresh)
    print(f"Fetched {len(raw)} tickers:")
    for ticker, df in raw.items():
        print(f"  {ticker:6s}  rows={len(df)}  from={df.index.min().date()} to={df.index.max().date()}")
    return raw


def run_indicators(raw, args):
    print("\n[QuickTest Data] Stage: indicators")
    processed = compute_all(raw, CFG.indicators)
    print(f"Computed indicators for {len(processed)} tickers.")
    for ticker, (df_feat, cols) in processed.items():
        print(f"  {ticker:6s}  rows={len(df_feat)}  features={cols}")
    
    # Plot indicator dashboards
    for ticker, (df_feat, _) in processed.items():
        plotter.indicator_dashboard(df_feat, ticker, args.plots_dir, args.label)
    
    return processed


def run_vectors(raw, args):
    print("\n[QuickTest Data] Stage: vectors")
    aligned = aligned_closes(raw)
    print(f"Aligned closes shape: {aligned.shape}")
    # Build vectors on training split only, to match the full pipeline
    n_tr = int(len(aligned) * args.train_ratio)
    train_aligned = aligned.iloc[:n_tr]
    print(f"Using training split: {len(train_aligned)} rows for PCA")
    vectors, pca, variance = build_vectors(
        train_aligned,
        n_components=args.n_components,
        save_dir=args.vectors_dir,
        label=args.label,
    )
    print(f"Built {len(vectors)} Stock2Vec vectors with {args.n_components} components.")
    print(f"Total variance captured: {variance:.2%}")
    return vectors, aligned


def run_datasets(processed, vectors, args):
    print("\n[QuickTest Data] Stage: datasets")
    scaled_splits = {}
    for ticker, (df, cols) in processed.items():
        tr, va, te = chronological_split(df, args.train_ratio, args.val_ratio)
        scaled_splits[ticker] = scale_splits(tr, va, te, cols, args.scaler_method)
        print(f"  {ticker:6s} split: train={len(tr)} val={len(va)} test={len(te)}")

    from data.dataset import MIMODataset
    ret_idx = FEATURE_COLS.index("close_return")
    tr_ds, va_ds, te_ds = build_datasets(
        processed, vectors, scaled_splits,
        list(processed.keys()), args.lookback, args.horizon, ret_idx,
    )

    print("\nDataset summary:")
    print(f"  train samples = {len(tr_ds)}")
    print(f"  val   samples = {len(va_ds)}")
    print(f"  test  samples = {len(te_ds)}")
    print(f"  lookback      = {args.lookback}")
    print(f"  horizon       = {args.horizon}")

    for name, ds in [("train", tr_ds), ("val", va_ds), ("test", te_ds)]:
        print(f"\n{name.upper()} sample examples:")
        for idx in range(min(args.inspect_samples, len(ds))):
            sample = ds[idx]
            print(f"  idx={idx}  ticker={sample['ticker']}  x_shape={sample['x'].shape} "
                  f"identity_shape={sample['identity'].shape} y_shape={sample['y'].shape} "
                  f"close_ref={sample['close_ref']:.2f}")
    return scaled_splits, (tr_ds, va_ds, te_ds)


def main():
    args = parse_args()
    CFG.make_dirs()

    raw = None
    processed = None
    vectors = None
    aligned = None

    if args.stage in ["fetch", "all"]:
        raw = run_fetch(args)
    if args.stage in ["indicators", "all"]:
        if raw is None:
            raw = run_fetch(args)
        processed = run_indicators(raw, args)
    if args.stage in ["vectors", "all"]:
        if raw is None:
            raw = run_fetch(args)
        vectors, aligned = run_vectors(raw, args)
    if args.stage in ["datasets", "all"]:
        if processed is None:
            if raw is None:
                raw = run_fetch(args)
            processed = run_indicators(raw, args)
        if vectors is None:
            if raw is None:
                raw = run_fetch(args)
            vectors, aligned = run_vectors(raw, args)
        run_datasets(processed, vectors, args)

    print("\n✅ quick_test_data complete.")


if __name__ == "__main__":
    main()
