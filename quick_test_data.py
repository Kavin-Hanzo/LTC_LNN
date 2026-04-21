import os
import torch
import numpy as np
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple

# Importing from your existing local modules
from config import CFG
from data.fetcher import fetch_all, aligned_closes
from data.indicators import compute_all, FEATURE_COLS
from data.scaler import chronological_split, scale_splits, WalkForwardScaler
from data.dataset import build_datasets, collate
from vectors.stock2vec import build as build_vectors

class StockDataModule(L.LightningDataModule):
    """
    Decoupled Data Pipeline for Multi-Stock Forecasting.
    Handles: Fetching -> Indicators -> Splitting -> Scaling -> Identity Vectors -> Datasets.
    """
    def __init__(self, 
                 tickers: List[str] = None, 
                 years: int = None, 
                 lookback: int = 30, 
                 horizon: int = 5, 
                 batch_size: int = 256,
                 num_workers: int = 4):
        super().__init__()
        self.tickers = tickers or CFG.data.tickers
        self.years = years or CFG.data.history_years[0]
        self.lookback = lookback
        self.horizon = horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Persistence for Scaling and Datasets
        self.scalers: Dict[str, WalkForwardScaler] = {}
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.identity_dim = 0

    def prepare_data(self):
        """
        Stage 1: Hardware-agnostic data download.
        Called once from the main process.
        """
        fetch_all(self.tickers, self.years, CFG.data.raw_dir)

    def setup(self, stage: Optional[str] = None):
        """
        Stage 2: Feature Engineering, Splitting, and Scaling.
        Called on every GPU/process.
        """
        # 1. Load Raw Data
        raw = fetch_all(self.tickers, self.years, CFG.data.raw_dir)
        
        # 2. Extract Aligned Closes (for Stock2Vec Correlation Matrix)
        # We must align them early to ensure they share the same date index
        aligned = aligned_closes(raw)
        
        # 3. Compute Technical Indicators (Returns, RSI, MACD, etc.)
        processed = compute_all(raw, CFG)
        
        # 4. CHRONOLOGICAL SPLITTING & SCALING (The "Scaling Process")
        # We iterate through each stock to ensure per-stock normalization
        scaled_splits = {}
        for ticker in self.tickers:
            df, _ = processed[ticker]
            
            # Split into raw DataFrames (e.g., 70/15/15)
            tr_df, va_df, te_df = chronological_split(df)
            
            # FIT scaler on Train, then TRANSFORM Val/Test
            # This is the core 'Walk-Forward' logic from scaler.py
            X_tr, X_va, X_te, scaler = scale_splits(
                tr_df, va_df, te_df, FEATURE_COLS, method=CFG.scaler.method
            )
            
            # Store scaled arrays and the scaler object for later inverse transforms
            scaled_splits[ticker] = (X_tr, X_va, X_te, scaler)
            self.scalers[ticker] = scaler

        # 5. IDENTITY VECTORS (Stock2Vec)
        # We only use the TRAINING portion of the aligned closes to build embeddings
        n_train_samples = len(scaled_splits[self.tickers[0]][0])
        train_aligned = aligned.iloc[:n_train_samples]
        
        vectors, _, _ = build_vectors(train_aligned, CFG.vectors.n_components)
        self.identity_dim = CFG.vectors.n_components

        # 6. INSTANTIATE DATASETS
        # ret_idx identifies which column in FEATURE_COLS is the target (close_return)
        ret_idx = FEATURE_COLS.index("close_return")
        
        self.train_ds, self.val_ds, self.test_ds = build_datasets(
            processed, vectors, scaled_splits, self.tickers, 
            self.lookback, self.horizon, ret_idx
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True
        )

# --- EXECUTION SCRIPT ---

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Lightning DataModule Quick Test")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--years", type=int, default=5)
    args = parser.parse_args()

    # Ensure output directories exist
    CFG.make_dirs()

    # Initialize the DataModule
    dm = StockDataModule(
        years=args.years, 
        batch_size=args.batch_size
    )
    
    print(f"[DM] Initializing pipeline for {len(dm.tickers)} tickers...")
    dm.prepare_data()
    dm.setup()
    
    # Validation Check
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    
    print("\n" + "="*30)
    print("DATA PIPELINE SUCCESS")
    print("="*30)
    print(f"X (Inputs) shape:      {batch['x'].shape}")          # [Batch, Seq, Feat]
    print(f"Y (Targets) shape:     {batch['y'].shape}")          # [Batch, Horizon]
    print(f"Identity shape:        {batch['identity'].shape}")   # [Batch, PCA_Dim]
    print(f"Number of Scalers:     {len(dm.scalers)}")
    print(f"Example Ticker:        {batch['ticker'][0]}")
    print("="*30)
