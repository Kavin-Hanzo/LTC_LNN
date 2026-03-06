import yfinance as yf
import pandas as pd
import numpy as np
import torch
import math
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from Utils import apply_boruta

class TimeSeriesDataModule:
    def __init__(self, config):
        self.cfg = config['data']
        self.train_cfg = config['training']
        self.scaler = MinMaxScaler()
        
    def _calculate_dynamic_seq_len(self, total_rows):
        """Logic: Ensures lookback is at least 2x the horizon, but doesn't starve the data."""
        horizon = self.cfg['prediction_horizon']
        
        # Base lookback: 3x the horizon for context
        base_lookback = horizon * 3
        
        # Data-driven constraint: Don't let seq_len exceed 15% of total data
        max_allowed = int(total_rows * 0.15)
        
        # Final Assignment
        dynamic_len = max(20, min(base_lookback, max_allowed, 120))
        print(f"⚙️  Dynamic Windowing: Rows({total_rows}) | Horizon({horizon}) -> Selected SeqLen: {dynamic_len}")
        return dynamic_len

    def _fetch_data(self):
        print(f"📉 Fetching {self.cfg['ticker']} ({self.cfg['interval']})...")
        df = yf.download(
            self.cfg['ticker'], 
            start=self.cfg['start_date'], 
            end=self.cfg['end_date'], 
            interval=self.cfg['interval'],
            progress=False
        )
        if df.empty:
            raise ValueError("Data fetch failed. yfinance limits likely exceeded for this interval.")
        
        # Basic Technical Indicators
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        df.dropna(inplace=True)
        return df

    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_data(self):
        df = self._fetch_data()
        
        # 1. Dynamic Sequence Length Assignment
        seq_len = self._calculate_dynamic_seq_len(len(df))
        horizon = self.cfg['prediction_horizon']
        
        # 2. Feature Selection
        target_col = self.cfg['target_column']
        features = df.drop(columns=[target_col])
        target = df[target_col]

        if self.cfg.get('use_boruta', False):
            selected_cols = apply_boruta(features, target)
            features = features[selected_cols]
            
        # 3. Scaling & Windowing
        full_df = pd.concat([features, target], axis=1)
        scaled_data = self.scaler.fit_transform(full_df)
        
        X, y = [], []
        # Sliding Window Logic: [t-seq_len : t] -> [t+horizon]
        for i in range(len(scaled_data) - seq_len - horizon + 1):
            X.append(scaled_data[i : i + seq_len, :-1]) 
            y.append(scaled_data[i + seq_len + horizon - 1, -1]) 
            
        X, y = np.array(X), np.array(y)
        
        # 4. Split and Loader Creation
        split_idx = int(len(X) * self.cfg['train_split'])
        
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X[:split_idx], dtype=torch.float32), 
                          torch.tensor(y[:split_idx], dtype=torch.float32)),
            batch_size=self.train_cfg['batch_size'], shuffle=False
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X[split_idx:], dtype=torch.float32), 
                          torch.tensor(y[split_idx:], dtype=torch.float32)),
            batch_size=self.train_cfg['batch_size'], shuffle=False
        )
        
        return train_loader, val_loader, self.scaler, X.shape[2], seq_len