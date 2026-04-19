# config.py
# Single source of truth. Every module imports CFG from here.
# To change any behaviour — edit ONLY this file.

from dataclasses import dataclass, field
from typing import List, Dict
import os


@dataclass
class DataConfig:
    tickers:       List[str] = field(default_factory=lambda: [
                       "AAPL", "MSFT", "GOOGL", "META", "IBM", "NVDA"
                   ])
    history_years: List[int] = field(default_factory=lambda: [10]) #tocheck
    interval:      str  = "1d"
    raw_dir:       str  = "outputs/raw"


@dataclass
class IndicatorConfig:
    bb_length:   int   = 20
    bb_std:      float = 2.0
    atr_length:  int   = 14
    rsi_length:  int   = 14
    macd_fast:   int   = 12
    macd_slow:   int   = 26
    macd_signal: int   = 9
    adx_length:  int   = 14


@dataclass
class ScalerConfig:
    # "standard" = Z-score  |  "minmax" = 0-to-1
    method: str = "minmax"                         #tocheck


@dataclass
class VectorConfig:
    n_components:    int   = 4
    vectors_dir:     str   = "outputs/vectors"
    good_variance:   float = 0.90
    fair_variance:   float = 0.75
    good_separation: float = 1.50
    fair_separation: float = 1.10


@dataclass
class LNNConfig:
    # Parameters for the Liquid Neural Network (LTC-based ODE cell).
    # Only used when ModelConfig.arch == "lnn".
    tau_constant: float = 3.0   # membrane time-constant τ (higher = slower decay)
    dt:           float = 0.3   # Euler integration step   (smaller = more precise)
    ode_unfolds:  int   = 1     # Euler steps per time-step (higher = finer approx)


@dataclass
class ModelConfig:
    # ── Architecture selection ─────────────────────────────────────
    # "lstm" : Long Short-Term Memory   — best for long-range dependencies
    # "gru"  : Gated Recurrent Unit     — fewer params, trains faster
    # "rnn"  : Vanilla RNN              — simplest, good weak baseline
    # "lnn"  : Liquid Neural Network    — ODE-based continuous-time dynamics
    arch: str = "lnn"    #tocheck

    hidden_size:    int   = 64
    num_layers:     int   = 2       # 2=light  3=medium  4=deep
    dropout:        float = 0.20
    use_batch_norm: bool  = False
    use_identity:   bool  = True    # False = baseline, True = Stock2Vec
    identity_dim:   int   = 4       # must match VectorConfig.n_components

    # LNN-specific sub-config (ignored when arch != "lnn")
    lnn: LNNConfig = field(default_factory=LNNConfig)


@dataclass
class TrainingConfig:
    # Each mode: {lookback, horizon}
    # SHORT: 30-day context → predict next 5 days  (1 trading week)
    # LONG : 120-day context → predict next 21 days (1 trading month)
    modes: Dict = field(default_factory=lambda: {        
        "week": {"lookback": 30,  "horizon": 5},      #tocheck
        "mon":  {"lookback": 180, "horizon": 21},
        "3mon":  {"lookback": 450, "horizon": 63},
        "6mon":  {"lookback": 900, "horizon": 125}
    })
    train_ratio:  float = 0.70
    val_ratio:    float = 0.15
    #test_ratio  = 1 - train_ratio - val_ratio  #(always chronological)
    batch_size:   int   = 64
    epochs:       int   = 100
    lr:           float = 1e-3
    weight_decay: float = 1e-5
    patience:     int   = 15
    seed:         int   = 42
    device:       str   = "auto"   # "auto" | "cpu" | "cuda" | "mps"
    models_dir:   str   = "outputs/models"


@dataclass
class ExperimentConfig:
    results_dir: str = "outputs/results"
    plots_dir:   str = "outputs/plots"
    plot_dpi:    int = 130


@dataclass
class Config:
    data:        DataConfig       = field(default_factory=DataConfig)
    indicators:  IndicatorConfig  = field(default_factory=IndicatorConfig)
    scaler:      ScalerConfig     = field(default_factory=ScalerConfig)
    vectors:     VectorConfig     = field(default_factory=VectorConfig)
    model:       ModelConfig      = field(default_factory=ModelConfig)
    training:    TrainingConfig   = field(default_factory=TrainingConfig)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)

    def resolve_device(self):
        import torch
        if self.training.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.training.device)

    def make_dirs(self):
        for d in [
            self.data.raw_dir,
            self.vectors.vectors_dir,
            self.training.models_dir,
            self.experiments.results_dir,
            self.experiments.plots_dir,
        ]:
            os.makedirs(d, exist_ok=True)


# Singleton imported by every module
CFG = Config()
