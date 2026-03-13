"""
app/model_store.py

Singleton model store.
Loads model.pth + scaler.pkl + meta.json ONCE at server startup.
All endpoints call get_store() to access the cached objects.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from models import build_model


# ── Store dataclass ───────────────────────────────────────────────────────────

@dataclass
class ModelStore:
    model:          nn.Module
    scaler:         MinMaxScaler
    meta:           dict          # contents of meta.json
    device:         torch.device
    arch:           str
    ticker:         str
    window_size:    int
    forecast_horizon: int
    features:       list
    close_col_idx:  int
    input_dim:      int


# ── Module-level singleton ────────────────────────────────────────────────────

_store: Optional[ModelStore] = None


def load_store(artifacts_dir: str = "artifacts/best") -> ModelStore:
    """
    Load model, scaler, and meta from artifacts_dir.
    Called once at FastAPI startup.

    Raises:
        FileNotFoundError if any required artifact is missing.
    """
    global _store

    required = ["model.pth", "scaler.pkl", "meta.json"]
    for fname in required:
        path = os.path.join(artifacts_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing artifact: {path}\n"
                f"Run training then: python promote.py --model <arch>"
            )

    # ── meta
    with open(os.path.join(artifacts_dir, "meta.json")) as f:
        meta = json.load(f)

    arch       = meta["arch"]
    config_obj = _meta_to_config(meta)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── model
    model = build_model(arch, config_obj, input_dim=meta["input_dim"])
    ckpt  = torch.load(
        os.path.join(artifacts_dir, "model.pth"),
        map_location=device,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # ── scaler
    scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))

    _store = ModelStore(
        model            = model,
        scaler           = scaler,
        meta             = meta,
        device           = device,
        arch             = arch,
        ticker           = meta["ticker"],
        window_size      = meta["window_size"],
        forecast_horizon = meta["forecast_horizon"],
        features         = meta["features"],
        close_col_idx    = meta["close_col_idx"],
        input_dim        = meta["input_dim"],
    )

    print(f"[ModelStore]  loaded arch={arch.upper()}  ticker={meta['ticker']}  "
          f"device={device}  horizon={meta['forecast_horizon']}")
    return _store


def get_store() -> ModelStore:
    """Return the loaded store. Raises if load_store() was never called."""
    if _store is None:
        raise RuntimeError(
            "ModelStore not initialised. "
            "Ensure load_store() is called at app startup."
        )
    return _store


def is_loaded() -> bool:
    return _store is not None


# ── Helper ────────────────────────────────────────────────────────────────────

def _meta_to_config(meta: dict) -> dict:
    """
    Reconstruct the minimal config dict that build_model() needs
    from the stored meta.json fields.
    """
    return {
        "training": {
            "forecast_horizon": meta["forecast_horizon"],
        },
        "models": {
            "hidden_size": meta["hidden_size"],
            # safe defaults — only used if arch needs them
            "num_layers": meta.get("num_layers", 2),
            "dropout":    meta.get("dropout",    0.2),
            "lnn": {
                "tau_constant": meta.get("tau_constant", 1.0),
                "ode_unfolds":  meta.get("ode_unfolds",  6),
                "dt":           meta.get("dt",           0.1),
            },
        },
    }
