"""
app/main.py

FastAPI inference server.

Endpoints:
    GET /health
    GET /tickers
    GET /predict?ticker=AAPL&horizon=30
    GET /buysell?ticker=AAPL&horizon=30

Run:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.model_store import load_store, get_store, is_loaded
from app.predictor   import run_inference
from app.buysell     import build_buysell_response_data
from app.schemas     import (
    HealthResponse,
    TickersResponse,
    PredictResponse,
    BuySellResponse,
    PricePoint,
    BuySellPoint,
)


# ── Supported tickers ─────────────────────────────────────────────────────────
# Extend this list freely — any yfinance-valid ticker works
SUPPORTED_TICKERS = [
    "AAPL", "TSLA", "GOOGL", "MSFT", "AMZN",
    "NVDA", "META", "NFLX",  "AMD",  "INTC",
    "BLK","JPM"
]

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts/best")


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    print(f"\n[startup]  loading model from {ARTIFACTS_DIR} ...")
    try:
        load_store(ARTIFACTS_DIR)
        print("[startup]  model ready.\n")
    except FileNotFoundError as e:
        print(f"[startup]  WARNING: {e}")
        print("[startup]  Server will start but /predict and /buysell will fail "
              "until a model is promoted.\n")
    yield
    # shutdown (nothing to clean up)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Stock Forecast API",
    description = "Forecast stock prices with RNN / LSTM / GRU / LNN models.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """Server liveness + model status."""
    import torch
    device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if not is_loaded():
        return HealthResponse(
            status       = "degraded",
            model_loaded = False,
            device       = device,
        )

    store = get_store()
    return HealthResponse(
        status       = "ok",
        model_loaded = True,
        architecture = store.arch,
        ticker       = store.ticker,
        horizon      = store.forecast_horizon,
        device       = device,
    )


# ── /tickers ──────────────────────────────────────────────────────────────────

@app.get("/tickers", response_model=TickersResponse, tags=["meta"])
def tickers():
    """Return list of supported stock tickers."""
    return TickersResponse(tickers=SUPPORTED_TICKERS)


# ── /predict ──────────────────────────────────────────────────────────────────

@app.get("/predict", response_model=PredictResponse, tags=["inference"])
def predict(
    ticker:  str = Query(default=None, description="Stock ticker, e.g. AAPL"),
    horizon: int = Query(default=None, ge=1, le=365,
                         description="Forecast horizon in days"),
):
    """
    Forecast the next `horizon` trading days of Close price for `ticker`.

    - Fetches the last W real trading days via yfinance
    - Runs the promoted model
    - Returns forecasted prices with dates
    """
    _require_model()

    store    = get_store()
    ticker   = (ticker  or store.ticker).upper()
    horizon  = horizon  or store.forecast_horizon

    _validate_ticker(ticker)
    _validate_horizon(horizon, store)

    try:
        historical, forecast = run_inference(store, ticker, horizon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(
        ticker     = ticker,
        model_used = store.arch,
        horizon    = horizon,
        forecast   = [PricePoint(**p) for p in forecast],
    )


# ── /buysell ──────────────────────────────────────────────────────────────────

@app.get("/buysell", response_model=BuySellResponse, tags=["inference"])
def buysell(
    ticker:  str = Query(default=None, description="Stock ticker, e.g. AAPL"),
    horizon: int = Query(default=None, ge=1, le=365,
                         description="Forecast horizon in days"),
):
    """
    Forecast prices + compute optimal buy/sell points and total profit.

    Returns:
    - historical: last W days of real Close prices
    - forecast:   next horizon days of predicted prices
    - buysell_points: list of {type, date, price, profit}
    - total_profit: sum of all trade profits
    """
    _require_model()

    store   = get_store()
    ticker  = (ticker  or store.ticker).upper()
    horizon = horizon  or store.forecast_horizon

    _validate_ticker(ticker)
    _validate_horizon(horizon, store)

    try:
        historical, forecast = run_inference(store, ticker, horizon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    buysell_points, total_profit = build_buysell_response_data(forecast)

    return BuySellResponse(
        ticker          = ticker,
        model_used      = store.arch,
        horizon         = horizon,
        historical      = [PricePoint(**p) for p in historical],
        forecast        = [PricePoint(**p) for p in forecast],
        buysell_points  = [BuySellPoint(**pt) for pt in buysell_points],
        total_profit    = total_profit,
    )


# ── Guards ────────────────────────────────────────────────────────────────────

def _require_model():
    if not is_loaded():
        raise HTTPException(
            status_code = 503,
            detail      = (
                "No model loaded. "
                "Run training then: python promote.py --model <arch>"
            ),
        )


def _validate_ticker(ticker: str):
    if ticker not in SUPPORTED_TICKERS:
        raise HTTPException(
            status_code = 400,
            detail      = (
                f"Ticker '{ticker}' not supported. "
                f"Supported: {SUPPORTED_TICKERS}"
            ),
        )


def _validate_horizon(horizon: int, store):
    # Autoregressive model rolls forward step-by-step at inference time,
    # so any horizon value is valid regardless of what the model was trained with.
    # We only enforce a reasonable upper bound to protect server resources.
    MAX_HORIZON = 365
    if horizon > MAX_HORIZON:
        raise HTTPException(
            status_code = 400,
            detail      = f"horizon={horizon} exceeds maximum allowed value of {MAX_HORIZON}."
        )
    