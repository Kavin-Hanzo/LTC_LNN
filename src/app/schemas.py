"""
app/schemas.py

All Pydantic request/response models for the FastAPI endpoints.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    architecture:  Optional[str] = None
    ticker:        Optional[str] = None
    horizon:       Optional[int] = None
    device:        str


# ── Tickers ───────────────────────────────────────────────────────────────────

class TickersResponse(BaseModel):
    tickers: List[str]


# ── Predict ───────────────────────────────────────────────────────────────────

class PricePoint(BaseModel):
    date:  str           # "YYYY-MM-DD"
    price: float


class PredictResponse(BaseModel):
    ticker:      str
    model_used:  str
    horizon:     int
    forecast:    List[PricePoint]


# ── BuySell ───────────────────────────────────────────────────────────────────

class BuySellPoint(BaseModel):
    type:   str          # "buy" or "sell"
    date:   str          # "YYYY-MM-DD"
    price:  float
    profit: Optional[float] = None   # filled only on "sell" points


class BuySellResponse(BaseModel):
    ticker:         str
    model_used:     str
    horizon:        int

    historical:     List[PricePoint]   # last W days of real Close prices
    forecast:       List[PricePoint]   # next N days predicted Close prices

    buysell_points: List[BuySellPoint]
    total_profit:   float
