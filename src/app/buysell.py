"""
app/buysell.py

Buy/Sell algorithm on forecasted prices.

Strategy: multiple transactions allowed (greedy valley -> peak).
  - Find every consecutive (local_min, local_max) pair where max > min
  - Record buy @ valley, sell @ peak
  - Total profit = sum of (sell_price - buy_price) for all pairs

This matches the UI in the sample image:
  prices = [100, 180, 260, 310, 40, 535, 695]
  -> Buy@100, Sell@310  profit=210
  -> Buy@40,  Sell@695  profit=655
  -> Total = 865
"""

from __future__ import annotations
from typing import List, Tuple


# ── Core algorithm ────────────────────────────────────────────────────────────

def compute_buysell(
    prices: List[float],
    dates:  List[str],
) -> Tuple[List[dict], float]:
    """
    Greedy valley-to-peak multiple-transaction algorithm.

    Args:
        prices: list of forecasted close prices
        dates:  corresponding date strings ("YYYY-MM-DD")

    Returns:
        buysell_points: list of dicts with keys: type, date, price, profit
        total_profit:   sum of all trade profits (float)
    """
    n = len(prices)
    if n < 2:
        return [], 0.0

    buysell_points = []
    total_profit   = 0.0
    i = 0

    while i < n - 1:

        # ── find next valley (local min / start of upswing)
        while i < n - 1 and prices[i] >= prices[i + 1]:
            i += 1
        buy_idx = i

        # ── find next peak (local max / end of upswing)
        while i < n - 1 and prices[i] <= prices[i + 1]:
            i += 1
        sell_idx = i

        # only record if we actually moved up
        if sell_idx > buy_idx:
            buy_price  = round(prices[buy_idx],  4)
            sell_price = round(prices[sell_idx], 4)
            profit     = round(sell_price - buy_price, 4)

            buysell_points.append({
                "type":   "buy",
                "date":   dates[buy_idx],
                "price":  buy_price,
                "profit": None,
            })
            buysell_points.append({
                "type":   "sell",
                "date":   dates[sell_idx],
                "price":  sell_price,
                "profit": profit,
            })
            total_profit += profit

    return buysell_points, round(total_profit, 4)


# ── Convenience wrapper used by the endpoint ─────────────────────────────────

def build_buysell_response_data(
    forecast: List[dict],
) -> Tuple[List[dict], float]:
    """
    Accepts the forecast list [{date, price}, ...] from predictor.py
    and returns (buysell_points, total_profit).
    """
    prices = [p["price"] for p in forecast]
    dates  = [p["date"]  for p in forecast]
    return compute_buysell(prices, dates)
