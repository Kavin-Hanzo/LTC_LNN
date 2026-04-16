# evaluation/metrics.py
# All prediction quality metrics — return space AND price space.
#
# Reported per stock and aggregated:
#   mae_ret       MAE on normalised returns
#   rmse_ret      RMSE on normalised returns
#   mape_pct      MAPE (%) after inverse-transform to price
#   raw_dev_mean  Mean |predicted_price - actual_price|  in $
#   raw_dev_max   Worst-case absolute price error in $
#   dir_acc       % of steps where direction (up/down) was correct

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score as _sklearn_r2


# ── Scalar helpers ────────────────────────────────────────────────
def mae(t, p):
    return float(np.mean(np.abs(t - p)))

def rmse(t, p):
    return float(np.sqrt(np.mean((t - p) ** 2)))

def mape(t, p, eps=1e-8):
    return float(np.mean(np.abs((t - p) / (np.abs(t) + eps))) * 100)

def direction_accuracy(t, p):
    return float(np.mean(np.sign(t) == np.sign(p)) * 100)

def r2(t: np.ndarray, p: np.ndarray) -> float:
    """
    R² (coefficient of determination) computed on flattened arrays.
    Interpretation:
        1.0  perfect prediction
        0.0  model predicts no better than the mean of y_true
        < 0  model is worse than just predicting the mean
    Computed in RETURN space (same space as MAE / RMSE).
    Also reported in PRICE space via r2_price.
    """
    return float(_sklearn_r2(t.ravel(), p.ravel()))


# ── Price reconstruction ──────────────────────────────────────────
def returns_to_prices(returns: np.ndarray,
                      close_refs: np.ndarray) -> np.ndarray:
    """
    returns    : (N, H)  normalised returns
    close_refs : (N,)    last known close per sample
    Output     : (N, H)  reconstructed price trajectory
    """
    prices = np.zeros_like(returns)
    cum    = np.ones(len(close_refs))
    for h in range(returns.shape[1]):
        cum          = cum * (1.0 + returns[:, h])
        prices[:, h] = close_refs * cum
    return prices


# ── Full metric suite ─────────────────────────────────────────────
def compute(y_true: np.ndarray,
            y_pred: np.ndarray,
            close_refs: np.ndarray,
            ticker: str = "") -> Dict:
    mae_r   = mae(y_true.ravel(),  y_pred.ravel())
    rmse_r  = rmse(y_true.ravel(), y_pred.ravel())
    r2_r    = r2(y_true, y_pred)

    pt  = returns_to_prices(y_true, close_refs)
    pp  = returns_to_prices(y_pred, close_refs)
    raw = np.abs(pt - pp)
    r2_p = r2(pt, pp)

    return {
        "ticker":        ticker,
        "mae_ret":       mae_r,
        "rmse_ret":      rmse_r,
        "r2_ret":        r2_r,          # R² in return space
        "r2_price":      r2_p,          # R² in reconstructed price space
        "mape_pct":      mape(pt.ravel(), pp.ravel()),
        "raw_dev_mean":  float(raw.mean()),
        "raw_dev_max":   float(raw.max()),
        "dir_acc":       direction_accuracy(y_true.ravel(), y_pred.ravel()),
        "per_horizon_raw_dev": raw.mean(axis=0).tolist(),
    }


def aggregate(per_stock: Dict[str, Dict]) -> Dict[str, float]:
    keys = ["mae_ret", "rmse_ret", "r2_ret", "r2_price",
            "mape_pct", "raw_dev_mean", "raw_dev_max", "dir_acc"]
    return {k: float(np.mean([v[k] for v in per_stock.values()])) for k in keys}


def metrics_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    results: {model_name: {ticker: metrics_dict}}
    Returns a MultiIndex DataFrame (model, ticker) × metric columns.
    """
    rows = []
    for model, per_stock in results.items():
        for ticker, m in per_stock.items():
            rows.append({
                "model":          model,
                "ticker":         ticker,
                "MAE(ret)":       round(m["mae_ret"],      6),
                "RMSE(ret)":      round(m["rmse_ret"],     6),
                "R2(ret)":        round(m["r2_ret"],       4),
                "R2(price)":      round(m["r2_price"],     4),
                "MAPE(%)":        round(m["mape_pct"],     2),
                "RawDev_mean($)": round(m["raw_dev_mean"], 2),
                "RawDev_max($)":  round(m["raw_dev_max"],  2),
                "DirAcc(%)":      round(m["dir_acc"],      2),
            })
    return pd.DataFrame(rows).set_index(["model", "ticker"])


def improvement_table(base: Dict[str, Dict],
                      ours: Dict[str, Dict]) -> pd.DataFrame:
    """% improvement Baseline → Stock2Vec.  Positive = our model is better."""
    rows = []
    for ticker in base:
        if ticker not in ours:
            continue
        b, o = base[ticker], ours[ticker]
        def imp(bv, ov):
            return round((bv - ov) / (abs(bv) + 1e-12) * 100, 2)
        rows.append({
            "ticker":           ticker,
            "RMSE_imp(%)":      imp(b["rmse_ret"],     o["rmse_ret"]),
            "MAE_imp(%)":       imp(b["mae_ret"],      o["mae_ret"]),
            "R2_ret_delta":     round(o["r2_ret"]   - b["r2_ret"],   4),
            "R2_price_delta":   round(o["r2_price"] - b["r2_price"], 4),
            "MAPE_imp(%)":      imp(b["mape_pct"],     o["mape_pct"]),
            "RawDev_imp(%)":    imp(b["raw_dev_mean"], o["raw_dev_mean"]),
            "DirAcc_delta(%)":  round(o["dir_acc"] - b["dir_acc"], 2),
        })
    return pd.DataFrame(rows).set_index("ticker")
