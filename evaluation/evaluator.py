# evaluation/evaluator.py
# Runs inference on a test dataset, groups predictions by ticker,
# and returns per-stock metrics + raw arrays for plotting.

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import MIMODataset, collate
from evaluation.metrics import compute, aggregate


@torch.no_grad()
def run(model,
        dataset: MIMODataset,
        device:  torch.device,
        batch_size: int = 256) -> Tuple[Dict, Dict, Dict]:
    """
    Returns
    -------
    per_stock   : {ticker: metrics_dict}
    aggregated  : {metric: float}
    predictions : {ticker: {"y_true", "y_pred", "close_refs"}}
    """
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=collate, num_workers=0)

    buckets = defaultdict(lambda: {"y_true": [], "y_pred": [], "close_refs": []})

    for batch in loader:
        x   = batch["x"].to(device)
        idn = batch["identity"].to(device) if model.use_identity else None
        y_true    = batch["y"].cpu().numpy()
        close_ref = batch["close_ref"].cpu().numpy()
        tickers   = batch["ticker"]
        y_pred    = model(x, idn).cpu().numpy()

        for i, t in enumerate(tickers):
            buckets[t]["y_true"].append(y_true[i])
            buckets[t]["y_pred"].append(y_pred[i])
            buckets[t]["close_refs"].append(close_ref[i])

    predictions = {
        t: {
            "y_true":     np.stack(d["y_true"]),
            "y_pred":     np.stack(d["y_pred"]),
            "close_refs": np.array(d["close_refs"]),
        }
        for t, d in buckets.items()
    }

    per_stock = {
        t: compute(arrays["y_true"], arrays["y_pred"],
                   arrays["close_refs"], ticker=t)
        for t, arrays in predictions.items()
    }
    agg = aggregate(per_stock)

    # Print summary table
    print(f"\n  {'Ticker':8s}  {'RMSE':>10}  {'MAE':>10}  "
          f"{'R2(ret)':>8}  {'R2(price)':>10}  "
          f"{'MAPE%':>7}  {'RawDev$':>9}  {'DirAcc%':>9}")
    print("  " + "─" * 80)
    for t, m in sorted(per_stock.items()):
        print(f"  {t:8s}  {m['rmse_ret']:>10.6f}  {m['mae_ret']:>10.6f}  "
              f"{m['r2_ret']:>8.4f}  {m['r2_price']:>10.4f}  "
              f"{m['mape_pct']:>7.2f}  {m['raw_dev_mean']:>9.2f}  "
              f"{m['dir_acc']:>9.2f}")
    print("  " + "─" * 80)
    print(f"  {'AGGREGATE':8s}  {agg['rmse_ret']:>10.6f}  {agg['mae_ret']:>10.6f}  "
          f"{agg['r2_ret']:>8.4f}  {agg['r2_price']:>10.4f}  "
          f"{agg['mape_pct']:>7.2f}  {agg['raw_dev_mean']:>9.2f}  "
          f"{agg['dir_acc']:>9.2f}\n")

    return per_stock, agg, predictions
