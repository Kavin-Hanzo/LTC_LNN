# Stock2Vec — Multi-Stock MIMO LSTM with PCA Identity Vectors

## Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py
```

All outputs go to `outputs/`.

---

## Project Structure

```
stock2vec/
│
├── config.py               ← Edit ONLY this file to change any parameter
├── run_pipeline.py         ← Entry point
├── requirements.txt
│
├── data/
│   ├── fetcher.py          ← Phase 1 : Download OHLCV, cache to disk
│   ├── indicators.py       ← Phase 2 : 7 dimensionless features
│   ├── scaler.py           ← Phase 3 : Walk-forward scaler (no lookahead)
│   └── dataset.py          ← Phase 4 : MIMO sliding-window PyTorch Dataset
│
├── vectors/
│   └── stock2vec.py        ← Build 4D PCA identity vectors
│
├── models/
│   ├── lstm.py             ← Configurable MIMO LSTM
│   └── trainer.py          ← Training loop + early stopping
│
├── evaluation/
│   ├── metrics.py          ← MAE · RMSE · MAPE · RawDev($) · DirAcc%
│   └── evaluator.py        ← Inference → per-ticker metric tables
│
├── experiments/
│   ├── e1_clustering.py    ← t-SNE · silhouette · distance separation
│   ├── e2_ablation.py      ← Baseline vs OneHot vs Stock2Vec
│   ├── e3_generalization.py← Leave-one-out zero-shot test
│   └── e4_interpretability.py ← PC loading vs financial proxy
│
├── visualization/
│   └── plotter.py          ← Indicator dashboards · return charts · summary
│
└── outputs/                ← Auto-created at runtime
    ├── raw/                ← Cached OHLCV CSVs (avoids re-downloading)
    ├── vectors/            ← stock2vec_{N}y.csv · pca_variance_{N}y.csv
    ├── models/             ← .pt checkpoints
    ├── results/            ← Metric and experiment CSVs
    └── plots/              ← All PNG figures
```

---

## Pipeline Flow

```
yfinance OHLCV
    └─► data/fetcher.py          raw OHLCV cached to outputs/raw/
         └─► data/indicators.py  7 ratio features (no raw prices)
              └─► data/scaler.py walk-forward StandardScaler
                   ├─► vectors/stock2vec.py  4D PCA identity per stock
                   └─► data/dataset.py       MIMO sliding windows
                            └─► models/lstm.py + trainer.py
                                  ├── Baseline   (7 features)
                                  ├── OneHot     (7 + 3 one-hot)
                                  └── Stock2Vec  (7 + 4 PCA)
                                        └─► evaluation/evaluator.py
                                              ├── e1_clustering.py
                                              ├── e2_ablation.py
                                              ├── e3_generalization.py
                                              └── e4_interpretability.py
```

---

## MIMO Architecture

```
SHORT mode : lookback=30  days → predict next 5  days (1 trading week)
LONG  mode : lookback=120 days → predict next 21 days (1 trading month)

Input  : (batch, lookback, 7 features + 4 identity dims)
Output : (batch, horizon)   ← all H steps in ONE forward pass
```

The 4D identity vector is repeated at every time-step:

```python
id_exp = identity.unsqueeze(1).expand(-1, lookback, -1)   # (B, T, 4)
x      = torch.cat([x, id_exp], dim=-1)                    # (B, T, 11)
```

---

## Configurable Parameters (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `tickers` | 6 US IT stocks | Add/remove tickers freely |
| `history_years` | `[5, 10]` | Runs both windows in sequence |
| `num_layers` | `2` | LSTM depth: 2=light, 3=medium, 4=deep |
| `hidden_size` | `128` | Units per LSTM layer |
| `dropout` | `0.20` | Applied between layers and before head |
| `epochs` | `100` | Max training epochs (early stopping applies) |
| `patience` | `15` | Early stopping patience |
| `modes` | short + long | Change lookback/horizon freely |

---

## Outputs Reference

```
outputs/plots/
  dashboard_{TICKER}_{N}y.png            indicator chart per stock
  e1_tsne_{N}y.png                       t-SNE cluster map
  e1_cosine_{N}y.png                     cosine similarity heatmap
  e1_distances_{N}y.png                  intra vs inter-sector distances
  e1_loadings_{N}y.png                   PC loading bars
  e2_rmse_{N}y_{mode}.png                RMSE comparison: 3 models
  e2_rawdev_{N}y_{mode}.png              raw price deviation ($)
  e2_diracc_{N}y_{mode}.png              direction accuracy (%)
  e2_loss_curves_{N}y_{mode}.png         training dynamics
  e2_pred_vs_actual_{N}y_{mode}.png      price overlay chart
  e2_rawdev_horizon_{TICKER}_{mode}.png  deviation growth per horizon step
  e3_dist_vs_rmse_{N}y.png               generalisation scatter
  e3_zeroshot_vs_full_{N}y.png           zero-shot vs full training
  e4_radar_{N}y.png                      PC fingerprint radar
  e4_pc_vs_proxy_{N}y.png               PC loading vs financial proxy
  e4_corr_heatmap_{N}y.png              correlation heatmap
  cumret_{TICKER}_{MODEL}_{mode}.png     cumulative return + raw deviation
  research_summary_{N}y.png             one-page research summary

outputs/results/
  e1_loadings_{N}y.csv                   4D vectors per stock
  e1_summary_{N}y.csv                    E1 scalar results
  e2_metrics_{N}y_{mode}.csv             full metric table
  e2_improvement_{N}y_{mode}.csv         % improvement per stock
  e3_results_{N}y.csv                    zero-shot vs full-training RMSE
  e4_ranked_PC{k}_{N}y.csv              stocks ranked by PC loading

outputs/vectors/
  stock2vec_{N}y.csv                     ticker → [v1, v2, v3, v4]
  pca_variance_{N}y.csv                  per-component explained variance

outputs/models/
  {ModelName}_{N}y_{mode}.pt             one checkpoint per variant
```

---

## Experiments Summary

| # | Claim | Key Figure | Key Number |
|---|---|---|---|
| E1 | Vectors capture sector identity | t-SNE scatter | Separation ratio, Silhouette score |
| E2 | Vector improves prediction | Grouped RMSE bars | % RMSE reduction vs Baseline |
| E3 | Closer vector → better zero-shot | Distance vs RMSE scatter | Pearson ρ |
| E4 | Each PC has financial meaning | Radar + proxy scatter | PC1 × volatility ρ |

---

## Known Limitations

- **Survivorship bias** — yfinance returns only currently listed tickers.
- **Static identity** — the vector is computed once on the training split.
  It does not adapt to regime changes (e.g. NVDA post-2023 AI boom).
- **US IT only** — all 6 stocks are highly correlated (r ≈ 0.6–0.85).
  Cross-sector data would sharpen E1 and E3.
- **No macro features** — VIX, Fed rate, sector ETF flows are absent.
