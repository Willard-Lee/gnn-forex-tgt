# Probabilistic Forecasting of EUR/USD Using Graph Neural Networks

**Final Year Project — Willard Lee Si Ang (0206502)**
**UOW Malaysia KDU Penang University College**
**Bachelor of Computer Science (Hons) — June 2025**

---

## Architecture: Temporal Graph Transformer (TGT)

Unlike the v1 GAT-LSTM (which used separate spatial/temporal paths with late fusion
and underperformed the LSTM baseline), v2 integrates graph processing and temporal
modelling into a single unified pipeline.

```
For each timestep t in [1..T]:
    1. Compute 24 indicator node features (raw + z-score + slope + vol = 96 dims)
    2. Recompute dynamic graph edges from trailing window (every N days)
    3. GAT attention over graph → graph-enriched node embeddings (24 × 128)
    4. Pool graph → single 128-dim snapshot vector

Stack T snapshot vectors → (batch, T, 128)
    5. Temporal Transformer encoder (self-attention over time)
    6. Take [CLS] token or mean-pool → 128-dim representation
    7. Three prediction heads:
       - Direction: 3-class (up/flat/down) → CrossEntropy
       - Returns: scalar → MSE
       - Volatility: scalar → MAE
```

### Why this fixes the v1 failure

- v1 GAT processed only the last timestep → spatial snapshot, no temporal graph evolution
- v1 concatenated 3200-dim GAT + 128-dim LSTM → GAT dominated, LSTM signal drowned
- v2 runs GAT at every timestep → model sees how indicator relationships evolve
- v2 Transformer attends over graph-enriched snapshots → learns which timesteps matter

## Data

- **Primary**: EUR/USD daily OHLCV (2014–2024) via yfinance
- **Validation**: Cross-check against MetaTrader 5 export
- **Supplementary** (future): VIX, 10Y Treasury yields

## 24 Technical Indicators (Nodes)

| Category   | Indicators                                                       |
|------------|------------------------------------------------------------------|
| Momentum   | RSI(14), MACD, MACD_Signal, Stoch_K, Stoch_D, Williams_%R,     |
|            | CCI, ROC, MFI, CMO                                              |
| Volatility | ATR(14), ATR(20), BB_Width, ADX, NATR, StdDev(20)              |
| Trend      | EMA(20), EMA(50), SMA(20), SMA(50), Ichimoku_Tenkan, Ichi_Kijun|
| Volume     | OBV, CMF, AD_Line                                                |

Note: 25 indicators listed but AD_Line may be dropped if volume data is unreliable
from yfinance (forex volume is tick volume, not real volume).

## Graph Construction (Dynamic)

| Edge Type           | Window | Threshold  | Weight | Recompute    |
|---------------------|--------|------------|--------|--------------|
| Pearson correlation | 30d    | |ρ| > 0.45 | 40%    | Every 20 days|
| DCC proxy (vol-adj) | 7d     | —          | 40%    | Every 20 days|
| Granger causality   | 5 lags | p < 0.05   | 20%    | Every 20 days|

Sparsification: top-k=6 per node, min edge weight 0.05.
**Critical**: Graph edges computed ONLY from data available at time t (no leakage).

## Training

- **Split**: Walk-forward expanding window
  - Initial train: 2014–2020, val: 2021, test: 2022
  - Slide forward monthly, retrain
- **Loss**: 0.4×MSE(returns) + 0.4×MAE(volatility) + 0.2×CE(direction)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: Cosine annealing, 100 epochs max, early stopping patience=15
- **Batch**: 32 sequences × 30 timesteps

## Baselines

1. Random Walk (no-skill)
2. MA Crossover (20/50 EMA)
3. LSTM-only (same features, no graph)
4. Random Forest (same features)
5. Buy & Hold

## Evaluation

- **Statistical**: Accuracy, F1 macro, RMSE, MAE, Brier score
- **Financial**: Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor
- **Significance**: Paired t-test, bootstrap CIs
- **Regime**: Performance breakdown by vol regime (low/medium/high)

## Project Structure

```
gnn-forex-v2/
├── README.md
├── requirements.txt
├── main.py                    # CLI entry point
├── configs/
│   └── config.py              # All hyperparameters (dataclass-based)
├── data/
│   └── (raw CSV files)
├── models/
│   ├── temporal_graph_transformer.py  # Main model
│   └── layers.py              # GAT layer, Transformer encoder
├── baselines/
│   ├── lstm_baseline.py
│   ├── random_forest.py
│   └── ma_crossover.py
├── utils/
│   ├── data_pipeline.py       # OHLCV → 24 indicators → targets
│   ├── graph_builder.py       # Dynamic multi-edge graph
│   ├── trainer.py             # Training loop with walk-forward
│   ├── evaluator.py           # Metrics + significance tests
│   └── backtester.py          # Strategy simulation
├── outputs/                   # Generated artifacts
├── notebooks/                 # EDA, analysis
└── references/                # Indicator definitions, etc.
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py fetch_data          # Download EUR/USD via yfinance
python main.py train               # Train TGT + baselines
python main.py eval                # Evaluate on test set
python main.py backtest            # Run trading simulation
python main.py all                 # Full pipeline
streamlit run app.py               # Launch dashboard
```
