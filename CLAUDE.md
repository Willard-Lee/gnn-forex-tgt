# Temporal Graph Transformer — EUR/USD Forex Forecasting

## Architecture
Temporal Graph Transformer (TGT) — NOT the old GAT-LSTM (v1 failed, GAT hurt performance).

The key design: GAT runs at EVERY timestep (not just the last one), producing graph-enriched
snapshots. A Transformer encoder then attends over the sequence of snapshots. Three prediction
heads on top: direction (3-class CE), returns (MSE), volatility (MAE).

```
For each t in [1..30]:
    24 indicator nodes → 4 features each → GAT attention → pool → 128-dim snapshot
Stack 30 snapshots → Transformer encoder → [CLS] → shared FC →
    Head 1: direction (up/flat/down, CrossEntropy ×0.2)
    Head 2: returns (scalar, MSE ×0.4)
    Head 3: volatility (scalar, MAE ×0.4)
```

## Critical rules
- NO data leakage. Graph edges, scaler, indicators — all from past data only at time t.
- Walk-forward validation with expanding window. Not a single train/test split.
- Graph is DYNAMIC — recomputed every 20 trading days from trailing window.
- All 24 indicators computed from pure pandas/numpy. No TA-Lib dependency.
- EUR/USD single pair only (multi-pair is future work).
- This is academic proof-of-concept. NOT live trading. NOT financial advice.
## Tech stack
- Python 3.10+, PyTorch, pandas, numpy, scikit-learn, scipy
- NO PyTorch Geometric (manual GAT implementation, graph is small enough)
- Streamlit for dashboard
- yfinance for data (primary), MetaTrader CSV for validation

## Project structure
```
configs/config.py          — all hyperparameters (dataclasses)
utils/data_pipeline.py     — OHLCV → 24 indicators → targets → sequences
utils/graph_builder.py     — dynamic multi-edge graph (Pearson+DCC+Granger)
models/temporal_graph_transformer.py — main TGT model (~598K params)
models/layers.py           — GATLayer, MultiHeadGAT, GATBlock, PositionalEncoding
utils/trainer.py           — walk-forward trainer + node feature augmentation
utils/evaluator.py         — metrics, significance tests, Diebold-Mariano
utils/backtester.py        — strategy simulation with ATR stops + circuit breaker
baselines/lstm_baseline.py — 2-layer LSTM baseline
baselines/rf_baseline.py   — Random Forest baseline
baselines/ma_baseline.py   — EMA(20)/EMA(50) crossover baseline
baselines/buy_and_hold.py  — Always-long baseline
baselines/run_baselines.py — Runner for all baselines across walk-forward folds
app.py                     — Streamlit dashboard (5 pages)
main.py                    — CLI entry point (pipeline/train/walkforward/baselines/full/dashboard)
```

## Build order (phases)
- [x] Phase 1: Data pipeline + config
- [x] Phase 2: Graph builder (dynamic, multi-edge)
- [x] Phase 3: Model (TGT architecture)
- [x] Phase 4: Trainer (walk-forward, multi-task loss)
- [x] Phase 5: Evaluator (metrics, t-tests, confusion matrices)
- [x] Phase 6: Backtester (ATR stops, circuit breaker, realistic costs)
- [x] Phase 7: Baselines (LSTM, RF, MA, buy-and-hold)
- [x] Phase 8: Dashboard (Streamlit)

## 24 indicator nodes
Momentum: RSI_14, MACD, MACD_Signal, Stochastic_K, Stochastic_D, Williams_R, CCI, ROC, MFI, CMO
Volatility: ATR_14, ATR_20, BB_Width, ADX, NATR, StdDev_20
Trend: EMA_20, EMA_50, SMA_20, SMA_50, Ichimoku_Tenkan, Ichimoku_Kijun
Volume: OBV, CMF

## Graph edges
| Type | Window | Threshold | Weight |
|------|--------|-----------|--------|
| Pearson | 30d | |ρ|>0.45 | 40% |
| DCC proxy | 7d | — | 40% |
| Granger | 5 lags | p<0.05 | 20% |
Sparsified: top-k=6 per node. Recompute every 20 trading days.

## Known issues from v1
- Static graph (built once, frozen) → misleading spatial info during test period
- Late fusion (3200-dim GAT + 128-dim LSTM concat) → GAT dominated, drowned LSTM signal
- No walk-forward → single split, results may be period-specific
- "DCC-GARCH proxy" is just short-window vol-adjusted correlation — label honestly

## Code style
- Type hints on function signatures
- Docstrings on all public functions
- Print progress with emoji status indicators (✅ ❌ ⚠️ 📊)
- Config via dataclasses, not magic numbers
- Test each module before building the next
