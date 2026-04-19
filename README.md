# Probabilistic Forecasting of EUR/USD Using Graph Neural Networks

Academic proof-of-concept. NOT live trading. NOT financial advice.

## Architecture: Temporal Graph Transformer (TGT)

Unlike the v1 GAT-LSTM (which used separate spatial/temporal paths with late fusion
and underperformed the LSTM baseline), v2 integrates graph processing and temporal
modelling into a single unified pipeline.

```
For each timestep t in [1..30]:
    1. Compute 24 indicator node features (raw + z-score + slope + vol = 4 dims each)
    2. Project to 64-dim → 2-layer GAT (8-head concat → 4-head avg) → 128-dim per node
    3. Mean pool 24 nodes → 128-dim snapshot vector

Stack 30 snapshots → prepend [CLS] token + positional encoding
    4. Transformer encoder (2 layers, 4 heads, pre-norm)
    5. [CLS] output → shared FC(128) → three prediction heads:
       - Direction: 3-class (up/flat/down) → CrossEntropy ×0.2
       - Returns: scalar → MSE ×0.4
       - Volatility: scalar (Softplus) → MAE ×0.4
```

**~598K parameters** — lean and trainable on CPU.

### Why this fixes the v1 failure

- v1 GAT processed only the last timestep → spatial snapshot, no temporal graph evolution
- v1 concatenated 3200-dim GAT + 128-dim LSTM → GAT dominated, LSTM signal drowned
- v2 runs GAT at every timestep → model sees how indicator relationships evolve
- v2 Transformer attends over graph-enriched snapshots → learns which timesteps matter

## Data

- **Primary**: EUR/USD daily OHLCV (2014-2024) via yfinance
- **Validation**: Cross-check against MetaTrader 5 export
- 24 technical indicators computed from pure pandas/numpy (no TA-Lib)

## 24 Technical Indicators (Graph Nodes)

| Category   | Indicators                                                        |
|------------|-------------------------------------------------------------------|
| Momentum   | RSI(14), MACD, MACD_Signal, Stoch_K, Stoch_D, Williams_%R,      |
|            | CCI, ROC, MFI, CMO                                               |
| Volatility | ATR(14), ATR(20), BB_Width, ADX, NATR, StdDev(20)               |
| Trend      | EMA(20), EMA(50), SMA(20), SMA(50), Ichimoku_Tenkan, Ichi_Kijun |
| Volume     | OBV, CMF                                                         |

Each node has 4 features: raw value, z-score(20), 3-day slope, 5-day volatility.

## Graph Construction (Dynamic)

| Edge Type           | Window | Threshold  | Weight | Recompute     |
|---------------------|--------|------------|--------|---------------|
| Pearson correlation | 30d    | |ρ| > 0.45 | 40%    | Every 20 days |
| DCC proxy (vol-adj) | 7d     | --         | 40%    | Every 20 days |
| Granger causality   | 5 lags | p < 0.05   | 20%    | Every 20 days |

Sparsification: top-k=6 per node, min edge weight 0.05.

**Critical**: Graph edges computed ONLY from data available at time t (no leakage).
The "DCC-GARCH proxy" is a short-window vol-adjusted correlation, not a true DCC-GARCH model.

## Training

- **Walk-forward**: Expanding window (initial train: 6y, val: 1y, test: 1y, slide 1mo)
- **Loss**: 0.4 x MSE(returns) + 0.4 x MAE(volatility) + 0.2 x CE(direction)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: Cosine annealing, 100 epochs max, early stopping patience=15
- **Gradient clipping**: max_norm=1.0
- **Batch**: 32 sequences x 30 timesteps

## Baselines

| Baseline       | Description                                      |
|----------------|--------------------------------------------------|
| LSTM           | 2-layer LSTM, same multi-task loss + early stop   |
| Random Forest  | Flattened sequences, separate clf + 2 regressors  |
| MA Crossover   | EMA(20) vs EMA(50), rule-based, no learning       |
| Buy & Hold     | Always predict Up, constant return/vol from train |

All baselines produce the same output format for direct comparison via evaluator.

## Evaluation

- **Classification**: Accuracy, per-class P/R/F1, confusion matrix
- **Regression**: MSE, RMSE, MAE, R², directional agreement, IC (Spearman)
- **Volatility**: MAE, RMSE, R², QLIKE (Patton quasi-likelihood)
- **Significance**: Paired t-test (Cohen's d), Diebold-Mariano test (Newey-West)
- **Financial**: Sharpe, Sortino, max drawdown, win rate, profit factor

## Backtesting

- ATR-based stops: SL = 2x ATR(14), TP = 3x ATR(14)
- Position sizing: 2% equity risk per trade, capped by 10x leverage
- Transaction costs: 1.5 pip spread + 0.5 pip slippage
- Confidence threshold: only trade when softmax confidence >= 0.55
- Circuit breaker: halt at 20% max drawdown

## Project Structure

```
gnn-forex-v2/
├── CLAUDE.md                          # Project instructions
├── README.md                          # This file
├── main.py                            # CLI entry point
├── app.py                             # Streamlit dashboard
├── configs/
│   └── config.py                      # All hyperparameters (dataclass-based)
├── data/
│   └── (raw CSV files)
├── models/
│   ├── layers.py                      # GATLayer, MultiHeadGAT, GATBlock, PositionalEncoding
│   └── temporal_graph_transformer.py  # Main TGT model (~598K params)
├── baselines/
│   ├── lstm_baseline.py               # 2-layer LSTM baseline
│   ├── rf_baseline.py                 # Random Forest baseline
│   ├── ma_baseline.py                 # MA crossover baseline
│   ├── buy_and_hold.py                # Buy & Hold baseline
│   └── run_baselines.py               # Run all baselines across folds
├── utils/
│   ├── data_pipeline.py               # OHLCV → 24 indicators → targets → sequences
│   ├── graph_builder.py               # Dynamic multi-edge graph construction
│   ├── trainer.py                     # Walk-forward training + node feature augmentation
│   ├── evaluator.py                   # Metrics, significance tests, reporting
│   └── backtester.py                  # Strategy simulation with realistic friction
└── results/
    └── experiment.pkl                 # Saved experiment results
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full experiment (pipeline → TGT → baselines → evaluate)
python main.py --mode full

# Or run individual stages
python main.py --mode pipeline       # Data pipeline only
python main.py --mode train          # Quick single-split TGT training
python main.py --mode walkforward    # Full walk-forward training
python main.py --mode baselines      # Run all baselines

# Launch dashboard to view results
streamlit run app.py

# CLI options
python main.py --mode full --epochs 50 --batch-size 16 --device mps
```

## Dashboard

The Streamlit dashboard (`app.py`) has 5 pages:

| Page                 | Content                                                    |
|----------------------|------------------------------------------------------------|
| Overview             | Architecture diagram, design choices, result status         |
| Data Pipeline        | Price chart, indicator plots, target distribution           |
| Model Results        | Walk-forward metrics, confusion matrices, per-fold details  |
| Baseline Comparison  | Side-by-side table, accuracy chart, significance tests      |
| Backtest             | Equity curve, trade log, Sharpe, drawdown, profit factor    |

## Known Issues from v1 (Resolved)

| v1 Problem | v2 Solution |
|---|---|
| Static graph (built once, frozen) | Dynamic graph recomputed every 20 trading days |
| Late fusion (3200-dim GAT + 128-dim LSTM) | GAT at every timestep, Transformer over snapshots |
| No walk-forward validation | Expanding window walk-forward with monthly slides |
| "DCC-GARCH" mislabeled | Honestly labeled as "DCC proxy" (vol-adjusted short correlation) |
