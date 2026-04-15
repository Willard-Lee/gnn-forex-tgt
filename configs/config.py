"""
Configuration for Temporal Graph Transformer EUR/USD Forecasting System v2.

All hyperparameters in one place. Dataclass-based for type safety and IDE support.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    # Source
    ticker: str = "EURUSD=X"
    start_date: str = "2014-01-01"
    end_date: str = "2024-12-31"
    csv_path: str = "data/EURUSD_daily.csv"
    mt5_csv_path: str = "data/EURUSD_mt5.csv"  # optional validation source

    # OHLCV columns (after standardisation)
    date_col: str = "Date"
    ohlcv_cols: List[str] = field(default_factory=lambda: [
        "Open", "High", "Low", "Close", "Volume"
    ])

    # 24 technical indicator nodes (+ AD_Line = 25, may drop if volume unreliable)
    feature_nodes: List[str] = field(default_factory=lambda: [
        # Momentum (10)
        "RSI_14", "MACD", "MACD_Signal", "Stochastic_K", "Stochastic_D",
        "Williams_R", "CCI", "ROC", "MFI", "CMO",
        # Volatility (6)
        "ATR_14", "ATR_20", "BB_Width", "ADX", "NATR", "StdDev_20",
        # Trend (6)
        "EMA_20", "EMA_50", "SMA_20", "SMA_50",
        "Ichimoku_Tenkan", "Ichimoku_Kijun",
        # Volume (2-3)
        "OBV", "CMF",
    ])

    # Node feature augmentation: [raw, z-score, 3d-slope, 5d-vol] = 4 per node
    node_feature_dims: int = 4

    # Sequence / lookback
    sequence_length: int = 30

    # Target engineering
    forecast_horizon: int = 1
    direction_threshold: float = 0.002  # ±0.2% for up/down vs flat

    # Walk-forward validation
    initial_train_years: int = 6        # 2014-2019
    validation_years: int = 1           # 2020
    test_years: int = 1                 # 2021 (first fold)
    walk_forward_step_months: int = 1   # slide forward 1 month
    expanding_window: bool = True       # expand train, don't slide


@dataclass
class GraphConfig:
    """Dynamic multi-edge graph construction."""

    # Pearson correlation edges
    pearson_window: int = 30
    pearson_threshold: float = 0.45

    # DCC-GARCH proxy (vol-adjusted short correlation)
    dcc_window: int = 7

    # Granger causality
    granger_max_lag: int = 5
    granger_p_threshold: float = 0.05

    # Edge weight composition
    weight_pearson: float = 0.40
    weight_dcc: float = 0.40
    weight_granger: float = 0.20

    # Sparsification
    top_k: int = 6
    min_edge_weight: float = 0.05

    # Recompute frequency (trading days)
    recompute_every: int = 20


@dataclass
class ModelConfig:
    """Temporal Graph Transformer architecture."""

    # GAT spatial block (applied at each timestep)
    gat_in_dim: int = 64               # after node_fc projection
    gat_hidden: int = 64
    gat_out_dim: int = 128
    gat_heads_l1: int = 8              # concat → 8*64 = 512
    gat_heads_l2: int = 4              # average → 128
    gat_dropout: float = 0.1
    edge_dropout: float = 0.15

    # Graph pooling output dim (per timestep snapshot)
    graph_snapshot_dim: int = 128

    # Temporal Transformer
    transformer_d_model: int = 128
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 256
    transformer_dropout: float = 0.1

    # Shared FC before heads
    shared_dim: int = 128

    # Output heads
    num_direction_classes: int = 3      # up / flat / down
    use_layer_norm: bool = True
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """Training configuration."""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15

    # Multi-task loss weights
    loss_weight_return: float = 0.4     # MSE
    loss_weight_volatility: float = 0.4 # MAE
    loss_weight_direction: float = 0.2  # CrossEntropy

    # Scheduler
    scheduler: str = "cosine"

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Reproducibility
    seed: int = 42


@dataclass
class BacktestConfig:
    """Backtesting configuration."""

    initial_capital: float = 100_000.0
    leverage: float = 10.0
    spread_pips: float = 1.5
    pip_value: float = 0.0001
    slippage_pips: float = 0.5
    commission_pct: float = 0.0
    max_risk_per_trade: float = 0.02
    confidence_threshold: float = 0.55
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 3.0
    max_drawdown_pct: float = 0.20


@dataclass
class Config:
    """Master configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    device: str = "cpu"
