"""
Data Pipeline for Temporal Graph Transformer EUR/USD Forecasting v2.

Pipeline order (strict no-leakage):
  1. Fetch/load OHLCV data
  2. Compute 24 technical indicators (all strictly causal)
  3. Create targets (return, volatility, 3-class direction)
  4. Walk-forward split generator (expanding window)
  5. Per-fold: fit scaler on train only → transform val/test
  6. Create sequences

No TA-Lib dependency — pure pandas/numpy for portability.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from typing import Tuple, Dict, List, Generator
import warnings
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ===========================================================================
# 1. Data Loading
# ===========================================================================

def fetch_yfinance(ticker: str, start: str, end: str, save_path: str) -> pd.DataFrame:
    """Download EUR/USD data from yfinance and save to CSV."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        df = df.reset_index()
        df = df.rename(columns={"index": "Date"})

        # Standardise columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                if col == "Volume":
                    df["Volume"] = 0
                else:
                    raise ValueError(f"Missing column: {col}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✅ Downloaded {len(df)} rows → {save_path}")
        return df
    except ImportError:
        print("⚠️  yfinance not installed. Use: pip install yfinance")
        raise


def load_csv(csv_path: str, date_col: str = "Date") -> pd.DataFrame:
    """Load OHLCV from CSV. Handles both yfinance and MetaTrader formats."""
    df = pd.read_csv(csv_path)

    # MetaTrader format detection
    if "<DATE>" in df.columns:
        df = df.rename(columns={
            "<DATE>": "Date", "<OPEN>": "Open", "<HIGH>": "High",
            "<LOW>": "Low", "<CLOSE>": "Close", "<TICKVOL>": "Volume"
        })
        sep_cols = [c for c in df.columns if c.startswith("<")]
        df = df.drop(columns=sep_cols, errors="ignore")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col)

    # Ensure Volume exists
    if "Volume" not in df.columns:
        df["Volume"] = 0
        print("  ℹ️  No Volume column — filled with 0")

    # Basic validation
    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"✅ Loaded {len(df)} rows | {df.index.min().date()} → {df.index.max().date()}")
    return df


# ===========================================================================
# 2. Technical Indicators (24 nodes, all strictly causal)
# ===========================================================================

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 24 technical indicators from OHLCV. Pure pandas/numpy."""
    out = df.copy()
    c, h, l, o, v = out["Close"], out["High"], out["Low"], out["Open"], out["Volume"]

    # --- Momentum (10) ---

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    # Stochastic K/D (14,3)
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    out["Stochastic_K"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    out["Stochastic_D"] = out["Stochastic_K"].rolling(3).mean()

    # Williams %R (14)
    out["Williams_R"] = -100 * (high14 - c) / (high14 - low14).replace(0, np.nan)

    # CCI (20)
    tp = (h + l + c) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    out["CCI"] = (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)

    # ROC (12)
    out["ROC"] = c.pct_change(12) * 100

    # MFI (14)
    tp_mfi = (h + l + c) / 3
    raw_mf = tp_mfi * v
    pos_mf = raw_mf.where(tp_mfi > tp_mfi.shift(1), 0).rolling(14).sum()
    neg_mf = raw_mf.where(tp_mfi <= tp_mfi.shift(1), 0).rolling(14).sum()
    mfi_ratio = pos_mf / neg_mf.replace(0, np.nan)
    out["MFI"] = 100 - (100 / (1 + mfi_ratio))

    # CMO (14)
    up_sum = delta.clip(lower=0).rolling(14).sum()
    dn_sum = (-delta.clip(upper=0)).rolling(14).sum()
    out["CMO"] = 100 * (up_sum - dn_sum) / (up_sum + dn_sum).replace(0, np.nan)

    # --- Volatility (6) ---

    # ATR(14) and ATR(20)
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    out["ATR_14"] = tr.rolling(14).mean()
    out["ATR_20"] = tr.rolling(20).mean()

    # Bollinger Band Width (20, 2std)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    out["BB_Width"] = (bb_upper - bb_lower) / sma20.replace(0, np.nan)

    # ADX (14)
    plus_dm = (h - h.shift(1)).clip(lower=0)
    minus_dm = (l.shift(1) - l).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["ADX"] = dx.rolling(14).mean()

    # NATR (14)
    out["NATR"] = (out["ATR_14"] / c) * 100

    # StdDev(20)
    out["StdDev_20"] = c.rolling(20).std()

    # --- Trend (6) ---

    out["EMA_20"] = c.ewm(span=20, adjust=False).mean()
    out["EMA_50"] = c.ewm(span=50, adjust=False).mean()
    out["SMA_20"] = c.rolling(20).mean()
    out["SMA_50"] = c.rolling(50).mean()

    # Ichimoku (Tenkan=9, Kijun=26)
    out["Ichimoku_Tenkan"] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    out["Ichimoku_Kijun"] = (h.rolling(26).max() + l.rolling(26).min()) / 2

    # --- Volume (2) ---

    out["OBV"] = (v * np.sign(delta).fillna(0)).cumsum()
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = mfm.fillna(0) * v
    out["CMF"] = mfv.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

    # Drop warmup NaN rows
    out = out.dropna(subset=["ADX", "SMA_50", "Ichimoku_Kijun"])

    n_indicators = sum(1 for col in out.columns if col not in df.columns)
    print(f"✅ Computed {n_indicators} indicators | {len(out)} rows after warmup")
    return out


# ===========================================================================
# 3. Target Engineering
# ===========================================================================

def create_targets(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.002) -> pd.DataFrame:
    """
    Create prediction targets:
      - target_return: log(close_{t+h} / close_t)
      - target_volatility: 5-day rolling std of log returns
      - target_direction: 0=down, 1=flat, 2=up (±threshold)
    """
    out = df.copy()
    future_close = out["Close"].shift(-horizon)
    log_ret = np.log(future_close / out["Close"])

    out["target_return"] = log_ret
    out["target_volatility"] = np.log(out["Close"] / out["Close"].shift(1)).rolling(5).std()

    out["target_direction"] = 1  # flat
    out.loc[log_ret > threshold, "target_direction"] = 2   # up
    out.loc[log_ret < -threshold, "target_direction"] = 0  # down
    out["target_direction"] = out["target_direction"].astype(int)

    out = out.dropna(subset=["target_return", "target_volatility"])

    counts = out["target_direction"].value_counts().sort_index()
    labels = {0: "Down", 1: "Flat", 2: "Up"}
    dist = " | ".join(f"{labels[i]}: {counts.get(i, 0)} ({counts.get(i,0)/len(out)*100:.1f}%)"
                       for i in [0, 1, 2])
    print(f"✅ Targets | {len(out)} rows | {dist}")
    return out


# ===========================================================================
# 4. Walk-Forward Split Generator
# ===========================================================================

def walk_forward_splits(
    df: pd.DataFrame,
    initial_train_years: int = 6,
    val_years: int = 1,
    test_years: int = 1,
    step_months: int = 1,
    expanding: bool = True,
) -> Generator[Dict[str, pd.DataFrame], None, None]:
    """
    Generate walk-forward train/val/test splits.

    Expanding window: train grows each step.
    Sliding window: train stays fixed size, slides forward.

    Yields dicts with 'train', 'val', 'test' DataFrames and 'fold_id'.
    """
    start = df.index.min()
    end = df.index.max()

    train_end = start + pd.DateOffset(years=initial_train_years)
    val_end = train_end + pd.DateOffset(years=val_years)
    test_end = val_end + pd.DateOffset(years=test_years)

    fold = 0
    while test_end <= end + pd.DateOffset(months=1):
        if expanding:
            train_start = start
        else:
            train_start = train_end - pd.DateOffset(years=initial_train_years)

        train = df.loc[train_start:train_end].copy()
        val = df.loc[train_end:val_end].copy()
        test = df.loc[val_end:test_end].copy()

        if len(train) > 100 and len(val) > 20 and len(test) > 20:
            yield {
                "fold_id": fold,
                "train": train,
                "val": val,
                "test": test,
                "train_end": train_end,
                "val_end": val_end,
                "test_end": test_end,
            }
            fold += 1

        # Slide forward
        train_end += pd.DateOffset(months=step_months)
        val_end += pd.DateOffset(months=step_months)
        test_end += pd.DateOffset(months=step_months)

    print(f"✅ Walk-forward | {fold} folds generated")


# ===========================================================================
# 5. Feature Scaling (per fold, fit on train only)
# ===========================================================================

def scale_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, QuantileTransformer]:
    """Quantile transform to [0,1]. Fitted ONLY on training data."""
    scaler = QuantileTransformer(
        output_distribution="uniform",
        n_quantiles=min(1000, len(train)),
    )

    train = train.copy()
    val = val.copy()
    test = test.copy()

    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    val[feature_cols] = scaler.transform(val[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])

    return train, val, test, scaler


# ===========================================================================
# 6. Sequence Creation
# ===========================================================================

def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 30,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Create overlapping sequences.

    Returns:
        X: (N, seq_len, num_features) float32
        y: dict with 'direction' (int64), 'return' (float32), 'volatility' (float32)
    """
    features = df[feature_cols].values
    dir_t = df["target_direction"].values
    ret_t = df["target_return"].values
    vol_t = df["target_volatility"].values

    X, y_dir, y_ret, y_vol = [], [], [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len:i])
        y_dir.append(dir_t[i])
        y_ret.append(ret_t[i])
        y_vol.append(vol_t[i])

    return (
        np.array(X, dtype=np.float32),
        {
            "direction": np.array(y_dir, dtype=np.int64),
            "return": np.array(y_ret, dtype=np.float32),
            "volatility": np.array(y_vol, dtype=np.float32),
        },
    )


# ===========================================================================
# 7. Full Pipeline
# ===========================================================================

def run_pipeline(cfg) -> Dict:
    """Execute the full data pipeline (single pass, no walk-forward)."""

    # Load or fetch
    if os.path.exists(cfg.data.csv_path):
        df = load_csv(cfg.data.csv_path)
    else:
        df = fetch_yfinance(
            cfg.data.ticker, cfg.data.start_date,
            cfg.data.end_date, cfg.data.csv_path,
        )
        df[cfg.data.date_col] = pd.to_datetime(df[cfg.data.date_col])
        df = df.set_index(cfg.data.date_col)

    # Compute indicators and targets
    df = compute_indicators(df)
    df = create_targets(df, cfg.data.forecast_horizon, cfg.data.direction_threshold)

    # For quick testing: single chronological split (70/15/15)
    n = len(df)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)

    train_df = df.iloc[:t_end].copy()
    val_df = df.iloc[t_end:v_end].copy()
    test_df = df.iloc[v_end:].copy()

    feat = cfg.data.feature_nodes
    # Filter to features that actually exist (volume indicators may be missing)
    feat = [f for f in feat if f in df.columns]

    train_df, val_df, test_df, scaler = scale_features(train_df, val_df, test_df, feat)

    train_X, train_y = create_sequences(train_df, feat, cfg.data.sequence_length)
    val_X, val_y = create_sequences(val_df, feat, cfg.data.sequence_length)
    test_X, test_y = create_sequences(test_df, feat, cfg.data.sequence_length)

    print(f"\n📊 Sequence shapes:")
    print(f"   Train: {train_X.shape} | Val: {val_X.shape} | Test: {test_X.shape}")

    return {
        "train_X": train_X, "train_y": train_y,
        "val_X": val_X, "val_y": val_y,
        "test_X": test_X, "test_y": test_y,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "scaler": scaler, "feature_cols": feat, "df_full": df,
    }
