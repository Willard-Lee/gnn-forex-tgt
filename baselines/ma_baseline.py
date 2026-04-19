"""
Moving Average Crossover baseline for EUR/USD forecasting.

Classic technical strategy: EMA(20) vs EMA(50) crossover.
  - EMA(20) > EMA(50) → Up (2)
  - EMA(20) < EMA(50) → Down (0)
  - Within threshold → Flat (1)

No learning — purely rule-based. Uses actual indicator values from the DataFrame.
"""

import numpy as np
import pandas as pd
from typing import Dict


def run_ma_baseline(
    test_df: pd.DataFrame,
    seq_len: int = 30,
    cross_threshold: float = 0.0005,
    fold_id: int = 0,
) -> Dict:
    """
    Generate MA crossover predictions aligned with model test predictions.

    Args:
        test_df: test DataFrame with EMA_20, EMA_50, Close columns.
        seq_len: predictions start seq_len rows into test_df.
        cross_threshold: relative diff threshold for flat zone.
        fold_id: fold identifier.

    Returns:
        Fold result dict compatible with evaluator/backtester.
    """
    aligned = test_df.iloc[seq_len:].copy()
    n = len(aligned)

    ema20 = aligned["EMA_20"].values
    ema50 = aligned["EMA_50"].values
    close = aligned["Close"].values

    # Relative difference
    rel_diff = (ema20 - ema50) / np.where(ema50 != 0, ema50, 1)

    # Direction
    direction = np.ones(n, dtype=np.int64)  # Default flat
    direction[rel_diff > cross_threshold] = 2   # Up
    direction[rel_diff < -cross_threshold] = 0  # Down

    # Fake logits: one-hot with small noise for confidence
    logits = np.full((n, 3), -2.0, dtype=np.float32)
    logits[np.arange(n), direction] = 2.0

    # Return prediction: simple momentum (EMA20 direction)
    ema20_change = np.diff(ema20, prepend=ema20[0])
    ret_pred = (ema20_change / np.where(close != 0, close, 1)).astype(np.float32)

    # Volatility prediction: recent rolling std as naive estimate
    vol_pred = pd.Series(close).pct_change().rolling(5, min_periods=1).std().fillna(0).values.astype(np.float32)

    # Targets
    targets = {
        "direction": aligned["target_direction"].values.astype(np.int64),
        "return": aligned["target_return"].values.astype(np.float32),
        "volatility": aligned["target_volatility"].values.astype(np.float32),
    }

    dir_acc = (direction == targets["direction"]).mean()
    print(f"      📊 MA Fold {fold_id} | test_acc={dir_acc:.3f} | "
          f"signals: Up={int((direction==2).sum())}, Down={int((direction==0).sum())}, Flat={int((direction==1).sum())}")

    return {
        "fold_id": fold_id,
        "best_epoch": 0,
        "best_val_loss": 0.0,
        "test_losses": {"total": 1.0 - dir_acc},
        "test_direction_accuracy": float(dir_acc),
        "test_predictions": {
            "direction": logits,
            "return": ret_pred,
            "volatility": vol_pred,
        },
        "test_targets": targets,
    }
