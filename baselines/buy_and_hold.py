"""
Buy-and-Hold baseline for EUR/USD forecasting.

Always predicts "Up" (direction=2). The simplest possible benchmark.
For return/volatility, predicts the training set mean (constant forecast).
"""

import numpy as np
import pandas as pd
from typing import Dict


def run_buy_and_hold(
    test_df: pd.DataFrame,
    train_y: Dict,
    seq_len: int = 30,
    fold_id: int = 0,
) -> Dict:
    """
    Generate buy-and-hold predictions.

    Args:
        test_df: test DataFrame with targets.
        train_y: training targets dict (to compute mean predictions).
        seq_len: predictions start seq_len rows into test_df.
        fold_id: fold identifier.

    Returns:
        Fold result dict compatible with evaluator/backtester.
    """
    aligned = test_df.iloc[seq_len:]
    n = len(aligned)

    # Always predict Up
    direction = np.full(n, 2, dtype=np.int64)
    logits = np.full((n, 3), -2.0, dtype=np.float32)
    logits[:, 2] = 2.0  # High confidence in Up

    # Constant predictions: training set means
    mean_ret = float(np.mean(train_y["return"]))
    mean_vol = float(np.mean(train_y["volatility"]))
    ret_pred = np.full(n, mean_ret, dtype=np.float32)
    vol_pred = np.full(n, max(mean_vol, 0), dtype=np.float32)

    targets = {
        "direction": aligned["target_direction"].values.astype(np.int64),
        "return": aligned["target_return"].values.astype(np.float32),
        "volatility": aligned["target_volatility"].values.astype(np.float32),
    }

    dir_acc = (direction == targets["direction"]).mean()
    print(f"      📊 B&H Fold {fold_id} | test_acc={dir_acc:.3f} | "
          f"(always Up, const ret={mean_ret:.5f}, vol={mean_vol:.5f})")

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
