"""
Run all baselines on the same walk-forward splits as TGT.

Baselines:
  1. LSTM (2-layer, same multi-task loss)
  2. Random Forest (flattened sequences)
  3. MA Crossover (EMA20 vs EMA50, rule-based)
  4. Buy-and-Hold (always predict Up)
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from configs.config import Config
from utils.data_pipeline import walk_forward_splits, scale_features, create_sequences
from baselines.lstm_baseline import train_lstm_fold
from baselines.rf_baseline import train_rf_fold
from baselines.ma_baseline import run_ma_baseline
from baselines.buy_and_hold import run_buy_and_hold


def run_all_baselines(df: pd.DataFrame, cfg: Config) -> Dict[str, List[Dict]]:
    """
    Run all baselines across walk-forward folds.

    Args:
        df: Full DataFrame with indicators + targets.
        cfg: Config object.

    Returns:
        Dict mapping baseline name → list of fold results.
    """
    feature_cols = [f for f in cfg.data.feature_nodes if f in df.columns]

    results = {
        "LSTM": [],
        "RandomForest": [],
        "MA_Crossover": [],
        "BuyAndHold": [],
    }

    print("=" * 60)
    print("🧪 Running All Baselines (Walk-Forward)")
    print("=" * 60)

    for fold_data in walk_forward_splits(
        df,
        initial_train_years=cfg.data.initial_train_years,
        val_years=cfg.data.validation_years,
        test_years=cfg.data.test_years,
        step_months=cfg.data.walk_forward_step_months,
        expanding=cfg.data.expanding_window,
    ):
        fold_id = fold_data["fold_id"]
        print(f"\n{'─' * 50}")
        print(f"📂 Fold {fold_id}")

        # Scale
        train_s, val_s, test_s, _ = scale_features(
            fold_data["train"], fold_data["val"], fold_data["test"], feature_cols,
        )

        # Create sequences
        train_X, train_y = create_sequences(train_s, feature_cols, cfg.data.sequence_length)
        val_X, val_y = create_sequences(val_s, feature_cols, cfg.data.sequence_length)
        test_X, test_y = create_sequences(test_s, feature_cols, cfg.data.sequence_length)

        if len(test_X) < 5:
            print(f"   ⚠️  Too few test samples ({len(test_X)}), skipping fold")
            continue

        # 1. LSTM
        print(f"\n   🔷 LSTM")
        lstm_result = train_lstm_fold(train_X, train_y, val_X, val_y, test_X, test_y, cfg, fold_id)
        results["LSTM"].append(lstm_result)

        # 2. Random Forest
        print(f"\n   🟩 Random Forest")
        rf_result = train_rf_fold(train_X, train_y, val_X, val_y, test_X, test_y, fold_id)
        results["RandomForest"].append(rf_result)

        # 3. MA Crossover (needs unscaled test_df for EMA values)
        print(f"\n   🟨 MA Crossover")
        ma_result = run_ma_baseline(fold_data["test"], cfg.data.sequence_length, fold_id=fold_id)
        results["MA_Crossover"].append(ma_result)

        # 4. Buy-and-Hold
        print(f"\n   ⬜ Buy & Hold")
        bh_result = run_buy_and_hold(fold_data["test"], train_y, cfg.data.sequence_length, fold_id)
        results["BuyAndHold"].append(bh_result)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"📊 Baseline Summary")
    print(f"{'=' * 60}")
    for name, fold_results in results.items():
        if fold_results:
            accs = [r["test_direction_accuracy"] for r in fold_results]
            print(f"   {name:15s}: acc={np.mean(accs):.3f} ± {np.std(accs):.3f} ({len(fold_results)} folds)")

    return results


# ===========================================================================
# Smoke Test
# ===========================================================================

def test_baselines():
    """Quick smoke test with synthetic data."""
    import warnings
    warnings.filterwarnings("ignore")

    cfg = Config()
    cfg.train.epochs = 3
    cfg.train.patience = 2
    cfg.train.batch_size = 8
    cfg.data.sequence_length = 10

    print("=" * 60)
    print("🧪 Baselines Smoke Test")
    print("=" * 60)

    np.random.seed(42)
    n_days = 100
    feature_cols = cfg.data.feature_nodes

    data = {}
    base = np.cumsum(np.random.randn(n_days) * 0.01)
    for name in feature_cols:
        data[name] = base + np.random.randn(n_days) * 0.3

    data["Close"] = 1.08 + np.cumsum(np.random.randn(n_days) * 0.005)
    data["EMA_20"] = pd.Series(data["Close"]).ewm(span=20).mean().values
    data["EMA_50"] = pd.Series(data["Close"]).ewm(span=50).mean().values
    data["target_direction"] = np.random.randint(0, 3, n_days)
    data["target_return"] = np.random.randn(n_days) * 0.01
    data["target_volatility"] = np.abs(np.random.randn(n_days) * 0.005)

    # Split manually (no walk-forward for smoke test)
    t_end, v_end = 60, 80
    train_X_raw = np.random.randn(t_end - cfg.data.sequence_length, cfg.data.sequence_length, len(feature_cols)).astype(np.float32)
    val_X_raw = np.random.randn(v_end - t_end - cfg.data.sequence_length, cfg.data.sequence_length, len(feature_cols)).astype(np.float32)
    test_X_raw = np.random.randn(n_days - v_end - cfg.data.sequence_length, cfg.data.sequence_length, len(feature_cols)).astype(np.float32)

    n_tr, n_va, n_te = len(train_X_raw), len(val_X_raw), len(test_X_raw)

    train_y = {"direction": np.random.randint(0, 3, n_tr).astype(np.int64),
               "return": np.random.randn(n_tr).astype(np.float32),
               "volatility": np.abs(np.random.randn(n_tr) * 0.005).astype(np.float32)}
    val_y = {"direction": np.random.randint(0, 3, n_va).astype(np.int64),
             "return": np.random.randn(n_va).astype(np.float32),
             "volatility": np.abs(np.random.randn(n_va) * 0.005).astype(np.float32)}
    test_y = {"direction": np.random.randint(0, 3, n_te).astype(np.int64),
              "return": np.random.randn(n_te).astype(np.float32),
              "volatility": np.abs(np.random.randn(n_te) * 0.005).astype(np.float32)}

    test_df = pd.DataFrame(data).iloc[v_end:]

    # 1. LSTM
    print("\n1️⃣  LSTM Baseline")
    lstm_r = train_lstm_fold(train_X_raw, train_y, val_X_raw, val_y, test_X_raw, test_y, cfg)
    assert "test_predictions" in lstm_r
    print("   ✅ LSTM OK")

    # 2. Random Forest
    print("\n2️⃣  Random Forest Baseline")
    rf_r = train_rf_fold(train_X_raw, train_y, val_X_raw, val_y, test_X_raw, test_y)
    assert "test_predictions" in rf_r
    print("   ✅ RF OK")

    # 3. MA Crossover
    print("\n3️⃣  MA Crossover Baseline")
    test_df_ma = test_df.copy()
    test_df_ma["target_direction"] = np.random.randint(0, 3, len(test_df_ma))
    test_df_ma["target_return"] = np.random.randn(len(test_df_ma)) * 0.01
    test_df_ma["target_volatility"] = np.abs(np.random.randn(len(test_df_ma)) * 0.005)
    ma_r = run_ma_baseline(test_df_ma, cfg.data.sequence_length)
    assert "test_predictions" in ma_r
    print("   ✅ MA OK")

    # 4. Buy-and-Hold
    print("\n4️⃣  Buy & Hold Baseline")
    bh_r = run_buy_and_hold(test_df_ma, train_y, cfg.data.sequence_length)
    assert "test_predictions" in bh_r
    print("   ✅ B&H OK")

    # Verify all outputs have same format
    for name, r in [("LSTM", lstm_r), ("RF", rf_r), ("MA", ma_r), ("B&H", bh_r)]:
        p = r["test_predictions"]
        assert "direction" in p and "return" in p and "volatility" in p, f"{name} missing prediction key"
        assert p["direction"].ndim == 2 and p["direction"].shape[1] == 3, f"{name} direction shape wrong"

    print(f"\n✅ All baseline tests passed!")


if __name__ == "__main__":
    test_baselines()
