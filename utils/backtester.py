"""
Backtester for Temporal Graph Transformer v2.

Simulates a realistic EUR/USD trading strategy with:
  - ATR-based stop-loss (2×ATR) and take-profit (3×ATR)
  - Position sizing via max risk per trade (2% of equity)
  - Confidence threshold filtering (only trade when model is confident)
  - Spread + slippage transaction costs
  - Circuit breaker: halt trading at max drawdown (20%)
  - Walk-forward compatible: backtest each fold's test predictions

NOT financial advice. Academic proof-of-concept only.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from configs.config import BacktestConfig


# ===========================================================================
# 1. Trade and Position Tracking
# ===========================================================================

@dataclass
class Trade:
    """Record of a single completed trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int          # 2=long, 0=short
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size: float    # units of base currency
    pnl: float             # net P&L after costs
    pnl_pct: float         # P&L as % of equity at entry
    exit_reason: str        # "stop_loss", "take_profit", "end_of_period"
    confidence: float
    atr_at_entry: float


@dataclass
class BacktestState:
    """Mutable state during backtesting."""
    equity: float
    peak_equity: float
    position_open: bool = False
    position_direction: int = 1  # 2=long, 0=short
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size: float = 0.0
    entry_date: Optional[pd.Timestamp] = None
    entry_confidence: float = 0.0
    entry_atr: float = 0.0
    circuit_breaker_active: bool = False
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    dates: List[pd.Timestamp] = field(default_factory=list)


# ===========================================================================
# 2. Core Backtester
# ===========================================================================

class Backtester:
    """
    Event-driven backtester for EUR/USD direction predictions.

    For each trading day:
      1. If position open: check SL/TP against High/Low
      2. If no position and not halted: check model signal + confidence
      3. If signal passes: compute position size, enter trade
      4. Update equity curve and drawdown

    Position sizing: risk_amount = equity × max_risk_per_trade
                     position_size = risk_amount / (ATR × SL_multiplier × pip_value)
    """

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg

    def run(
        self,
        dates: np.ndarray,
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        atr_values: np.ndarray,
        pred_direction: np.ndarray,
        pred_confidence: np.ndarray,
    ) -> Dict:
        """
        Run backtest over a test period.

        Args:
            dates: (N,) array of timestamps.
            close_prices: (N,) close prices.
            high_prices: (N,) high prices.
            low_prices: (N,) low prices.
            atr_values: (N,) ATR(14) values.
            pred_direction: (N,) predicted direction {0=down, 1=flat, 2=up}.
            pred_confidence: (N,) model confidence (max softmax prob).

        Returns:
            Dict with trades, equity curve, and performance metrics.
        """
        cfg = self.cfg
        state = BacktestState(
            equity=cfg.initial_capital,
            peak_equity=cfg.initial_capital,
        )

        # Transaction cost in price terms
        spread_cost = cfg.spread_pips * cfg.pip_value
        slippage_cost = cfg.slippage_pips * cfg.pip_value
        total_entry_cost = spread_cost + slippage_cost

        for i in range(len(dates)):
            date = dates[i]
            close = close_prices[i]
            high = high_prices[i]
            low = low_prices[i]
            atr = atr_values[i]

            # --- Check circuit breaker ---
            drawdown = (state.peak_equity - state.equity) / state.peak_equity
            if drawdown >= cfg.max_drawdown_pct and not state.circuit_breaker_active:
                state.circuit_breaker_active = True
                # Force close any open position
                if state.position_open:
                    self._close_position(state, close, date, "circuit_breaker", total_entry_cost)

            # --- Check open position for SL/TP ---
            if state.position_open:
                hit_sl, hit_tp = self._check_stops(state, high, low)

                if hit_sl:
                    self._close_position(state, state.stop_loss, date, "stop_loss", total_entry_cost)
                elif hit_tp:
                    self._close_position(state, state.take_profit, date, "take_profit", total_entry_cost)

            # --- Entry logic ---
            if (
                not state.position_open
                and not state.circuit_breaker_active
                and pred_direction[i] != 1  # Not flat
                and pred_confidence[i] >= cfg.confidence_threshold
                and atr > 0
            ):
                direction = pred_direction[i]

                # ATR-based stops
                atr_distance_sl = atr * cfg.atr_sl_multiplier
                atr_distance_tp = atr * cfg.atr_tp_multiplier

                if direction == 2:  # Long
                    entry = close + total_entry_cost  # Worse fill for buyer
                    sl = entry - atr_distance_sl
                    tp = entry + atr_distance_tp
                else:  # Short (direction == 0)
                    entry = close - total_entry_cost  # Worse fill for seller
                    sl = entry + atr_distance_sl
                    tp = entry - atr_distance_tp

                # Position sizing: risk max_risk_per_trade of equity
                risk_amount = state.equity * cfg.max_risk_per_trade
                risk_per_unit = atr_distance_sl  # Price distance to SL
                position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

                # Apply leverage cap
                max_position = (state.equity * cfg.leverage) / entry if entry > 0 else 0
                position_size = min(position_size, max_position)

                if position_size > 0:
                    state.position_open = True
                    state.position_direction = direction
                    state.entry_price = entry
                    state.stop_loss = sl
                    state.take_profit = tp
                    state.position_size = position_size
                    state.entry_date = date
                    state.entry_confidence = pred_confidence[i]
                    state.entry_atr = atr

            # --- Update equity curve ---
            # Mark-to-market if position open
            if state.position_open:
                if state.position_direction == 2:  # Long
                    unrealized = (close - state.entry_price) * state.position_size
                else:  # Short
                    unrealized = (state.entry_price - close) * state.position_size
                current_equity = state.equity + unrealized
            else:
                current_equity = state.equity

            state.peak_equity = max(state.peak_equity, current_equity)
            state.equity_curve.append(current_equity)
            state.dates.append(date)

        # Close any remaining position at end
        if state.position_open:
            self._close_position(
                state, close_prices[-1], dates[-1], "end_of_period", total_entry_cost,
            )

        # Compute performance metrics
        metrics = self._compute_metrics(state)

        return {
            "trades": state.trades,
            "equity_curve": np.array(state.equity_curve),
            "dates": state.dates,
            "metrics": metrics,
            "circuit_breaker_triggered": state.circuit_breaker_active,
        }

    def _check_stops(self, state: BacktestState, high: float, low: float) -> Tuple[bool, bool]:
        """Check if SL or TP was hit (using high/low for realism)."""
        if state.position_direction == 2:  # Long
            hit_sl = low <= state.stop_loss
            hit_tp = high >= state.take_profit
        else:  # Short
            hit_sl = high >= state.stop_loss
            hit_tp = low <= state.take_profit

        # If both hit in same bar, assume SL hit first (conservative)
        if hit_sl and hit_tp:
            hit_tp = False

        return hit_sl, hit_tp

    def _close_position(
        self,
        state: BacktestState,
        exit_price: float,
        exit_date: pd.Timestamp,
        reason: str,
        cost: float,
    ):
        """Close the current position and record the trade."""
        # Apply exit cost
        if state.position_direction == 2:  # Long: sell at worse price
            effective_exit = exit_price - cost
        else:  # Short: buy back at worse price
            effective_exit = exit_price + cost

        # Calculate P&L
        if state.position_direction == 2:
            pnl = (effective_exit - state.entry_price) * state.position_size
        else:
            pnl = (state.entry_price - effective_exit) * state.position_size

        # Commission
        trade_value = state.entry_price * state.position_size
        commission = trade_value * self.cfg.commission_pct * 2  # Entry + exit
        pnl -= commission

        equity_at_entry = state.equity
        pnl_pct = pnl / equity_at_entry if equity_at_entry > 0 else 0

        state.equity += pnl
        state.peak_equity = max(state.peak_equity, state.equity)

        trade = Trade(
            entry_date=state.entry_date,
            exit_date=exit_date,
            direction=state.position_direction,
            entry_price=state.entry_price,
            exit_price=effective_exit,
            stop_loss=state.stop_loss,
            take_profit=state.take_profit,
            position_size=state.position_size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            confidence=state.entry_confidence,
            atr_at_entry=state.entry_atr,
        )
        state.trades.append(trade)

        # Reset position
        state.position_open = False
        state.entry_price = 0.0
        state.stop_loss = 0.0
        state.take_profit = 0.0
        state.position_size = 0.0
        state.entry_date = None

    def _compute_metrics(self, state: BacktestState) -> Dict:
        """Compute performance metrics from completed backtest."""
        trades = state.trades
        equity = np.array(state.equity_curve)
        initial = self.cfg.initial_capital

        if len(trades) == 0:
            return {
                "total_return_pct": 0.0,
                "n_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "avg_trade_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_consecutive_losses": 0,
                "avg_holding_days": 0.0,
                "final_equity": float(state.equity),
            }

        # Basic stats
        pnls = np.array([t.pnl for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        win_rate = len(wins) / len(trades) if trades else 0

        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-10
        profit_factor = gross_profit / gross_loss

        # Max drawdown from equity curve
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.where(peak > 0, peak, 1)
        max_dd = float(drawdown.max())

        # Daily returns for Sharpe/Sortino
        daily_returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0])
        daily_returns = daily_returns[np.isfinite(daily_returns)]

        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = float(np.sqrt(252) * daily_returns.mean() / daily_returns.std())
        else:
            sharpe = 0.0

        # Sortino (downside deviation)
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = float(np.sqrt(252) * daily_returns.mean() / downside.std())
        else:
            sortino = 0.0

        # Max consecutive losses
        max_consec_loss = 0
        current_streak = 0
        for pnl in pnls:
            if pnl <= 0:
                current_streak += 1
                max_consec_loss = max(max_consec_loss, current_streak)
            else:
                current_streak = 0

        # Average holding period
        holding_days = []
        for t in trades:
            if t.entry_date is not None:
                delta = t.exit_date - t.entry_date
                if hasattr(delta, 'days'):
                    holding_days.append(delta.days)
                else:
                    # numpy timedelta64
                    holding_days.append(int(delta / np.timedelta64(1, 'D')))
        avg_holding = float(np.mean(holding_days)) if holding_days else 0

        # Trade direction breakdown
        long_trades = [t for t in trades if t.direction == 2]
        short_trades = [t for t in trades if t.direction == 0]

        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        return {
            "total_return_pct": float((state.equity - initial) / initial * 100),
            "final_equity": float(state.equity),
            "n_trades": len(trades),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "max_drawdown_pct": float(max_dd * 100),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "avg_trade_pnl": float(pnls.mean()),
            "avg_win": float(wins.mean()) if len(wins) > 0 else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) > 0 else 0.0,
            "max_consecutive_losses": max_consec_loss,
            "avg_holding_days": avg_holding,
            "n_long": len(long_trades),
            "n_short": len(short_trades),
            "long_win_rate": float(
                sum(1 for t in long_trades if t.pnl > 0) / len(long_trades)
            ) if long_trades else 0.0,
            "short_win_rate": float(
                sum(1 for t in short_trades if t.pnl > 0) / len(short_trades)
            ) if short_trades else 0.0,
            "exit_reasons": exit_reasons,
        }


# ===========================================================================
# 3. Walk-Forward Backtest Runner
# ===========================================================================

def backtest_walk_forward(
    fold_results: List[Dict],
    test_dfs: List[pd.DataFrame],
    cfg: BacktestConfig,
    seq_len: int = 30,
) -> Dict:
    """
    Run backtester on each walk-forward fold's test predictions.

    Args:
        fold_results: list of fold result dicts from trainer.
        test_dfs: list of test DataFrames (with OHLCV + ATR_14) per fold.
        cfg: BacktestConfig.
        seq_len: sequence length (predictions start seq_len rows into test_df).

    Returns:
        Dict with per-fold and aggregate backtest results.
    """
    backtester = Backtester(cfg)
    fold_backtests = []

    for fold_result, test_df in zip(fold_results, test_dfs):
        fold_id = fold_result["fold_id"]
        preds = fold_result["test_predictions"]

        # Predictions start seq_len rows into the test_df
        aligned_df = test_df.iloc[seq_len:].copy()
        n_preds = len(preds["direction"])
        n_rows = len(aligned_df)
        n = min(n_preds, n_rows)

        if n < 5:
            print(f"   ⚠️  Fold {fold_id}: only {n} aligned rows, skipping")
            continue

        aligned_df = aligned_df.iloc[:n]

        # Extract OHLCV + ATR
        dates = aligned_df.index.values
        close = aligned_df["Close"].values
        high = aligned_df["High"].values
        low = aligned_df["Low"].values
        atr = aligned_df["ATR_14"].values

        # Direction predictions + confidence
        dir_logits = preds["direction"][:n]
        dir_pred = dir_logits.argmax(axis=1)

        # Softmax for confidence
        logits_exp = np.exp(dir_logits - dir_logits.max(axis=1, keepdims=True))
        confidence = (logits_exp / logits_exp.sum(axis=1, keepdims=True)).max(axis=1)

        result = backtester.run(dates, close, high, low, atr, dir_pred, confidence)
        result["fold_id"] = fold_id
        fold_backtests.append(result)

        m = result["metrics"]
        print(f"   📈 Fold {fold_id}: {m['n_trades']} trades | "
              f"return={m['total_return_pct']:+.2f}% | "
              f"win={m['win_rate']:.1%} | "
              f"sharpe={m['sharpe_ratio']:.2f} | "
              f"maxDD={m['max_drawdown_pct']:.1f}%"
              f"{' ⛔ CB' if result['circuit_breaker_triggered'] else ''}")

    # Aggregate
    agg = _aggregate_backtest_results(fold_backtests, cfg.initial_capital)
    return {"per_fold": fold_backtests, "aggregate": agg}


def _aggregate_backtest_results(fold_backtests: List[Dict], initial_capital: float) -> Dict:
    """Aggregate backtest metrics across folds."""
    if not fold_backtests:
        return {"n_folds": 0}

    metrics_list = [fb["metrics"] for fb in fold_backtests]

    scalar_keys = [
        "total_return_pct", "win_rate", "profit_factor",
        "max_drawdown_pct", "sharpe_ratio", "sortino_ratio",
        "avg_trade_pnl", "avg_holding_days", "n_trades",
    ]

    agg = {"n_folds": len(fold_backtests)}
    for key in scalar_keys:
        values = [m[key] for m in metrics_list]
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_std"] = float(np.std(values))
        agg[f"{key}_values"] = values

    # Total trades and wins across all folds
    total_trades = sum(m["n_trades"] for m in metrics_list)
    total_wins = sum(int(m["win_rate"] * m["n_trades"]) for m in metrics_list)
    agg["total_trades"] = total_trades
    agg["pooled_win_rate"] = total_wins / total_trades if total_trades > 0 else 0

    # Circuit breaker count
    agg["circuit_breaker_count"] = sum(1 for fb in fold_backtests if fb["circuit_breaker_triggered"])

    return agg


# ===========================================================================
# 4. Pretty Printing
# ===========================================================================

def print_backtest_report(result: Dict):
    """Print backtest results for a single fold or aggregate."""
    m = result["metrics"]

    print(f"\n  {'─' * 45}")
    print(f"  📈 Backtest Results")
    print(f"  {'─' * 45}")
    print(f"    Total Return:     {m['total_return_pct']:+.2f}%")
    print(f"    Final Equity:     ${m['final_equity']:,.2f}")
    print(f"    Trades:           {m['n_trades']} (L:{m.get('n_long',0)} / S:{m.get('n_short',0)})")
    print(f"    Win Rate:         {m['win_rate']:.1%}")
    print(f"    Profit Factor:    {m['profit_factor']:.2f}")
    print(f"    Sharpe Ratio:     {m['sharpe_ratio']:.2f}")
    print(f"    Sortino Ratio:    {m['sortino_ratio']:.2f}")
    print(f"    Max Drawdown:     {m['max_drawdown_pct']:.1f}%")
    print(f"    Avg Trade P&L:    ${m['avg_trade_pnl']:.2f}")
    print(f"    Avg Win:          ${m['avg_win']:.2f}")
    print(f"    Avg Loss:         ${m['avg_loss']:.2f}")
    print(f"    Max Consec Loss:  {m['max_consecutive_losses']}")
    print(f"    Avg Holding:      {m['avg_holding_days']:.1f} days")
    if "exit_reasons" in m:
        print(f"    Exit Reasons:     {m['exit_reasons']}")
    if result.get("circuit_breaker_triggered"):
        print(f"    ⛔ Circuit breaker triggered!")


def print_aggregate_backtest_report(agg: Dict):
    """Print aggregated backtest results across folds."""
    n = agg["n_folds"]
    if n == 0:
        print("  No backtest results to aggregate.")
        return

    print(f"\n{'=' * 55}")
    print(f"📈 Walk-Forward Backtest Summary ({n} folds)")
    print(f"{'=' * 55}")
    print(f"    Return:       {agg['total_return_pct_mean']:+.2f}% ± {agg['total_return_pct_std']:.2f}%")
    print(f"    Win Rate:     {agg['win_rate_mean']:.1%} ± {agg['win_rate_std']:.1%}")
    print(f"    Profit Factor:{agg['profit_factor_mean']:.2f} ± {agg['profit_factor_std']:.2f}")
    print(f"    Sharpe:       {agg['sharpe_ratio_mean']:.2f} ± {agg['sharpe_ratio_std']:.2f}")
    print(f"    Sortino:      {agg['sortino_ratio_mean']:.2f} ± {agg['sortino_ratio_std']:.2f}")
    print(f"    Max DD:       {agg['max_drawdown_pct_mean']:.1f}% ± {agg['max_drawdown_pct_std']:.1f}%")
    print(f"    Avg Trades/fold: {agg['n_trades_mean']:.0f}")
    print(f"    Total Trades: {agg['total_trades']}")
    print(f"    Pooled WR:    {agg['pooled_win_rate']:.1%}")
    if agg["circuit_breaker_count"] > 0:
        print(f"    ⛔ Circuit breaker triggered in {agg['circuit_breaker_count']}/{n} folds")


# ===========================================================================
# 5. Smoke Test
# ===========================================================================

def test_backtester():
    """Smoke test with synthetic price data and predictions."""
    np.random.seed(42)

    print("=" * 60)
    print("🧪 Backtester Smoke Test")
    print("=" * 60)

    cfg = BacktestConfig()

    # Synthetic EUR/USD-like data
    n_days = 250
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    base_price = 1.0800
    returns = np.random.randn(n_days) * 0.005
    close = base_price + np.cumsum(returns)
    high = close + np.abs(np.random.randn(n_days) * 0.003)
    low = close - np.abs(np.random.randn(n_days) * 0.003)
    atr = pd.Series(high - low).rolling(14).mean().fillna(0.005).values

    # Synthetic predictions with slight edge
    true_direction = np.where(returns > 0.002, 2, np.where(returns < -0.002, 0, 1))
    # Model has ~55% accuracy on directional calls
    pred_direction = true_direction.copy()
    noise_mask = np.random.random(n_days) < 0.45
    pred_direction[noise_mask] = np.random.randint(0, 3, noise_mask.sum())

    confidence = np.random.uniform(0.4, 0.8, n_days)

    # 1. Basic backtest
    print("\n1️⃣  Running basic backtest...")
    backtester = Backtester(cfg)
    result = backtester.run(dates, close, high, low, atr, pred_direction, confidence)
    print_backtest_report(result)

    m = result["metrics"]
    assert m["n_trades"] > 0, "No trades executed!"
    assert len(result["equity_curve"]) == n_days
    print(f"   ✅ Basic backtest OK ({m['n_trades']} trades)")

    # 2. Test circuit breaker
    print("\n2️⃣  Testing circuit breaker...")
    bad_cfg = BacktestConfig(max_drawdown_pct=0.05)  # Very tight 5% DD limit
    bad_preds = np.where(true_direction == 2, 0, 2)  # Always wrong
    bad_confidence = np.ones(n_days) * 0.9  # High confidence = always trade

    bad_result = Backtester(bad_cfg).run(dates, close, high, low, atr, bad_preds, bad_confidence)
    assert bad_result["circuit_breaker_triggered"], "Circuit breaker should have triggered!"
    print(f"   ✅ Circuit breaker triggered after {bad_result['metrics']['n_trades']} trades")

    # 3. Test confidence filtering
    print("\n3️⃣  Testing confidence filtering...")
    low_conf = np.ones(n_days) * 0.3  # Below threshold
    no_trade_result = backtester.run(dates, close, high, low, atr, pred_direction, low_conf)
    assert no_trade_result["metrics"]["n_trades"] == 0, "Should have 0 trades with low confidence"
    print("   ✅ No trades with low confidence")

    # 4. Test walk-forward backtest
    print("\n4️⃣  Testing walk-forward backtest runner...")
    fake_folds = []
    fake_dfs = []
    for fold_id in range(3):
        n = 60
        logits = np.random.randn(n, 3)
        logits[np.arange(n), np.random.randint(0, 3, n)] += 1.0

        fold_dates = pd.bdate_range(f"2023-{fold_id*3+1:02d}-01", periods=n + 30)
        fold_close = base_price + np.cumsum(np.random.randn(n + 30) * 0.005)
        fold_high = fold_close + np.abs(np.random.randn(n + 30) * 0.003)
        fold_low = fold_close - np.abs(np.random.randn(n + 30) * 0.003)
        fold_atr = pd.Series(fold_high - fold_low).rolling(14).mean().fillna(0.005).values

        fold_df = pd.DataFrame({
            "Close": fold_close, "High": fold_high, "Low": fold_low,
            "ATR_14": fold_atr,
        }, index=fold_dates)

        fake_folds.append({
            "fold_id": fold_id,
            "test_predictions": {
                "direction": logits,
                "return": np.random.randn(n) * 0.01,
                "volatility": np.abs(np.random.randn(n) * 0.005),
            },
            "test_targets": {
                "direction": np.random.randint(0, 3, n),
                "return": np.random.randn(n) * 0.01,
                "volatility": np.abs(np.random.randn(n) * 0.005),
            },
        })
        fake_dfs.append(fold_df)

    wf_result = backtest_walk_forward(fake_folds, fake_dfs, cfg, seq_len=30)
    print_aggregate_backtest_report(wf_result["aggregate"])
    assert wf_result["aggregate"]["n_folds"] == 3
    print("   ✅ Walk-forward backtest OK")

    print(f"\n✅ All backtester tests passed!")


if __name__ == "__main__":
    test_backtester()
