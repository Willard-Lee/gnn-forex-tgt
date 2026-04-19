"""
Streamlit Dashboard for Temporal Graph Transformer v2.

Visualizes:
  - Data pipeline: price chart, indicator heatmap, target distribution
  - Model performance: confusion matrices, equity curves, metric comparisons
  - Walk-forward results: per-fold breakdown
  - Baseline comparison: TGT vs LSTM/RF/MA/B&H

Launch: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="TGT EUR/USD Forecaster",
    page_icon="📈",
    layout="wide",
)


# ===========================================================================
# Sidebar
# ===========================================================================

st.sidebar.title("📈 TGT Dashboard")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📦 Data Pipeline",
    "📊 Model Results",
    "🔬 Baseline Comparison",
    "💹 Backtest",
])


# ===========================================================================
# Helper functions
# ===========================================================================

@st.cache_data
def load_results(path: str = "results/experiment.pkl"):
    """Load saved experiment results."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data
def run_pipeline_cached():
    """Run data pipeline (cached)."""
    from configs.config import Config
    from utils.data_pipeline import run_pipeline
    cfg = Config()
    return run_pipeline(cfg)


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix"):
    """Render confusion matrix as a styled DataFrame."""
    labels = ["Down", "Flat", "Up"]
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.index.name = "True"
    df_cm.columns.name = "Predicted"
    st.dataframe(df_cm.style.background_gradient(cmap="Blues"), use_container_width=True)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ===========================================================================
# Pages
# ===========================================================================

if page == "🏠 Overview":
    st.title("🧠 Temporal Graph Transformer — EUR/USD Forecasting v2")
    st.markdown("""
    ### Architecture
    ```
    For each t in [1..30]:
        24 indicator nodes × 4 features → GAT attention → pool → 128-dim snapshot
    Stack 30 snapshots → Transformer encoder → [CLS] → shared FC →
        Head 1: direction (up/flat/down)
        Head 2: returns (scalar)
        Head 3: volatility (scalar)
    ```

    ### Key Design Choices
    - **GAT at every timestep** — not just the last one
    - **Dynamic graph** — recomputed every 20 trading days (Pearson + DCC proxy + Granger)
    - **Walk-forward validation** — expanding window, no single-split leakage
    - **Multi-task learning** — direction (CE ×0.2) + returns (MSE ×0.4) + volatility (MAE ×0.4)

    ### How to Use
    1. Run `python main.py --mode full` to generate results
    2. View them here in the dashboard

    ⚠️ *Academic proof-of-concept. NOT financial advice.*
    """)

    # Check for results
    results = load_results()
    if results:
        st.success("✅ Experiment results loaded!")
        if "tgt_results" in results:
            st.metric("TGT Folds", len(results["tgt_results"]))
        if "baseline_results" in results:
            st.metric("Baselines", len(results["baseline_results"]))
    else:
        st.info("No results found. Run `python main.py --mode full` first.")


elif page == "📦 Data Pipeline":
    st.title("📦 Data Pipeline")

    try:
        pipeline = run_pipeline_cached()
        df = pipeline["df_full"]
        feat_cols = pipeline["feature_cols"]

        # Price chart
        st.subheader("EUR/USD Price")
        st.line_chart(df["Close"])

        # Indicator overview
        st.subheader("Technical Indicators")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Indicators", len(feat_cols))
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Date Range", f"{df.index.min().date()} → {df.index.max().date()}")
            st.metric("Missing Values", int(df[feat_cols].isna().sum().sum()))

        # Indicator selector
        selected = st.multiselect("Plot indicators", feat_cols, default=feat_cols[:3])
        if selected:
            st.line_chart(df[selected])

        # Target distribution
        st.subheader("Target Distribution")
        dir_counts = df["target_direction"].value_counts().sort_index()
        labels = {0: "Down", 1: "Flat", 2: "Up"}
        dist_df = pd.DataFrame({
            "Direction": [labels[i] for i in dir_counts.index],
            "Count": dir_counts.values,
            "Percentage": (dir_counts.values / len(df) * 100).round(1),
        })
        st.dataframe(dist_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Return Distribution")
            st.bar_chart(pd.cut(df["target_return"], bins=50).value_counts().sort_index())
        with col2:
            st.subheader("Volatility Distribution")
            st.bar_chart(pd.cut(df["target_volatility"], bins=50).value_counts().sort_index())

        # Sequence shapes
        st.subheader("Sequence Shapes")
        st.json({
            "train": list(pipeline["train_X"].shape),
            "val": list(pipeline["val_X"].shape),
            "test": list(pipeline["test_X"].shape),
            "features_per_timestep": len(feat_cols),
        })

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.info("Make sure data is available. Run `python main.py --mode pipeline` first.")


elif page == "📊 Model Results":
    st.title("📊 Model Results")

    results = load_results()
    if not results or "tgt_results" not in results:
        st.warning("No TGT results found. Run `python main.py --mode full` first.")
        st.stop()

    tgt_results = results["tgt_results"]

    from utils.evaluator import evaluate_fold, aggregate_fold_metrics

    fold_metrics = [evaluate_fold(r) for r in tgt_results]
    agg = aggregate_fold_metrics(fold_metrics)

    # Top-level metrics
    st.subheader("Walk-Forward Aggregate")
    d = agg["direction"]
    r = agg["return"]
    v = agg["volatility"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Direction Accuracy", f"{d['accuracy_mean']:.1%} ± {d['accuracy_std']:.1%}")
    col2.metric("Macro F1", f"{d['macro_f1_mean']:.3f}")
    col3.metric("Return R²", f"{r['r2_mean']:.3f}")
    col4.metric("Volatility MAE", f"{v['mae_mean']:.6f}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("IC (Spearman)", f"{r['ic_mean']:.3f}")
    col2.metric("Dir Agreement", f"{r['directional_agreement_mean']:.1%}")
    col3.metric("Vol R²", f"{v['r2_mean']:.3f}")
    col4.metric("Folds", agg["n_folds"])

    # Pooled confusion matrix
    st.subheader("Pooled Confusion Matrix")
    plot_confusion_matrix(d["pooled_confusion_matrix"])

    # Per-fold breakdown
    st.subheader("Per-Fold Results")
    fold_table = []
    for m in fold_metrics:
        fold_table.append({
            "Fold": m["fold_id"],
            "Best Epoch": m["best_epoch"],
            "Dir Acc": f"{m['direction']['accuracy']:.3f}",
            "Macro F1": f"{m['direction']['macro_f1']:.3f}",
            "Ret R²": f"{m['return']['r2']:.3f}",
            "IC": f"{m['return']['ic']:.3f}",
            "Vol MAE": f"{m['volatility']['mae']:.6f}",
        })
    st.dataframe(pd.DataFrame(fold_table), use_container_width=True)

    # Per-fold accuracy chart
    st.subheader("Direction Accuracy Across Folds")
    acc_df = pd.DataFrame({
        "Fold": [m["fold_id"] for m in fold_metrics],
        "Accuracy": [m["direction"]["accuracy"] for m in fold_metrics],
    })
    st.bar_chart(acc_df.set_index("Fold"))

    # Per-fold confusion matrices
    with st.expander("Per-Fold Confusion Matrices"):
        for m in fold_metrics:
            st.markdown(f"**Fold {m['fold_id']}** (acc={m['direction']['accuracy']:.3f})")
            plot_confusion_matrix(m["direction"]["confusion_matrix"])


elif page == "🔬 Baseline Comparison":
    st.title("🔬 Baseline Comparison")

    results = load_results()
    if not results:
        st.warning("No results found. Run `python main.py --mode full` first.")
        st.stop()

    tgt_results = results.get("tgt_results", [])
    bl_results = results.get("baseline_results", {})

    if not tgt_results or not bl_results:
        st.warning("Need both TGT and baseline results. Run `python main.py --mode full`.")
        st.stop()

    from utils.evaluator import evaluate_fold, aggregate_fold_metrics, compare_models

    # Aggregate all models
    all_models = {"TGT": tgt_results}
    all_models.update(bl_results)

    comparison_data = []
    for name, fold_results in all_models.items():
        if not fold_results:
            continue
        metrics = [evaluate_fold(r) for r in fold_results]
        agg = aggregate_fold_metrics(metrics)
        comparison_data.append({
            "Model": name,
            "Dir Acc": f"{agg['direction']['accuracy_mean']:.3f} ± {agg['direction']['accuracy_std']:.3f}",
            "Macro F1": f"{agg['direction']['macro_f1_mean']:.3f}",
            "Ret R²": f"{agg['return']['r2_mean']:.3f}",
            "IC": f"{agg['return']['ic_mean']:.3f}",
            "Vol MAE": f"{agg['volatility']['mae_mean']:.6f}",
            "Folds": agg["n_folds"],
        })

    st.subheader("Model Comparison")
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    # Accuracy comparison chart
    st.subheader("Direction Accuracy by Model")
    acc_chart_data = {}
    for name, fold_results in all_models.items():
        if fold_results:
            accs = [evaluate_fold(r)["direction"]["accuracy"] for r in fold_results]
            acc_chart_data[name] = np.mean(accs)
    st.bar_chart(pd.DataFrame(acc_chart_data, index=["Accuracy"]).T)

    # Statistical significance
    st.subheader("Statistical Significance (TGT vs Baselines)")
    for bl_name, bl_folds in bl_results.items():
        if bl_folds and len(bl_folds) == len(tgt_results):
            comp = compare_models(tgt_results, bl_folds, "TGT", bl_name)

            with st.expander(f"TGT vs {bl_name}"):
                for key in ["direction_accuracy", "return_mse", "volatility_mae"]:
                    t = comp[key]
                    sig = "✅ p<0.05" if t["significant_005"] else ("⚠️ p<0.10" if t["significant_010"] else "❌ n.s.")
                    st.markdown(f"**{t['metric']}**: t={t['t_stat']:.3f}, p={t['p_value']:.4f} {sig} "
                                f"(d={t['cohens_d']:.3f})")

                for key in ["dm_return_mse", "dm_return_mae"]:
                    dm = comp[key]
                    sig = "✅" if dm["model_better"] else "❌"
                    st.markdown(f"**DM ({dm.get('loss_fn', key)})**: stat={dm['dm_stat']:.3f}, "
                                f"p={dm['p_value']:.4f} {sig}")


elif page == "💹 Backtest":
    st.title("💹 Backtest Results")

    results = load_results()
    if not results or "tgt_results" not in results:
        st.warning("No results found. Run `python main.py --mode full` first.")
        st.stop()

    st.markdown("""
    > Backtest uses ATR-based stops (SL=2×ATR, TP=3×ATR), 2% risk per trade,
    > confidence threshold filtering, and 20% max drawdown circuit breaker.
    >
    > ⚠️ *Past simulated performance does not indicate future results.*
    """)

    # Try to run backtest if we have the data
    try:
        pipeline = run_pipeline_cached()
        df_full = pipeline["df_full"]
        tgt_results = results["tgt_results"]

        from configs.config import Config
        from utils.backtester import Backtester, backtest_walk_forward

        cfg = Config()

        st.subheader("Backtest Configuration")
        col1, col2, col3 = st.columns(3)
        col1.metric("Initial Capital", f"${cfg.backtest.initial_capital:,.0f}")
        col1.metric("Leverage", f"{cfg.backtest.leverage}x")
        col2.metric("Spread", f"{cfg.backtest.spread_pips} pips")
        col2.metric("Slippage", f"{cfg.backtest.slippage_pips} pips")
        col3.metric("SL Multiplier", f"{cfg.backtest.atr_sl_multiplier}× ATR")
        col3.metric("TP Multiplier", f"{cfg.backtest.atr_tp_multiplier}× ATR")

        # Run backtest on single-split results
        if len(tgt_results) == 1:
            # Single split mode
            test_df = pipeline["test_df"]
            preds = tgt_results[0]["test_predictions"]
            seq_len = cfg.data.sequence_length

            aligned = test_df.iloc[seq_len:]
            n = min(len(preds["direction"]), len(aligned))
            aligned = aligned.iloc[:n]

            dir_logits = preds["direction"][:n]
            dir_pred = dir_logits.argmax(axis=1)
            probs = softmax(dir_logits)
            confidence = probs.max(axis=1)

            backtester = Backtester(cfg.backtest)
            bt_result = backtester.run(
                aligned.index.values,
                aligned["Close"].values,
                aligned["High"].values,
                aligned["Low"].values,
                aligned["ATR_14"].values,
                dir_pred,
                confidence,
            )

            m = bt_result["metrics"]

            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{m['total_return_pct']:+.2f}%")
            col2.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
            col3.metric("Win Rate", f"{m['win_rate']:.1%}")
            col4.metric("Max Drawdown", f"{m['max_drawdown_pct']:.1f}%")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Trades", m["n_trades"])
            col2.metric("Profit Factor", f"{m['profit_factor']:.2f}")
            col3.metric("Sortino", f"{m['sortino_ratio']:.2f}")
            col4.metric("Avg Holding", f"{m['avg_holding_days']:.1f} days")

            if bt_result["circuit_breaker_triggered"]:
                st.error("⛔ Circuit breaker triggered!")

            # Equity curve
            st.subheader("Equity Curve")
            eq_df = pd.DataFrame({
                "Equity": bt_result["equity_curve"],
            }, index=bt_result["dates"])
            st.line_chart(eq_df)

            # Trade log
            if bt_result["trades"]:
                st.subheader(f"Trade Log ({m['n_trades']} trades)")
                trade_data = []
                for t in bt_result["trades"]:
                    trade_data.append({
                        "Entry": str(t.entry_date)[:10] if t.entry_date else "",
                        "Exit": str(t.exit_date)[:10],
                        "Dir": "Long" if t.direction == 2 else "Short",
                        "P&L": f"${t.pnl:+,.2f}",
                        "P&L %": f"{t.pnl_pct:+.2%}",
                        "Exit Reason": t.exit_reason,
                        "Confidence": f"{t.confidence:.2f}",
                    })
                st.dataframe(pd.DataFrame(trade_data), use_container_width=True)

                # Exit reason breakdown
                st.subheader("Exit Reasons")
                st.bar_chart(pd.DataFrame(m["exit_reasons"], index=["Count"]).T)
        else:
            st.info("Walk-forward backtest view — showing per-fold summaries.")
            st.markdown("Run with `--mode full` to see detailed single-split backtest.")

    except Exception as e:
        st.error(f"Backtest error: {e}")
        st.info("Ensure data pipeline has been run first.")
