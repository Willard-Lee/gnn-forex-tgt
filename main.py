"""
CLI entry point for Temporal Graph Transformer v2.

Usage:
    python main.py --mode pipeline       # Run data pipeline only
    python main.py --mode train          # Train TGT (single split, quick test)
    python main.py --mode walkforward    # Full walk-forward training
    python main.py --mode baselines      # Run all baselines
    python main.py --mode full           # Pipeline + TGT + baselines + evaluate
    python main.py --mode dashboard      # Launch Streamlit dashboard
"""

import argparse
import json
import os
import pickle
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

from configs.config import Config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline(cfg: Config) -> dict:
    """Run data pipeline and return processed data."""
    from utils.data_pipeline import run_pipeline as _run_pipeline
    print("\n📦 Running data pipeline...")
    return _run_pipeline(cfg)


def run_train_single(cfg: Config, pipeline_output: dict) -> dict:
    """Train TGT on single split."""
    from utils.trainer import WalkForwardTrainer
    print("\n🚀 Training TGT (single split)...")
    trainer = WalkForwardTrainer(cfg)
    result = trainer.train_single_split(pipeline_output)
    return result


def run_walkforward(cfg: Config, df) -> list:
    """Run full walk-forward training."""
    from utils.trainer import WalkForwardTrainer
    feature_cols = [f for f in cfg.data.feature_nodes if f in df.columns]
    trainer = WalkForwardTrainer(cfg)
    return trainer.train_walk_forward(df, feature_cols)


def run_baselines(cfg: Config, df) -> dict:
    """Run all baselines."""
    from baselines.run_baselines import run_all_baselines
    return run_all_baselines(df, cfg)


def run_evaluate(tgt_results: list, baseline_results: dict):
    """Evaluate TGT and compare against baselines."""
    from utils.evaluator import (
        evaluate_fold, aggregate_fold_metrics,
        compare_models, print_aggregate_report, print_comparison_report,
    )

    print("\n📊 Evaluating TGT...")
    tgt_metrics = [evaluate_fold(r) for r in tgt_results]
    tgt_agg = aggregate_fold_metrics(tgt_metrics)
    print_aggregate_report(tgt_agg)

    for name, bl_results in baseline_results.items():
        if bl_results and len(bl_results) == len(tgt_results):
            comp = compare_models(tgt_results, bl_results, "TGT", name)
            print_comparison_report(comp)

    return tgt_agg


def save_results(results: dict, path: str = "results/experiment.pkl"):
    """Save experiment results to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert model states to CPU before saving
    clean = {}
    for k, v in results.items():
        if k == "tgt_results" and isinstance(v, list):
            clean_folds = []
            for fold in v:
                fold_copy = {fk: fv for fk, fv in fold.items() if fk != "model_state"}
                clean_folds.append(fold_copy)
            clean[k] = clean_folds
        else:
            clean[k] = v
    with open(path, "wb") as f:
        pickle.dump(clean, f)
    print(f"\n💾 Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="TGT EUR/USD Forecasting v2")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["pipeline", "train", "walkforward", "baselines", "full", "dashboard"],
                        help="Execution mode")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda/mps)")
    parser.add_argument("--save", type=str, default="results/experiment.pkl", help="Save path")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.device:
        cfg.device = args.device
    elif torch.cuda.is_available():
        cfg.device = "cuda"
    elif torch.backends.mps.is_available():
        cfg.device = "mps"

    set_seed(cfg.train.seed)

    print("=" * 60)
    print("🧠 Temporal Graph Transformer — EUR/USD Forecasting v2")
    print(f"   Mode: {args.mode} | Device: {cfg.device}")
    print("=" * 60)

    if args.mode == "dashboard":
        os.system("streamlit run app.py")
        return

    # Pipeline (needed for all modes except dashboard)
    pipeline_output = run_pipeline(cfg)
    df_full = pipeline_output["df_full"]

    if args.mode == "pipeline":
        print("\n✅ Pipeline complete.")
        return

    if args.mode == "train":
        result = run_train_single(cfg, pipeline_output)
        save_results({"tgt_results": [result]}, args.save)
        return

    if args.mode == "walkforward":
        tgt_results = run_walkforward(cfg, df_full)
        save_results({"tgt_results": tgt_results}, args.save)
        return

    if args.mode == "baselines":
        bl_results = run_baselines(cfg, df_full)
        save_results({"baseline_results": bl_results}, args.save)
        return

    if args.mode == "full":
        # Full experiment
        tgt_results = run_walkforward(cfg, df_full)
        bl_results = run_baselines(cfg, df_full)
        tgt_agg = run_evaluate(tgt_results, bl_results)
        save_results({
            "tgt_results": tgt_results,
            "baseline_results": bl_results,
            "tgt_aggregate": tgt_agg,
        }, args.save)
        print("\n✅ Full experiment complete!")


if __name__ == "__main__":
    main()
