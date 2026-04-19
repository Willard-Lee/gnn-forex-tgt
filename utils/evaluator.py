"""
Evaluator for Temporal Graph Transformer

Computes metrics across walk-forward folds:
  - Direction: accuracy, per-class precision/recall/F1, confusion matrix
  - Returns: MSE, MAE, R², directional agreement
  - Volatility: MAE, RMSE, R²
  - Statistical significance: paired t-tests, Diebold-Mariano test
  - Aggregate: mean ± std across folds, pooled confusion matrix
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats as sp_stats
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


# ===========================================================================
# 1. Per-Fold Metrics
# ===========================================================================

DIRECTION_LABELS = {0: "Down", 1: "Flat", 2: "Up"}


def compute_direction_metrics(
    y_true: np.ndarray,
    y_pred_logits: np.ndarray,
) -> Dict:
    """
    Compute direction classification metrics.

    Args:
        y_true: (N,) int array of true classes {0, 1, 2}.
        y_pred_logits: (N, 3) float array of logits.

    Returns:
        Dict with accuracy, per-class P/R/F1, confusion matrix, class distribution.
    """
    y_pred = y_pred_logits.argmax(axis=1)
    n = len(y_true)

    acc = accuracy_score(y_true, y_pred)

    # Per-class metrics
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0,
    )

    per_class = {}
    for i, label in DIRECTION_LABELS.items():
        per_class[label] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    # Macro and weighted F1
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Confidence from softmax
    probs = _softmax(y_pred_logits)
    confidence = probs.max(axis=1)

    # Class distribution
    true_dist = {DIRECTION_LABELS[i]: int((y_true == i).sum()) for i in range(3)}
    pred_dist = {DIRECTION_LABELS[i]: int((y_pred == i).sum()) for i in range(3)}

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
        "mean_confidence": float(confidence.mean()),
        "true_distribution": true_dist,
        "pred_distribution": pred_dist,
        "n_samples": n,
    }


def compute_return_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """
    Compute return prediction metrics.

    Args:
        y_true: (N,) actual log returns.
        y_pred: (N,) predicted log returns.

    Returns:
        Dict with MSE, RMSE, MAE, R², directional agreement.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0

    # Directional agreement: did we get the sign right?
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    dir_agreement = float((true_sign == pred_sign).mean())

    # IC (information coefficient): rank correlation
    if len(y_true) > 2:
        ic, ic_pval = sp_stats.spearmanr(y_true, y_pred)
    else:
        ic, ic_pval = 0.0, 1.0

    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mae),
        "r2": float(r2),
        "directional_agreement": dir_agreement,
        "ic": float(ic) if np.isfinite(ic) else 0.0,
        "ic_pval": float(ic_pval) if np.isfinite(ic_pval) else 1.0,
        "n_samples": len(y_true),
    }


def compute_volatility_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """
    Compute volatility prediction metrics.

    Args:
        y_true: (N,) actual volatility.
        y_pred: (N,) predicted volatility.

    Returns:
        Dict with MAE, RMSE, R², QLIKE (quasi-likelihood loss).
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0

    # QLIKE: quasi-likelihood loss for volatility (Patton, 2011)
    # QLIKE = mean(y_true / y_pred + log(y_pred))
    # Lower is better. Only valid when y_pred > 0.
    y_pred_safe = np.maximum(y_pred, 1e-8)
    y_true_safe = np.maximum(y_true, 1e-8)
    qlike = float(np.mean(y_true_safe / y_pred_safe + np.log(y_pred_safe)))

    return {
        "mae": float(mae),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2),
        "qlike": qlike,
        "n_samples": len(y_true),
    }


def evaluate_fold(fold_result: Dict) -> Dict:
    """
    Compute all metrics for a single fold.

    Args:
        fold_result: output from WalkForwardTrainer._train_fold()

    Returns:
        Dict with direction, return, volatility metrics.
    """
    preds = fold_result["test_predictions"]
    targets = fold_result["test_targets"]

    return {
        "fold_id": fold_result["fold_id"],
        "best_epoch": fold_result["best_epoch"],
        "direction": compute_direction_metrics(
            targets["direction"], preds["direction"],
        ),
        "return": compute_return_metrics(
            targets["return"], preds["return"],
        ),
        "volatility": compute_volatility_metrics(
            targets["volatility"], preds["volatility"],
        ),
    }


# ===========================================================================
# 2. Cross-Fold Aggregation
# ===========================================================================

def aggregate_fold_metrics(fold_metrics: List[Dict]) -> Dict:
    """
    Aggregate metrics across walk-forward folds.

    Returns mean ± std for scalar metrics, pooled confusion matrix,
    and per-fold detail.
    """
    n_folds = len(fold_metrics)

    # Collect scalar metrics across folds
    dir_scalars = _collect_scalars([f["direction"] for f in fold_metrics],
                                   ["accuracy", "macro_f1", "weighted_f1", "mean_confidence"])
    ret_scalars = _collect_scalars([f["return"] for f in fold_metrics],
                                   ["mse", "rmse", "mae", "r2", "directional_agreement", "ic"])
    vol_scalars = _collect_scalars([f["volatility"] for f in fold_metrics],
                                   ["mae", "rmse", "r2", "qlike"])

    # Pooled confusion matrix
    pooled_cm = sum(f["direction"]["confusion_matrix"] for f in fold_metrics)

    # Pooled accuracy from confusion matrix
    pooled_acc = float(np.trace(pooled_cm) / pooled_cm.sum()) if pooled_cm.sum() > 0 else 0.0

    return {
        "n_folds": n_folds,
        "direction": {
            **dir_scalars,
            "pooled_confusion_matrix": pooled_cm,
            "pooled_accuracy": pooled_acc,
        },
        "return": ret_scalars,
        "volatility": vol_scalars,
        "per_fold": fold_metrics,
    }


def _collect_scalars(metric_dicts: List[Dict], keys: List[str]) -> Dict:
    """Compute mean ± std for selected scalar keys across folds."""
    result = {}
    for key in keys:
        values = [d[key] for d in metric_dicts if key in d]
        if values:
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
            result[f"{key}_values"] = values
    return result


# ===========================================================================
# 3. Statistical Significance Tests
# ===========================================================================

def paired_t_test(
    metric_model: List[float],
    metric_baseline: List[float],
    metric_name: str = "metric",
) -> Dict:
    """
    Paired t-test: is the model significantly better than baseline?

    Args:
        metric_model: per-fold metric values for the model.
        metric_baseline: per-fold metric values for the baseline.
        metric_name: name for reporting.

    Returns:
        Dict with t-statistic, p-value, significance, effect size (Cohen's d).
    """
    assert len(metric_model) == len(metric_baseline), "Need same number of folds"
    n = len(metric_model)

    if n < 2:
        return {
            "metric": metric_name,
            "n_folds": n,
            "t_stat": 0.0,
            "p_value": 1.0,
            "significant_005": False,
            "significant_010": False,
            "cohens_d": 0.0,
            "note": "Need ≥2 folds for t-test",
        }

    diffs = np.array(metric_model) - np.array(metric_baseline)
    t_stat, p_value = sp_stats.ttest_rel(metric_model, metric_baseline)

    # Cohen's d for paired samples
    d_mean = np.mean(diffs)
    d_std = np.std(diffs, ddof=1)
    cohens_d = d_mean / d_std if d_std > 0 else 0.0

    return {
        "metric": metric_name,
        "n_folds": n,
        "model_mean": float(np.mean(metric_model)),
        "baseline_mean": float(np.mean(metric_baseline)),
        "diff_mean": float(d_mean),
        "t_stat": float(t_stat) if np.isfinite(t_stat) else 0.0,
        "p_value": float(p_value) if np.isfinite(p_value) else 1.0,
        "significant_005": bool(p_value < 0.05) if np.isfinite(p_value) else False,
        "significant_010": bool(p_value < 0.10) if np.isfinite(p_value) else False,
        "cohens_d": float(cohens_d),
    }


def diebold_mariano_test(
    actual: np.ndarray,
    pred_model: np.ndarray,
    pred_baseline: np.ndarray,
    loss_fn: str = "mse",
) -> Dict:
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Tests H0: equal predictive accuracy.
    H1: model has lower loss than baseline.

    Args:
        actual: (N,) true values.
        pred_model: (N,) model predictions.
        pred_baseline: (N,) baseline predictions.
        loss_fn: "mse" or "mae".

    Returns:
        Dict with DM statistic, p-value.
    """
    if loss_fn == "mse":
        e_model = (actual - pred_model) ** 2
        e_baseline = (actual - pred_baseline) ** 2
    elif loss_fn == "mae":
        e_model = np.abs(actual - pred_model)
        e_baseline = np.abs(actual - pred_baseline)
    else:
        raise ValueError(f"Unknown loss_fn: {loss_fn}")

    d = e_baseline - e_model  # Positive if model is better

    n = len(d)
    if n < 10:
        return {
            "dm_stat": 0.0,
            "p_value": 1.0,
            "model_better": False,
            "note": "Need ≥10 samples",
        }

    d_mean = np.mean(d)
    # Newey-West variance estimator (lag = int(n^(1/3)))
    max_lag = int(np.ceil(n ** (1 / 3)))
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for k in range(1, max_lag + 1):
        weight = 1 - k / (max_lag + 1)
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        gamma_sum += 2 * weight * gamma_k
    var_d = (gamma_0 + gamma_sum) / n

    if var_d <= 0:
        return {"dm_stat": 0.0, "p_value": 1.0, "model_better": False}

    dm_stat = d_mean / np.sqrt(var_d)
    # One-sided: model is better (d_mean > 0)
    p_value = 1 - sp_stats.norm.cdf(dm_stat)

    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "model_better": bool(p_value < 0.05),
        "loss_fn": loss_fn,
        "n_samples": n,
    }


def compare_models(
    fold_results_model: List[Dict],
    fold_results_baseline: List[Dict],
    model_name: str = "TGT",
    baseline_name: str = "Baseline",
) -> Dict:
    """
    Compare TGT against a baseline across walk-forward folds.

    Runs paired t-tests on direction accuracy, return MSE, and volatility MAE.
    Runs Diebold-Mariano on pooled return predictions.

    Returns:
        Dict with all comparison results.
    """
    model_metrics = [evaluate_fold(r) for r in fold_results_model]
    baseline_metrics = [evaluate_fold(r) for r in fold_results_baseline]

    comparisons = {}

    # Paired t-tests on per-fold metrics
    comparisons["direction_accuracy"] = paired_t_test(
        [m["direction"]["accuracy"] for m in model_metrics],
        [m["direction"]["accuracy"] for m in baseline_metrics],
        "direction_accuracy",
    )
    comparisons["return_mse"] = paired_t_test(
        # Negate so "higher is better" convention holds
        [-m["return"]["mse"] for m in model_metrics],
        [-m["return"]["mse"] for m in baseline_metrics],
        "return_mse (negated, higher=better)",
    )
    comparisons["volatility_mae"] = paired_t_test(
        [-m["volatility"]["mae"] for m in model_metrics],
        [-m["volatility"]["mae"] for m in baseline_metrics],
        "volatility_mae (negated, higher=better)",
    )

    # Diebold-Mariano on pooled predictions
    model_returns_pred = np.concatenate([r["test_predictions"]["return"] for r in fold_results_model])
    baseline_returns_pred = np.concatenate([r["test_predictions"]["return"] for r in fold_results_baseline])
    actual_returns = np.concatenate([r["test_targets"]["return"] for r in fold_results_model])

    comparisons["dm_return_mse"] = diebold_mariano_test(
        actual_returns, model_returns_pred, baseline_returns_pred, "mse",
    )
    comparisons["dm_return_mae"] = diebold_mariano_test(
        actual_returns, model_returns_pred, baseline_returns_pred, "mae",
    )

    comparisons["model_name"] = model_name
    comparisons["baseline_name"] = baseline_name

    return comparisons


# ===========================================================================
# 4. Pretty Printing
# ===========================================================================

def print_fold_report(metrics: Dict):
    """Print a detailed report for a single fold."""
    fold_id = metrics["fold_id"]
    d = metrics["direction"]
    r = metrics["return"]
    v = metrics["volatility"]

    print(f"\n{'─' * 50}")
    print(f"📊 Fold {fold_id} Evaluation (best epoch: {metrics['best_epoch']})")
    print(f"{'─' * 50}")

    # Direction
    print(f"\n  Direction Classification ({d['n_samples']} samples)")
    print(f"    Accuracy:    {d['accuracy']:.3f}")
    print(f"    Macro F1:    {d['macro_f1']:.3f}")
    print(f"    Weighted F1: {d['weighted_f1']:.3f}")
    print(f"    Confidence:  {d['mean_confidence']:.3f}")
    print(f"    {'Class':>8s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'N':>5s}")
    for cls_name, cls_m in d["per_class"].items():
        print(f"    {cls_name:>8s}  {cls_m['precision']:6.3f}  {cls_m['recall']:6.3f}  "
              f"{cls_m['f1']:6.3f}  {cls_m['support']:5d}")

    # Confusion matrix
    cm = d["confusion_matrix"]
    print(f"\n    Confusion Matrix (rows=true, cols=pred)")
    print(f"    {'':>8s}  {'Down':>6s}  {'Flat':>6s}  {'Up':>6s}")
    for i, label in DIRECTION_LABELS.items():
        print(f"    {label:>8s}  {cm[i,0]:6d}  {cm[i,1]:6d}  {cm[i,2]:6d}")

    # Returns
    print(f"\n  Return Prediction ({r['n_samples']} samples)")
    print(f"    MSE:   {r['mse']:.6f}")
    print(f"    RMSE:  {r['rmse']:.6f}")
    print(f"    MAE:   {r['mae']:.6f}")
    print(f"    R²:    {r['r2']:.4f}")
    print(f"    Dir agreement: {r['directional_agreement']:.3f}")
    print(f"    IC (Spearman):  {r['ic']:.3f} (p={r['ic_pval']:.4f})")

    # Volatility
    print(f"\n  Volatility Prediction ({v['n_samples']} samples)")
    print(f"    MAE:   {v['mae']:.6f}")
    print(f"    RMSE:  {v['rmse']:.6f}")
    print(f"    R²:    {v['r2']:.4f}")
    print(f"    QLIKE: {v['qlike']:.4f}")


def print_aggregate_report(agg: Dict):
    """Print aggregated walk-forward results."""
    n = agg["n_folds"]
    d = agg["direction"]
    r = agg["return"]
    v = agg["volatility"]

    print(f"\n{'=' * 60}")
    print(f"📊 Walk-Forward Aggregate Results ({n} folds)")
    print(f"{'=' * 60}")

    print(f"\n  Direction Classification")
    print(f"    Accuracy:     {d['accuracy_mean']:.3f} ± {d['accuracy_std']:.3f}")
    print(f"    Macro F1:     {d['macro_f1_mean']:.3f} ± {d['macro_f1_std']:.3f}")
    print(f"    Weighted F1:  {d['weighted_f1_mean']:.3f} ± {d['weighted_f1_std']:.3f}")
    print(f"    Pooled acc:   {d['pooled_accuracy']:.3f}")

    cm = d["pooled_confusion_matrix"]
    print(f"\n    Pooled Confusion Matrix")
    print(f"    {'':>8s}  {'Down':>6s}  {'Flat':>6s}  {'Up':>6s}")
    for i, label in DIRECTION_LABELS.items():
        print(f"    {label:>8s}  {cm[i,0]:6d}  {cm[i,1]:6d}  {cm[i,2]:6d}")

    print(f"\n  Return Prediction")
    print(f"    MSE:  {r['mse_mean']:.6f} ± {r['mse_std']:.6f}")
    print(f"    RMSE: {r['rmse_mean']:.6f} ± {r['rmse_std']:.6f}")
    print(f"    R²:   {r['r2_mean']:.4f} ± {r['r2_std']:.4f}")
    print(f"    Dir agreement: {r['directional_agreement_mean']:.3f} ± {r['directional_agreement_std']:.3f}")
    print(f"    IC:   {r['ic_mean']:.3f} ± {r['ic_std']:.3f}")

    print(f"\n  Volatility Prediction")
    print(f"    MAE:  {v['mae_mean']:.6f} ± {v['mae_std']:.6f}")
    print(f"    RMSE: {v['rmse_mean']:.6f} ± {v['rmse_std']:.6f}")
    print(f"    R²:   {v['r2_mean']:.4f} ± {v['r2_std']:.4f}")
    print(f"    QLIKE: {v['qlike_mean']:.4f} ± {v['qlike_std']:.4f}")


def print_comparison_report(comp: Dict):
    """Print model vs baseline comparison."""
    print(f"\n{'=' * 60}")
    print(f"📊 {comp['model_name']} vs {comp['baseline_name']}")
    print(f"{'=' * 60}")

    for key in ["direction_accuracy", "return_mse", "volatility_mae"]:
        t = comp[key]
        sig = "✅ p<0.05" if t["significant_005"] else ("⚠️ p<0.10" if t["significant_010"] else "❌ n.s.")
        print(f"\n  {t['metric']}")
        print(f"    Model:    {t.get('model_mean', 0):.4f}")
        print(f"    Baseline: {t.get('baseline_mean', 0):.4f}")
        print(f"    Diff:     {t.get('diff_mean', 0):+.4f}")
        print(f"    t={t['t_stat']:.3f}, p={t['p_value']:.4f}  {sig}")
        print(f"    Cohen's d: {t['cohens_d']:.3f}")

    for key in ["dm_return_mse", "dm_return_mae"]:
        dm = comp[key]
        sig = "✅" if dm["model_better"] else "❌"
        print(f"\n  Diebold-Mariano ({dm.get('loss_fn', key)})")
        print(f"    DM stat: {dm['dm_stat']:.3f}, p={dm['p_value']:.4f}  {sig}")


# ===========================================================================
# 5. Utility
# ===========================================================================

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ===========================================================================
# 6. Smoke Test
# ===========================================================================

def test_evaluator():
    """Smoke test with synthetic fold results."""
    np.random.seed(42)

    print("=" * 60)
    print("🧪 Evaluator Smoke Test")
    print("=" * 60)

    def make_fake_fold(fold_id, acc_bias=0.0):
        n = 100
        y_true_dir = np.random.randint(0, 3, n)
        # Predictions with some accuracy
        logits = np.random.randn(n, 3)
        logits[np.arange(n), y_true_dir] += 1.0 + acc_bias  # Bias toward correct class

        y_true_ret = np.random.randn(n) * 0.01
        y_pred_ret = y_true_ret + np.random.randn(n) * 0.005

        y_true_vol = np.abs(np.random.randn(n) * 0.005)
        y_pred_vol = y_true_vol + np.abs(np.random.randn(n) * 0.002)

        return {
            "fold_id": fold_id,
            "best_epoch": 20 + fold_id,
            "test_losses": {"total": 0.3 - fold_id * 0.01},
            "test_direction_accuracy": 0.0,  # Will be recomputed
            "test_predictions": {
                "direction": logits,
                "return": y_pred_ret,
                "volatility": y_pred_vol,
            },
            "test_targets": {
                "direction": y_true_dir,
                "return": y_true_ret,
                "volatility": y_true_vol,
            },
        }

    # Test per-fold evaluation
    print("\n1️⃣  Testing per-fold metrics...")
    fold_results = [make_fake_fold(i) for i in range(5)]
    fold_metrics = [evaluate_fold(r) for r in fold_results]

    for m in fold_metrics:
        print_fold_report(m)
    print("   ✅ Per-fold metrics OK")

    # Test aggregation
    print("\n2️⃣  Testing cross-fold aggregation...")
    agg = aggregate_fold_metrics(fold_metrics)
    print_aggregate_report(agg)
    assert agg["n_folds"] == 5
    assert 0 < agg["direction"]["accuracy_mean"] < 1
    print("   ✅ Aggregation OK")

    # Test significance tests
    print("\n3️⃣  Testing statistical tests...")
    baseline_results = [make_fake_fold(i, acc_bias=-0.5) for i in range(5)]
    comp = compare_models(fold_results, baseline_results, "TGT", "WeakBaseline")
    print_comparison_report(comp)
    print("   ✅ Significance tests OK")

    # Test Diebold-Mariano standalone
    print("\n4️⃣  Testing Diebold-Mariano...")
    actual = np.random.randn(200) * 0.01
    pred_good = actual + np.random.randn(200) * 0.003
    pred_bad = actual + np.random.randn(200) * 0.01
    dm = diebold_mariano_test(actual, pred_good, pred_bad, "mse")
    print(f"   DM stat={dm['dm_stat']:.3f}, p={dm['p_value']:.4f}, better={dm['model_better']}")
    print("   ✅ Diebold-Mariano OK")

    print(f"\n✅ All evaluator tests passed!")


if __name__ == "__main__":
    test_evaluator()
